import argparse
import os
import json
from typing import List, Tuple, Optional

import numpy as np
import torch
import sys

# Ensure we can import modules from src/
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(REPO_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from model import FlowMatchingPolicy, VAE, NoiseToAction  # type: ignore
except ImportError:  # Defensive: older checkouts may not expose VAE/NoiseToAction
    from model import FlowMatchingPolicy  # type: ignore
    VAE = None  # type: ignore
    NoiseToAction = None  # type: ignore

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_policy(policy_path: str, obs_dim: int, action_dim: int):
    """Load FlowMatchingPolicy with config from checkpoint."""
    ckpt = torch.load(policy_path, map_location=device)
    if not (isinstance(ckpt, dict) and 'model_state_dict' in ckpt):
        raise ValueError("Checkpoint missing 'model_state_dict'. Retrain with updated train_flow.py")

    obs_hidden_dims = ckpt.get('obs_hidden_dims', [128, 128])
    policy_hidden_dims = ckpt.get('policy_hidden_dims', [128, 128])
    dropout_p = ckpt.get('dropout_p', 0.0)
    layernorm = ckpt.get('layernorm', False)
    arch = ckpt.get('arch', 'mlp')
    unet_base_ch = ckpt.get('unet_base_ch', 64)
    unet_depth = ckpt.get('unet_depth', 3)

    policy = FlowMatchingPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon=1,
        obs_hidden=obs_hidden_dims,
        policy_hidden=policy_hidden_dims,
        dropout_p=dropout_p,
        layernorm=layernorm,
        arch=arch,
        unet_base_ch=unet_base_ch,
        unet_depth=unet_depth,
    ).to(device)
    # Handle backward compatibility
    state_dict = ckpt['model_state_dict']

    # 1) Rename old obs encoder if needed
    if 'obs_encoder.0.weight' in state_dict and 'obs_encoder_mlp.0.weight' not in state_dict:
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith('obs_encoder.'):
                new_state[k.replace('obs_encoder.', 'obs_encoder_mlp.')] = v
            else:
                new_state[k] = v
        state_dict = new_state

    # 2) Add dummy time_mlp weights for MLP checkpoints that didn't have them
    #    (MLP forward doesn't use time_mlp, but model definition includes it.)
    if arch == 'mlp' and not any(k.startswith('time_mlp.') for k in state_dict.keys()):
        import torch.nn as nn
        time_mlp = nn.Sequential(
            nn.Linear(64, obs_hidden_dims[-1]),
            nn.SiLU(),
            nn.Linear(obs_hidden_dims[-1], obs_hidden_dims[-1])
        )
        tm_state = time_mlp.state_dict()
        for k, v in tm_state.items():
            state_dict[f'time_mlp.{k}'] = v

    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def _infer_mlp_hidden_dims(state_dict: dict, prefix: str) -> tuple[list[int], bool, bool]:
    """Infer hidden layer widths, layernorm, and dropout usage for sequential MLPs."""
    linear_entries: list[tuple[int, int, int]] = []
    layernorm_present = False
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        if key.endswith(".weight") and isinstance(value, torch.Tensor):
            if value.ndim == 2:
                try:
                    idx = int(key.split('.')[1])
                except (IndexError, ValueError):
                    continue
                linear_entries.append((idx, value.shape[1], value.shape[0]))
            elif value.ndim == 1:
                layernorm_present = True
    if not linear_entries:
        raise ValueError("Unable to infer hidden dims from state dict")
    linear_entries.sort(key=lambda x: x[0])
    hidden = [out_dim for (_, _, out_dim) in linear_entries[:-1]]
    if not hidden:
        hidden = [linear_entries[-1][1]]  # Degenerate single-layer; mirror input width

    dropout_present = False
    if len(linear_entries) >= 2:
        indices = [item[0] for item in linear_entries]
        for i in range(len(indices) - 1):
            step = indices[i + 1] - indices[i]
            modules_between = max(0, step - 1)
            # Baseline module count always includes the activation (1) and optional layernorm (1)
            baseline = 1 + (1 if layernorm_present else 0)
            if modules_between > baseline:
                dropout_present = True
                break
            # When layernorm absent, a modules_between of 2 implies dropout (activation + dropout)
            if not layernorm_present and modules_between >= 2:
                dropout_present = True
                break
    return hidden, layernorm_present, dropout_present


def _attach_optional_stats(model, metadata: dict) -> None:
    for attr in ("output_mean", "output_std", "clip_low", "clip_high"):
        if attr in metadata:
            setattr(model, attr, metadata[attr])


def load_pretrain_model(pretrain_path: str, action_dim: int):
    """Load a pretrain prior model (NoiseToAction/VAE) for x0 sampling."""
    ckpt = torch.load(pretrain_path, map_location=device)

    # Handle direct torch.nn.Module saves
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt.to(device)
        model.eval()
        return model

    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported pretrain checkpoint format; expected dict or nn.Module")

    metadata = {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
    state_dict = ckpt.get('model_state_dict')
    if state_dict is None:
        raise ValueError("Pretrain checkpoint missing 'model_state_dict'")

    # Guess model type
    model_type = str(metadata.get('model_type', metadata.get('arch', ''))).lower()
    if not model_type:
        keys = list(state_dict.keys())
        if any(k.startswith('decoder') for k in keys):
            model_type = 'vae'
        elif any(k.startswith('net.') for k in keys):
            model_type = 'noise_to_action'
        else:
            raise ValueError("Unable to infer model type from checkpoint; please specify metadata")

    if model_type in ('vae', 'flow_vae'):
        if VAE is None:
            raise ImportError("VAE class unavailable; cannot load pretrain model")
        hidden_dims = metadata.get('hidden_dims') or metadata.get('decoder_hidden_dims')
        if hidden_dims is None:
            hidden_dims, _, _ = _infer_mlp_hidden_dims(state_dict, prefix='decoder')
        latent_dim = int(metadata.get('latent_dim', 64))
        dropout_p = float(metadata.get('dropout_p', 0.0) or 0.0)
        layernorm = bool(metadata.get('layernorm', False))
        squash = bool(metadata.get('squash_actions', metadata.get('squash', False)))
        model = VAE(
            input_dim=action_dim,
            hidden_dims=list(map(int, hidden_dims)),
            latent_dim=latent_dim,
            dropout_p=dropout_p,
            layernorm=layernorm,
            squash_actions=squash,
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️  VAE checkpoint missing keys: {missing}")
        if unexpected:
            print(f"⚠️  VAE checkpoint unexpected keys: {unexpected}")
        _attach_optional_stats(model, metadata)
        model = model.to(device)
        model.eval()
        return model

    if model_type in ('noise_to_action', 'mlp', 'n2a'):
        if NoiseToAction is None:
            raise ImportError("NoiseToAction class unavailable; cannot load pretrain model")
        hidden_dims = metadata.get('hidden_dims') or metadata.get('policy_hidden_dims')
        inferred_layernorm = False
        dropout_present = False
        if hidden_dims is None:
            hidden_dims, inferred_layernorm, dropout_present = _infer_mlp_hidden_dims(state_dict, prefix='net')
        dropout_p = float(metadata.get('dropout_p', 0.0) or 0.0)
        layernorm = bool(metadata.get('layernorm', inferred_layernorm))
        if dropout_present and dropout_p <= 0.0:
            # Maintain architecture indices; keep dropout layer but default prob 0.
            dropout_p = float(metadata.get('dropout', 0.0) or 0.0)
            if dropout_p <= 0.0:
                dropout_p = 1e-8
        squash = bool(metadata.get('squash_actions', metadata.get('squash', False)))
        model = NoiseToAction(
            input_dim=action_dim,
            hidden_dims=list(map(int, hidden_dims)),
            dropout_p=dropout_p,
            layernorm=layernorm,
            squash_actions=squash,
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️  NoiseToAction checkpoint missing keys: {missing}")
        if unexpected:
            print(f"⚠️  NoiseToAction checkpoint unexpected keys: {unexpected}")
        _attach_optional_stats(model, metadata)
        model = model.to(device)
        model.eval()
        return model

    raise ValueError(f"Unsupported pretrain model type '{model_type}'")


class PretrainSampler:
    """Callable wrapper to draw normalized x0 samples from a pretrain model."""

    def __init__(self, model):
        self.model = model.to(device)
        self.model.eval()

    def __call__(self, batch_size: int) -> torch.Tensor:
        with torch.no_grad():
            samples = self.model.sample(batch_size, device=device)
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
        if not torch.is_tensor(samples):
            raise ValueError("Pretrain model sample() must return Tensor or ndarray")
        samples = samples.to(device=device, dtype=torch.float32)
        samples = samples.view(batch_size, -1)
        return samples


def parse_schedule(arg: str) -> Tuple[List[float], List[float]]:
    """Return (ts, dts) with ts[0]==0 and ts[-1]==1. Accepts:
    - int-like string 'N' => uniform
    - list string '[a,b,c]' => interior points; 0 and 1 will be anchored automatically
    """
    s = str(arg).strip()
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1].strip()
        pts = [float(x) for x in inner.split(',') if x.strip() != '']
        interior = []
        for v in pts:
            if v <= 0.0 or v >= 1.0:
                if np.isclose(v, 0.0) or np.isclose(v, 1.0):
                    continue
                raise ValueError("Interior points must lie strictly between 0 and 1")
            interior.append(float(v))
        # unique + sort
        seen = set()
        u = []
        for v in interior:
            if v not in seen:
                seen.add(v)
                u.append(v)
        interior = sorted(u)
        ts = [0.0] + interior + [1.0]
    else:
        try:
            n = int(float(s))
        except Exception:
            raise ValueError("flow schedule must be int N or list like [0.1,0.5]")
        n = max(1, n)
        ts = np.linspace(0.0, 1.0, n + 1).tolist()
    if not np.isclose(ts[0], 0.0) or not np.isclose(ts[-1], 1.0):
        raise ValueError("Schedule must start at 0 and end at 1")
    for i in range(1, len(ts)):
        if not (ts[i] > ts[i-1]):
            raise ValueError("Schedule must be strictly increasing")
    dts = [ts[i+1] - ts[i] for i in range(len(ts)-1)]
    return ts, dts


def try_import_sklearn_knn():
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        return NearestNeighbors
    except Exception:
        return None


def knn_1(train_actions: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Return indices of nearest neighbor in train_actions for each query.
    train_actions: (N, D), queries: (Q, D)
    """
    NN = try_import_sklearn_knn()
    if NN is not None:
        nbrs = NN(n_neighbors=1, algorithm='auto').fit(train_actions)
        dists, indices = nbrs.kneighbors(queries)
        return indices[:, 0]
    # Fallback: brute force in batches to save memory
    Q, D = queries.shape
    N = train_actions.shape[0]
    idxs = np.empty(Q, dtype=np.int64)
    bs = 8192
    for i in range(0, Q, bs):
        q = queries[i:i+bs]
        # (q,1,d) - (1,n,d) => (q,n,d)
        # Use chunking over N if needed
        best = None
        best_idx = None
        stride = 20000
        for j in range(0, N, stride):
            ta = train_actions[j:j+stride]
            d2 = ((q[:, None, :] - ta[None, :, :]) ** 2).sum(axis=2)
            b_idx = d2.argmin(axis=1)
            b_val = d2[np.arange(d2.shape[0]), b_idx]
            if best is None:
                best = b_val
                best_idx = b_idx + j
            else:
                mask = b_val < best
                best[mask] = b_val[mask]
                best_idx[mask] = b_idx[mask] + j
        idxs[i:i+bs] = best_idx
    return idxs


class KNNIndex:
    """Helper wrapper to use a prebuilt sklearn NearestNeighbors index (.joblib/.pkl).

    If provided, queries are expected in RAW action space matching the index training data.
    We expose a method to query with normalized vectors by internally de-normalizing and
    then re-normalizing neighbors back to normalized space using (act_mean, act_std).
    """
    def __init__(self, path: Optional[str] = None):
        self.nn = None
        if path is not None and os.path.exists(path):
            try:
                import joblib  # type: ignore
                self.nn = joblib.load(path)
                print(f"✅ Loaded prebuilt KNN index: {path}")
            except Exception as e:
                print(f"⚠️  Failed to load KNN index '{path}': {e}")
                self.nn = None

    def have_index(self) -> bool:
        return self.nn is not None and hasattr(self.nn, 'kneighbors') and hasattr(self.nn, '_fit_X')

    def query_norm(self, queries_norm: np.ndarray, act_mean: np.ndarray, act_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (idxs, neighbors_norm) for 1-NN given normalized queries.

        If no index, returns (None, None).
        """
        if not self.have_index():
            return None, None
        # De-normalize queries to raw space to match index space
        q_raw = queries_norm * act_std + act_mean
        idxs = self.nn.kneighbors(q_raw, n_neighbors=1, return_distance=False)[:, 0]
        # Recover neighbors in raw space from fitted corpus
        corpus_raw = getattr(self.nn, '_fit_X')
        neigh_raw = corpus_raw[idxs]
        neigh_norm = (neigh_raw - act_mean) / act_std
        return idxs, neigh_norm


def main():
    parser = argparse.ArgumentParser(description="Validate Flow Matching velocity field behavior")
    parser.add_argument('--data_path', type=str, required=True, help='Expert data file path')
    parser.add_argument('--policy_path', type=str, required=True, help='Trained policy checkpoint (.pth)')
    parser.add_argument('--num_train', type=int, required=True, help='Number of training samples; remainder used as validation')
    parser.add_argument('--euler_steps', type=str, default='[0.1,0.5,0.9]', help='Interior points, e.g., [0.1,0.5] or int N')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--knn_index', type=str, default='knn_index.joblib', help='Path to prebuilt KNN index (joblib/pkl). If missing, falls back to on-the-fly KNN.')
    parser.add_argument('--pretrain_model', type=str, default=None,
                        help='Optional NoiseToAction/VAE checkpoint to sample x0 from. Defaults to standard Normal when omitted.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--obs_noise_std', type=float, default=0.0, help='Gaussian noise std applied to normalized obs')
    parser.add_argument('--random_obs', action='store_true', help='Replace obs with random obs from training (diagnostic)')
    parser.add_argument('--jump', action='store_true', help='From each t, take a single Euler jump to t=1 and use the final point for KNN and velocity metrics')
    parser.add_argument('--validate_training', action='store_true',
                        help='Use the TRAINING subset as validation (i.e., validate on the first num_train samples)')
    parser.add_argument('--a_t', dest='a_t_mode', choices=['linear', 'euler'], default='linear',
                        help="How to construct a_t: 'linear' uses t*a1+(1-t)*x0; 'euler' integrates from x0 via Euler steps using policy v_hat over the schedule up to current t")
    parser.add_argument('--num_val', type=int, default=10000,
                        help='Number of validation samples. Uses the LAST N samples of the dataset for validation')
    # Plotting options
    parser.add_argument('--plot_cos', action='store_true',
                        help='If set, plot publication-quality curves for cos(v_hat, v_true) and cos(v_hat, v_knn) over t')
    parser.add_argument('--plot_out', type=str, default='./results/velocity_cosine',
                        help='Output path prefix for plots (saves .png and .pdf). Directory is created if missing')
    parser.add_argument('--plot_dpi', type=int, default=300, help='DPI for PNG export (default: 300)')
    parser.add_argument('--knn_on_a1', action='store_true',
                        help='Query KNN using a1 (final action) instead of a_t/a_eval at each t')
    args = parser.parse_args()

    set_seed(args.seed)

    # Reuse the canonical loader from train_flow.py
    from train_flow import load_data as _load_demo_data
    data_dict = _load_demo_data(args.data_path)
    obs_all = data_dict['observations']
    act_all = data_dict['actions']
    N = len(obs_all)
    if args.num_train <= 0 or args.num_train >= N:
        raise ValueError('num_train must be within (0, N)')

    # Compute stats on TRAINING subset only (to match training regime)
    train_obs = obs_all[:args.num_train]
    train_act = act_all[:args.num_train]
    obs_mean = train_obs.mean(axis=0)
    obs_std = train_obs.std(axis=0)
    act_mean = train_act.mean(axis=0)
    act_std = train_act.std(axis=0)
    eps = 1e-8
    obs_std = np.maximum(obs_std, eps)
    act_std = np.maximum(act_std, eps)

    # Normalize
    train_obs_n = (train_obs - obs_mean) / obs_std
    train_act_n = (train_act - act_mean) / act_std
    # Select validation set: either last N samples if --num_val provided, or remainder after training
    if args.num_val is not None and args.num_val > 0:
        n_val = min(args.num_val, N)
        # Warn if overlap with training region
        if args.num_train > N - n_val:
            print(f"⚠️  Validation set overlaps with training set (train={args.num_train}, val_last={n_val}, N={N})")
        val_obs = obs_all[-n_val:]
        val_act = act_all[-n_val:]
        print(f"📊 Using last {len(val_obs)} samples as validation")
    else:
        val_obs = obs_all[args.num_train:]
        val_act = act_all[args.num_train:]
    val_obs_n = (val_obs - obs_mean) / obs_std
    val_act_n = (val_act - act_mean) / act_std

    # Optionally validate on training subset
    if args.validate_training:
        print("ℹ️  Using TRAINING subset as validation (--validate_training enabled)")
        val_obs_n = train_obs_n
        val_act_n = train_act_n

    obs_dim = obs_all.shape[1] if len(obs_all.shape) > 1 else 1
    act_dim = act_all.shape[1] if len(act_all.shape) > 1 else 1

    # Load policy
    policy = load_policy(args.policy_path, obs_dim=obs_dim, action_dim=act_dim)

    # Optional pretrain sampler for x0
    pretrain_sampler: PretrainSampler | None = None
    if args.pretrain_model is not None:
        if not os.path.exists(args.pretrain_model):
            raise FileNotFoundError(f"Pretrain model not found: {args.pretrain_model}")
        pretrain_model = load_pretrain_model(args.pretrain_model, action_dim=act_dim)
        if not hasattr(pretrain_model, 'sample'):
            raise ValueError("Pretrain model must expose a sample(num_samples, device=...) method")
        pretrain_sampler = PretrainSampler(pretrain_model)
        print(f"✅ Using pretrain prior for x0 sampling: {args.pretrain_model}")

    # Time schedule
    ts, dts = parse_schedule(args.euler_steps)
    ts_np = np.array(ts, dtype=np.float32)

    # Prepare KNN
    knn_actions = train_act_n.astype(np.float32)  # normalized actions for fallback
    knn_index = KNNIndex(args.knn_index)

    # Metrics accumulators per t
    per_t = {
        'cos_to_true': [0.0 for _ in range(len(dts))],
        'cos_to_knn': [0.0 for _ in range(len(dts))],
        'err_norm': [0.0 for _ in range(len(dts))],
        'dist_to_knn': [0.0 for _ in range(len(dts))],  # mean ||v_hat - v_knn||
        'dist_ak_a1': [0.0 for _ in range(len(dts))],  # mean ||a_k - a_1|| (normalized)
        'dist_ahat_ak': [0.0 for _ in range(len(dts))],  # mean ||a^ - a_k|| where a^ is Euler-predicted action
        'dist_ahat_a1': [0.0 for _ in range(len(dts))],  # mean ||a^ - a_1||
        'count': 0,
    }

    # For K evaluation times (ts[:-1]), we have K-1 transitions to test NN consistency
    K = len(ts) - 1
    nn_same_num = [0 for _ in range(max(0, K - 1))]
    nn_same_den = [0 for _ in range(max(0, K - 1))]

    def cosine(a: torch.Tensor, b: torch.Tensor, eps=1e-8) -> torch.Tensor:
        an = torch.linalg.norm(a, dim=1).clamp_min(eps)
        bn = torch.linalg.norm(b, dim=1).clamp_min(eps)
        return (a * b).sum(dim=1) / (an * bn)

    B = args.batch_size
    M = val_obs_n.shape[0]
    rng = np.random.default_rng(args.seed)
    for i0 in range(0, M, B):
        sl = slice(i0, min(i0+B, M))
        obs_b = val_obs_n[sl].astype(np.float32)
        act_b = val_act_n[sl].astype(np.float32)

        if args.random_obs:
            # Replace with random training obs
            idx = rng.integers(0, train_obs_n.shape[0], size=obs_b.shape[0])
            obs_b = train_obs_n[idx]
        if args.obs_noise_std > 0.0:
            obs_b = obs_b + rng.normal(0.0, args.obs_noise_std, size=obs_b.shape).astype(np.float32)

        obs_t = torch.from_numpy(obs_b).to(device)
        a1_t = torch.from_numpy(act_b).to(device)

        # Sample x0 (normalized space)
        if pretrain_sampler is not None:
            x0_t = pretrain_sampler(obs_t.shape[0])
            if x0_t.shape != a1_t.shape:
                raise ValueError(
                    f"Pretrain sample shape {tuple(x0_t.shape)} does not match action shape {tuple(a1_t.shape)}"
                )
        else:
            x0_t = torch.randn_like(a1_t)

        # True velocity (a1 - x0), constant over t
        v_true = a1_t - x0_t

        # For each t, build a_t and evaluate
        prev_idxs_batch = None
        # For a_t='euler', maintain running a from x0 and integrate along the schedule
        if args.a_t_mode == 'euler':
            a_running = x0_t.clone()
        for ti, tval in enumerate(ts[:-1]):
            t_tensor = torch.full((obs_t.shape[0], 1), float(tval), device=device)
            if args.a_t_mode == 'euler':
                # Use integrated action at current time
                a_t = a_running
            else:
                # Linear interpolation between x0 and a1
                a_t = float(tval) * a1_t + (1.0 - float(tval)) * x0_t

            with torch.no_grad():
                v_hat = policy(obs_t, a_t, t_tensor)

            # Compute one-step Euler prediction a^ (a_hat) to t=1 from current t
            dt_to_one = 1.0 - float(tval)
            a_hat = a_t + dt_to_one * v_hat

            # Evaluation point for existing metrics (KNN + velocity): jump uses a_hat; otherwise a_t
            if args.jump:
                a_eval = a_hat
                v_hat_eval = a_eval - x0_t
            else:
                a_eval = a_t
                v_hat_eval = v_hat

            # KNN on chosen query vs training actions
            # If --knn_on_a1, use a1 as the KNN query (time-invariant); otherwise use a_eval (a_t or a_hat)
            knn_query = a1_t if args.knn_on_a1 else a_eval
            a_eval_np = knn_query.detach().cpu().numpy()
            if knn_index.have_index():
                idxs_curr, neigh_norm = knn_index.query_norm(a_eval_np, act_mean=act_mean, act_std=act_std)
                a_k = torch.from_numpy(neigh_norm.astype(np.float32)).to(device)
            else:
                idxs_curr = knn_1(knn_actions, a_eval_np)
                a_k = torch.from_numpy(knn_actions[idxs_curr]).to(device)
            v_knn = a_k - x0_t

            # KNN for a_hat specifically (for new distances). Reuse if identical to a_eval
            if args.jump:
                a_k_hat = a_k
            else:
                a_hat_np = a_hat.detach().cpu().numpy()
                if knn_index.have_index():
                    _idxs_hat, neigh_norm_hat = knn_index.query_norm(a_hat_np, act_mean=act_mean, act_std=act_std)
                    a_k_hat = torch.from_numpy(neigh_norm_hat.astype(np.float32)).to(device)
                else:
                    _idxs_hat = knn_1(knn_actions, a_hat_np)
                    a_k_hat = torch.from_numpy(knn_actions[_idxs_hat]).to(device)

            # Metrics (computed on v_hat_eval)
            cos_true = cosine(v_hat_eval, v_true).mean().item()
            cos_knn = cosine(v_hat_eval, v_knn).mean().item()
            err = torch.linalg.norm(v_hat_eval - v_true, dim=1).mean().item()
            dist_knn = torch.linalg.norm(v_hat_eval - v_knn, dim=1).mean().item()
            dist_ak_a1 = torch.linalg.norm(a_k - a1_t, dim=1).mean().item()

            per_t['cos_to_true'][ti] += cos_true * obs_t.shape[0]
            per_t['cos_to_knn'][ti] += cos_knn * obs_t.shape[0]
            per_t['err_norm'][ti] += err * obs_t.shape[0]
            per_t['dist_to_knn'][ti] += dist_knn * obs_t.shape[0]
            per_t['dist_ak_a1'][ti] += dist_ak_a1 * obs_t.shape[0]
            # New accumulators involving a_hat
            dist_ahat_ak = torch.linalg.norm(a_hat - a_k_hat, dim=1).mean().item()
            dist_ahat_a1 = torch.linalg.norm(a_hat - a1_t, dim=1).mean().item()
            per_t['dist_ahat_ak'][ti] += dist_ahat_ak * obs_t.shape[0]
            per_t['dist_ahat_a1'][ti] += dist_ahat_a1 * obs_t.shape[0]

            # NN consistency across consecutive t within this batch
            if prev_idxs_batch is not None and (K - 1) > 0:
                trans = ti - 1
                if trans >= 0:
                    same = (idxs_curr == prev_idxs_batch)
                    nn_same_num[trans] += int(np.sum(same))
                    nn_same_den[trans] += same.shape[0]
            prev_idxs_batch = idxs_curr

            # If using Euler a_t integration, advance to next time using current velocity
            if args.a_t_mode == 'euler':
                local_dt = float(dts[ti]) if ti < len(dts) else 0.0
                a_running = a_t + local_dt * v_hat
        per_t['count'] += obs_t.shape[0]

    # Aggregate
    C = per_t['count']
    print("==== Velocity Field Validation ====")
    print(f"Samples (validation): {C}")
    print(f"Flow schedule t (interior): {ts_np[1:-1].tolist()}")
    for ti, tval in enumerate(ts[:-1]):
        ct = per_t['cos_to_true'][ti] / C
        ck = per_t['cos_to_knn'][ti] / C
        en = per_t['err_norm'][ti] / C
        dk = per_t['dist_to_knn'][ti] / C
        dak = per_t['dist_ak_a1'][ti] / C
        dahk = per_t['dist_ahat_ak'][ti] / C
        dah1 = per_t['dist_ahat_a1'][ti] / C
        print(f"t={tval:.3f} | cos(v_hat, v_true)={ct:.4f} | cos(v_hat, v_knn)={ck:.4f} | ||v_hat-v_true||={en:.4f} | ||v_hat-v_knn||={dk:.4f} | ||a_k-a_1||={dak:.4f} | ||a^-a_k||={dahk:.4f} | ||a^-a_1||={dah1:.4f}")

    # Print NN consistency summary
    if sum(nn_same_den) > 0:
        print("\nNearest-Neighbor consistency (consecutive t):")
        for i in range(len(nn_same_den)):
            frac = (nn_same_num[i] / nn_same_den[i]) if nn_same_den[i] > 0 else 0.0
            print(f"t={ts[i]:.3f} -> t={ts[i+1]:.3f}: same_NN_frac={frac:.4f}")

    # Optional plotting of cosine metrics
    if args.plot_cos:
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
        except Exception as e:
            print(f"⚠️  Matplotlib not available; cannot plot (error: {e})")
            return

        # Prepare data
        t_axis = np.array(ts[:-1], dtype=np.float32)
        cos_true_arr = np.array([per_t['cos_to_true'][i] / C for i in range(len(t_axis))], dtype=np.float32)
        cos_knn_arr  = np.array([per_t['cos_to_knn'][i] / C  for i in range(len(t_axis))], dtype=np.float32)

        # A clean, publication-friendly style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            try:
                plt.style.use('seaborn-whitegrid')
            except Exception:
                pass
        mpl.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'axes.linewidth': 1.0,
            'grid.linewidth': 0.6,
            'grid.alpha': 0.6,
        })

        fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
        colors = mpl.colormaps.get_cmap('tab10')
        ax.plot(t_axis, cos_true_arr, color=colors(0), lw=2.2, marker='o', ms=5,
                label=r'$\cos(\hat{v},\, v_{\mathrm{true}})$')
        ax.plot(t_axis, cos_knn_arr,  color=colors(1), lw=2.2, marker='s', ms=5,
                label=r'$\cos(\hat{v},\, v_{\mathrm{KNN}})$')

        ax.set_xlabel('t')
        ax.set_ylabel('Cosine similarity')
        ax.set_title('Velocity Field Cosine Similarity vs t')
        ax.set_xlim(min(0.0, float(t_axis.min())), max(1.0, float(t_axis.max())))
        ax.set_ylim(-1.05, 1.05)
        ax.legend(frameon=True)
        ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)

        # Post-tuning for a clean, publication-style look (no markers)
        # - remove markers robustly across Matplotlib versions
        for ln in ax.lines:
            try:
                ln.set_marker('')  # preferred no-marker token
            except Exception:
                try:
                    ln.set_marker('None')
                except Exception:
                    try:
                        ln.set_marker(None)
                    except Exception:
                        pass
            ln.set_markersize(0)
            ln.set_linewidth(2.0)
            ln.set_solid_capstyle('round')
            ln.set_solid_joinstyle('round')
        # - recolor to requested palette
        muted_colors = ['#2878B5', '#F8AC8C']  # blue, peach
        if len(ax.lines) >= 1:
            ax.lines[0].set_color(muted_colors[0])
            ax.lines[0].set_alpha(0.95)
        if len(ax.lines) >= 2:
            ax.lines[1].set_color(muted_colors[1])
            ax.lines[1].set_alpha(0.95)
        # - adjust figure size for single-column
        fig.set_size_inches(3.6, 2.6)
        # - minimal axes and ticks
        ax.set_title('')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(1.2)
        # Set major ticks to 0, 0.2, 0.4, 0.6, 0.8, 1 on both axes
        ticks_01 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_xticks(ticks_01)
        ax.set_yticks(ticks_01)
        ax.tick_params(direction='out', length=4.0, width=1.0)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', direction='out', length=2.5, width=0.8)
        # - subtle y-grid only and reference line
        ax.grid(False)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.axhline(0.0, color='#999999', lw=0.8, alpha=0.6, zorder=0)
        # - rebuild legend after marker removal to ensure no markers shown
        old_leg = ax.get_legend()
        if old_leg is not None:
            old_leg.remove()
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(handles, labels, frameon=True, fancybox=False, framealpha=1.0, borderpad=0.6, handlelength=2.5)
        if leg is not None:
            leg.get_frame().set_edgecolor('#DDDDDD')
            leg.get_frame().set_linewidth(0.8)

        # Save outputs (PNG + PDF)
        out_prefix = args.plot_out.rstrip('.png').rstrip('.pdf')
        out_dir = os.path.dirname(out_prefix)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        png_path = out_prefix + '.png'
        pdf_path = out_prefix + '.pdf'
        fig.savefig(png_path, dpi=args.plot_dpi, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"📈 Saved cosine plots:\n  PNG: {png_path}\n  PDF: {pdf_path}")


if __name__ == "__main__":
    main()

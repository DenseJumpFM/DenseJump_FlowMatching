import gymnasium as gym
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import random
from model import FlowMatchingPolicy
import pickle
import gc
from typing import Optional, Tuple

# Determine the appropriate compute device. Prefer CUDA when available,
# fallback to Apple's Metal Performance Shaders (MPS) on macOS, and
# otherwise use CPU. This variable is referenced throughout training
# to move tensors and models onto the same device.
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility.

    Args:
        seed (int): The seed to use for Python's ``random`` module, NumPy,
            and PyTorch (both CPU and GPU backends). Deterministic flags
            for CuDNN are also set to ensure repeatability.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behaviour when using CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For gymnasium environments, propagate the seed via environment variables
    os.environ["PYTHONHASHSEED"] = str(seed)


@torch.jit.script
def compute_stats(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimized statistics computation using torch.jit"""
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    std = torch.maximum(std, torch.tensor(1e-6, device=std.device))
    return mean, std

try:
    import h5py  # type: ignore
except Exception:
    h5py = None  # Optional dependency for HDF5


def _is_minari_spec(spec: str) -> bool:
    """Heuristic check whether a data_path refers to a Minari dataset ID."""
    s = spec.strip()
    return s.startswith("minari://") or s.startswith("D4RL/") or s.startswith("minari:")


def _load_minari_dataset(spec: str) -> tuple[np.ndarray, np.ndarray]:
    """Load observations and actions from a Minari dataset ID.

    Supports IDs such as 'D4RL/pen-expert-v2' or URIs like 'minari://D4RL/pen-expert-v2'.
    """
    try:
        import minari  # type: ignore
    except Exception as e:
        raise ImportError(
            "minari package is required to load Minari datasets. Please install minari."
        ) from e

    ds_id = spec
    if spec.startswith("minari://"):
        ds_id = spec[len("minari://") :]
    elif spec.startswith("minari:"):
        ds_id = spec[len("minari:") :]

    dataset = minari.load_dataset(ds_id)

    # Try multiple episode access APIs for robustness across versions
    episodes = None
    for getter in ("recover_episodes", "get_episodes"):
        if hasattr(dataset, getter):
            episodes = getattr(dataset, getter)()
            break
    if episodes is None and hasattr(dataset, "iterate_episodes"):
        episodes = list(dataset.iterate_episodes())
    if episodes is None:
        raise RuntimeError("Unable to retrieve episodes from Minari dataset")

    # Concatenate transitions; align obs/actions length if observations include terminal state
    obs_list = []
    act_list = []
    for ep in episodes:
        ep_obs = np.array(ep["observations"], dtype=np.float32)
        ep_act = np.array(ep["actions"], dtype=np.float32)
        if ep_obs.shape[0] == ep_act.shape[0] + 1:
            ep_obs = ep_obs[:-1]
        elif ep_act.shape[0] == ep_obs.shape[0] + 1:
            ep_act = ep_act[:-1]
        min_len = min(ep_obs.shape[0], ep_act.shape[0])
        if min_len == 0:
            continue
        obs_list.append(ep_obs[:min_len])
        act_list.append(ep_act[:min_len])

    if not obs_list:
        raise RuntimeError("No transitions found in Minari dataset")

    observations = np.concatenate(obs_list, axis=0)
    actions = np.concatenate(act_list, axis=0)
    return observations, actions


def _load_hdf5(h5_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load observations/actions from an HDF5 file.

    Searches for common keys like 'observations'/'obs' and 'actions'/'acts'.
    If sequences differ by +1 due to terminal obs, it trims to align lengths.
    Returns (observations, actions).
    """
    if h5py is None:
        raise ImportError("h5py is required to read .hdf5 datasets. Please install h5py.")

    with h5py.File(h5_path, 'r') as f:
        # If organized as episodes (common), concatenate per-episode arrays
        top_keys = list(f.keys())
        episode_groups = [k for k in top_keys if isinstance(f[k], h5py.Group)]

        if episode_groups:
            obs_list, act_list = [], []
            for gname in episode_groups:
                g = f[gname]
                # Prefer standard keys
                ok = None
                ak = None
                for k in ['observations', 'obs', 'states']:
                    if k in g.keys():
                        ok = k
                        break
                for k in ['actions', 'acts', 'action']:
                    if k in g.keys():
                        ak = k
                        break
                # Fallback: scan group for datasets
                if ok is None or ak is None:
                    ds = {}
                    def _collect(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            ds[name] = obj
                    g.visititems(_collect)
                    if ok is None:
                        for name, d in ds.items():
                            if d.ndim >= 2:
                                ok = name
                                break
                    if ak is None:
                        for name, d in ds.items():
                            if d.ndim >= 2 and name != ok:
                                ak = name
                                break
                if ok is None or ak is None:
                    continue
                ep_obs = np.array(g[ok], dtype=np.float32)
                ep_act = np.array(g[ak], dtype=np.float32)
                if ep_obs.shape[0] == ep_act.shape[0] + 1:
                    ep_obs = ep_obs[:-1]
                elif ep_act.shape[0] == ep_obs.shape[0] + 1:
                    ep_act = ep_act[:-1]
                n = min(len(ep_obs), len(ep_act))
                if n <= 0:
                    continue
                obs_list.append(ep_obs[:n])
                act_list.append(ep_act[:n])
            if not obs_list:
                raise RuntimeError("No per-episode observations/actions found in HDF5 file")
            observations = np.concatenate(obs_list, axis=0)
            actions = np.concatenate(act_list, axis=0)
        else:
            # Recursively search for candidate datasets in flat layout
            datasets = {}
            def _collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets[name] = obj
            f.visititems(_collect)

            def pick_key(candidates: list[str]) -> Optional[str]:
                lowered = {k.lower(): k for k in datasets.keys()}
                for cand in candidates:
                    cand_l = cand.lower()
                    for path_l, orig in lowered.items():
                        if cand_l in path_l:
                            return orig
                return None

            obs_key = pick_key(['observations', 'obs', 'states'])
            act_key = pick_key(['actions', 'acts', 'action'])
            if obs_key is None:
                for k, d in datasets.items():
                    if d.ndim >= 2:
                        obs_key = k
                        break
            if act_key is None:
                for k, d in datasets.items():
                    if d.ndim >= 2 and k != obs_key:
                        act_key = k
                        break
            if obs_key is None or act_key is None:
                raise RuntimeError("Could not find observations/actions datasets in HDF5 file")
            observations = np.array(datasets[obs_key], dtype=np.float32)
            actions = np.array(datasets[act_key], dtype=np.float32)

    # Align lengths if necessary
    if observations.shape[0] == actions.shape[0] + 1:
        observations = observations[:-1]
    elif actions.shape[0] == observations.shape[0] + 1:
        actions = actions[:-1]
    n = min(observations.shape[0], actions.shape[0])
    observations = observations[:n]
    actions = actions[:n]

    return observations, actions


def load_data(data_path: str) -> dict:
    """Optimized data loading with memory-efficient processing.

    Supports .npz, .pkl, and Minari dataset IDs (e.g., 'D4RL/pen-expert-v2').
    """
    if data_path.endswith('.hdf5') or data_path.endswith('.h5'):
        observations, actions = _load_hdf5(data_path)
        obs_tensor = torch.from_numpy(observations)
        acts_tensor = torch.from_numpy(actions)
        obs_mean_t, obs_std_t = compute_stats(obs_tensor)
        action_mean_t, action_std_t = compute_stats(acts_tensor)
        obs_mean = obs_mean_t.numpy()
        obs_std = obs_std_t.numpy()
        action_mean = action_mean_t.numpy()
        action_std = action_std_t.numpy()
    elif _is_minari_spec(data_path):
        observations, actions = _load_minari_dataset(data_path)
        # Compute normalization stats
        obs_tensor = torch.from_numpy(observations)
        acts_tensor = torch.from_numpy(actions)
        obs_mean_t, obs_std_t = compute_stats(obs_tensor)
        action_mean_t, action_std_t = compute_stats(acts_tensor)
        obs_mean = obs_mean_t.numpy()
        obs_std = obs_std_t.numpy()
        action_mean = action_mean_t.numpy()
        action_std = action_std_t.numpy()
    elif data_path.endswith(".npz"):
        # Load NPZ format with memory mapping for large files
        data = np.load(data_path, mmap_mode='r' if os.path.getsize(data_path) > 1e8 else None)
        observations = np.array(data["observations"])
        actions = np.array(data["actions"])

        # Use provided normalization statistics when available
        if "action_mean" in data and "action_std" in data:
            action_mean = np.array(data["action_mean"])
            action_std = np.array(data["action_std"])
        else:
            # Use torch for faster computation
            actions_tensor = torch.from_numpy(actions).float()
            action_mean_t, action_std_t = compute_stats(actions_tensor)
            action_mean = action_mean_t.numpy()
            action_std = action_std_t.numpy()
            print("Computing action normalization stats")

        if "obs_mean" in data and "obs_std" in data:
            obs_mean = np.array(data["obs_mean"])
            obs_std = np.array(data["obs_std"])
        else:
            obs_tensor = torch.from_numpy(observations).float()
            obs_mean_t, obs_std_t = compute_stats(obs_tensor)
            obs_mean = obs_mean_t.numpy()
            obs_std = obs_std_t.numpy()
            print("Computing observation normalization stats")

    elif data_path.endswith(".pkl"):
        # Load PKL format with optimized memory usage
        with open(data_path, "rb") as f:
            episodes = pickle.load(f)

        print(f"Loaded {len(episodes)} episodes")

        # Pre-allocate arrays for better memory efficiency
        total_obs_len = sum(len(ep["observations"]) for ep in episodes)
        
        # Get shapes from first episode
        first_ep = episodes[0]
        obs_shape = np.array(first_ep['observations']).shape[1:]
        act_shape = np.array(first_ep['actions']).shape[1:]
        
        # Pre-allocate arrays
        observations = np.empty((total_obs_len,) + obs_shape, dtype=np.float32)
        actions = np.empty((total_obs_len,) + act_shape, dtype=np.float32)
        
        # Fill arrays efficiently
        obs_idx = 0
        for ep in episodes:
            ep_obs = np.array(ep["observations"], dtype=np.float32)
            ep_acts = np.array(ep["actions"], dtype=np.float32)
            
            ep_len = len(ep_obs)
            observations[obs_idx:obs_idx + ep_len] = ep_obs
            actions[obs_idx:obs_idx + ep_len] = ep_acts
            obs_idx += ep_len

        # Use torch for faster statistics computation
        obs_tensor = torch.from_numpy(observations)
        acts_tensor = torch.from_numpy(actions)
        
        obs_mean_t, obs_std_t = compute_stats(obs_tensor)
        action_mean_t, action_std_t = compute_stats(acts_tensor)
        
        obs_mean = obs_mean_t.numpy()
        obs_std = obs_std_t.numpy() 
        action_mean = action_mean_t.numpy()
        action_std = action_std_t.numpy()

    else:
        raise ValueError("Unsupported data source. Use .npz, .pkl, .hdf5, or a Minari dataset ID like 'D4RL/pen-expert-v2'")

    print(f"Data loaded: {len(observations)} samples, obs_dim={observations.shape[1]}, act_dim={actions.shape[1]}")

    # Ensure consistency with original function
    action_std = np.maximum(action_std, 1e-6)
    obs_std = np.maximum(obs_std, 1e-6)

    return {
        "observations": observations,
        "actions": actions,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }


def _resolve_action_bounds_from_tasks(data_path: str):
    try:
        import json
        TASKS_PATHS_FILE = 'tasks_paths.json'
        if os.path.exists(TASKS_PATHS_FILE):
            with open(TASKS_PATHS_FILE, 'r') as f:
                tasks = json.load(f)
            for k, v in tasks.items():
                if str(v.get('data_path', '')) == str(data_path):
                    low = v.get('action_low', None)
                    high = v.get('action_high', None)
                    if low is not None and high is not None:
                        return np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)
    except Exception:
        pass
    return None, None


def create_data_loader(
    data_path: str,
    batch_size: int = 128,
    seed: int | None = None,
    num_train: int | None = None,
    pin_memory: bool = True,
    num_workers: int = 2,
    action_mode: str = 'normalize',
) -> tuple[torch.utils.data.DataLoader, dict]:
    """Create an optimized mini-batch data loader from expert demonstrations."""
    # Load the raw data and compute normalization statistics
    data_dict = load_data(data_path)
    observations = data_dict["observations"]
    actions = data_dict["actions"]
    action_mean = data_dict["action_mean"]
    action_std = data_dict["action_std"]
    obs_mean = data_dict["obs_mean"]
    obs_std = data_dict["obs_std"]

    N = len(observations)
    # Determine number of training samples
    if num_train is None:
        num_train = N // 200
    num_train = max(1, min(num_train, N))
    print(f"Using {num_train}/{N} samples ({num_train / N * 100:.1f}%)")

    # IMPORTANT: Recompute statistics using ONLY training data (not all data)
    training_observations = observations[:num_train]
    training_actions = actions[:num_train]
    
    # Use torch for faster computation on training data only
    training_obs_tensor = torch.from_numpy(training_observations).float()
    training_actions_tensor = torch.from_numpy(training_actions).float()
    
    obs_mean_t, obs_std_t = compute_stats(training_obs_tensor)
    action_mean_t, action_std_t = compute_stats(training_actions_tensor)
    
    obs_mean = obs_mean_t.numpy()
    obs_std = obs_std_t.numpy()
    action_mean = action_mean_t.numpy()
    action_std = action_std_t.numpy()
    
    print("📊 Recomputed normalization stats using TRAINING data only")

    # Pre-allocate normalized arrays for better memory efficiency
    obs_norm = np.empty((num_train, observations.shape[1]), dtype=np.float32)
    action_norm = np.empty((num_train, actions.shape[1]), dtype=np.float32)
    
    # Vectorized normalization / scaling
    obs_norm = (training_observations - obs_mean) / obs_std
    if action_mode == 'scale':
        low, high = _resolve_action_bounds_from_tasks(data_path)
        if low is None or high is None:
            raise ValueError("action_mode='scale' requires action_low/high in tasks_paths.json for this data_path")
        # Expand scalar/len-1 shorthand to per-dimension bounds
        if low.ndim == 0 or (low.ndim == 1 and low.shape[0] == 1):
            low = np.repeat(low.reshape(1), actions.shape[1]).astype(np.float32)
        if high.ndim == 0 or (high.ndim == 1 and high.shape[0] == 1):
            high = np.repeat(high.reshape(1), actions.shape[1]).astype(np.float32)
        scale = (high - low)
        scale[scale == 0] = 1.0
        action_norm = 2.0 * (training_actions - low) / scale - 1.0
        action_norm = np.clip(action_norm, -1.0, 1.0)
    else:
        action_norm = (training_actions - action_mean) / action_std
    

    # Convert to tensors directly without intermediate lists
    obs_tensor = torch.from_numpy(obs_norm.astype(np.float32))
    action_tensor = torch.from_numpy(action_norm.astype(np.float32))
    dataset = TensorDataset(obs_tensor, action_tensor)

    # Create reproducible shuffling via generator when seed provided
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    # Optimized DataLoader with pin_memory and num_workers
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        generator=generator,
        pin_memory=pin_memory and torch.cuda.is_available(),
        num_workers=num_workers if torch.cuda.is_available() else 0,
        persistent_workers=num_workers > 0 and torch.cuda.is_available(),
        drop_last=True  # For more consistent batch sizes
    )

    # Package training information
    training_info = {
        "normalized_actions": action_norm,
        "raw_actions": actions[:num_train],
        "action_mean": action_mean,
        "action_std": action_std,
        "num_samples": num_train,
        "data_format": (
            "minari" if _is_minari_spec(data_path)
            else (".hdf5" if (data_path.endswith('.hdf5') or data_path.endswith('.h5'))
                  else (".pkl" if data_path.endswith(".pkl") else ".npz"))
        ),
    }

    return data_loader, training_info


def _detect_dims_from_data(data_path: str) -> Tuple[int, int]:
    """Lightweight detection of (obs_dim, act_dim) from data_path.

    Tries to avoid loading entire datasets; for HDF5 reads only the first episode group.
    """
    if data_path.endswith('.hdf5') or data_path.endswith('.h5'):
        if h5py is None:
            raise ImportError('h5py is required to read .hdf5 files. Please install h5py.')
        with h5py.File(data_path, 'r') as f:
            # pick first group
            for gname in f.keys():
                obj = f[gname]
                if isinstance(obj, h5py.Group):
                    g = obj
                    ok = None
                    ak = None
                    for k in ['observations', 'obs', 'states']:
                        if k in g.keys():
                            ok = k
                            break
                    for k in ['actions', 'acts', 'action']:
                        if k in g.keys():
                            ak = k
                            break
                    if ok is None or ak is None:
                        # fallback: search datasets in group
                        ds = {}
                        def _collect(name, obj2):
                            if isinstance(obj2, h5py.Dataset):
                                ds[name] = obj2
                        g.visititems(_collect)
                        if ok is None:
                            for name, d in ds.items():
                                if d.ndim >= 2:
                                    ok = name
                                    break
                        if ak is None:
                            for name, d in ds.items():
                                if d.ndim >= 2 and name != ok:
                                    ak = name
                                    break
                    if ok is not None and ak is not None:
                        od = g[ok].shape[-1] if len(g[ok].shape) > 1 else 1
                        ad = g[ak].shape[-1] if len(g[ak].shape) > 1 else 1
                        return int(od), int(ad)
            # fallback to full load if structure unexpected
        dd = load_data(data_path)
        return dd['observations'].shape[1], dd['actions'].shape[1]
    elif _is_minari_spec(data_path):
        try:
            import minari  # type: ignore
        except Exception:
            dd = load_data(data_path)
            return dd['observations'].shape[1], dd['actions'].shape[1]
        ds_id = data_path
        if data_path.startswith('minari://'):
            ds_id = data_path[len('minari://'):]
        elif data_path.startswith('minari:'):
            ds_id = data_path[len('minari:'):]
        dataset = minari.load_dataset(ds_id)
        # take first episode
        eps = None
        for getter in ("recover_episodes", "get_episodes"):
            if hasattr(dataset, getter):
                eps = getattr(dataset, getter)()
                break
        if eps is None and hasattr(dataset, 'iterate_episodes'):
            eps = list(dataset.iterate_episodes())
        ep0 = eps[0]
        od = np.array(ep0['observations']).shape[1]
        ad = np.array(ep0['actions']).shape[1]
        return int(od), int(ad)
    elif data_path.endswith('.npz'):
        data = np.load(data_path)
        od = data['observations'].shape[1] if len(data['observations'].shape) > 1 else 1
        ad = data['actions'].shape[1] if len(data['actions'].shape) > 1 else 1
        return int(od), int(ad)
    elif data_path.endswith('.pkl'):
        with open(data_path, 'rb') as f:
            episodes = pickle.load(f)
        od = np.array(episodes[0]['observations']).shape[1]
        ad = np.array(episodes[0]['actions']).shape[1]
        return int(od), int(ad)
    else:
        raise ValueError('Unsupported data file for dimension detection')


def _detect_task_name(data_path: str) -> str:
    """Infer task/environment name from tasks_paths.json, dims, or filename hints.

    Priority:
    1) tasks_paths.json data_path match → env_name
    2) dims mapping fallback
    3) filename hint
    """
    # 1) tasks_paths.json
    try:
        import json
        TASKS_PATHS_FILE = "tasks_paths.json"
        if os.path.exists(TASKS_PATHS_FILE):
            with open(TASKS_PATHS_FILE, 'r') as f:
                tasks_paths = json.load(f)
            for k, v in tasks_paths.items():
                if str(v.get('data_path', '')) == str(data_path):
                    return v.get('env_name', 'unknown')
    except Exception:
        pass

    # 2) dims mapping fallback
    try:
        od, ad = _detect_dims_from_data(data_path)
        mapping = {
            (3, 1): 'Pendulum-v1',
            (17, 6): 'Walker2d-v5',
            (11, 3): 'Hopper-v4',
            (45, 24): 'AdroitHandPen-v1',
            (8, 2): 'LunarLanderContinuous-v2',
        }
        return mapping.get((od, ad), 'unknown')
    except Exception:
        # 3) filename hints
        base = os.path.basename(data_path).lower()
        if 'pen' in base:
            return 'AdroitHandPen-v1'
        return 'unknown'


def train_flow_matching(
    data_path: str,
    lr: float,
    epochs: int,
    batch_size: int,
    log_dir: str,
    output_path: str,
    seed: int | None = None,
    num_train: int | None = None,
    dropout_p: float = 0.1,
    obs_hidden_dims: list[int] = [128, 128],
    policy_hidden_dims: list[int] = [128, 128],
    layernorm: bool = False,
    beta: list[float] = [1.0, 1.0],
    policy_arch: str = 'mlp',
    unet_base_ch: int = 64,
    unet_depth: int = 3,
    action_mode: str = 'normalize',
    t_choices: list[float] | None = None,
    t_dist: str = 'auto',
    l2: float = 0.0,
    resume_from: str | None = None,
) -> None:
    """Train a flow matching policy using mini-batch stochastic gradient descent.

    This function constructs a PyTorch ``DataLoader`` for the demonstration
    dataset and iterates through it using SGD optimizers.

    Args:
        data_path (str): Path to the expert demonstration data (``.npz`` or ``.pkl``).
        lr (float): Learning rate for the policy optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size for training.
        log_dir (str): Directory where TensorBoard logs are written.
        output_path (str): File path to save the trained policy.
        seed (int | None): Optional random seed for reproducibility.
        num_train (int | None): Number of training samples to use; defaults to
            ``N//200`` where ``N`` is the dataset size.
        dropout_p (float): Dropout probability for the flow matching policy.
        obs_hidden_dims (list[int]): Hidden dimensions for observation encoder.
        policy_hidden_dims (list[int]): Hidden dimensions for policy network.
        layernorm (bool): Whether to use layer normalization.
        beta (list[float]): Beta distribution parameters for time sampling.
        policy_arch (str): Policy architecture ('mlp', 'state_unet', 'rgb_unet').
        unet_base_ch (int): Base channels for UNet architecture.
        unet_depth (int): Depth for UNet architecture.
    """
    # Set seeds for reproducibility
    if seed is not None:
        set_seed(seed)
        print(f"🎲 Set seed to {seed} for flow matching training")

    # Infer observation and action dimensions to initialize the policy
    data_dict = load_data(data_path)
    obs_dim = (
        data_dict["observations"].shape[1]
        if len(data_dict["observations"].shape) > 1
        else 1
    )
    action_dim = (
        data_dict["actions"].shape[1]
        if len(data_dict["actions"].shape) > 1
        else 1
    )
    print(f"📊 Detected from data: obs_dim={obs_dim}, action_dim={action_dim}")

    # Optionally resume: load checkpoint to restore model config and weights
    start_epoch = 0
    resume_ckpt: dict | None = None
    if resume_from is not None and os.path.isfile(resume_from):
        try:
            resume_ckpt = torch.load(resume_from, map_location=device)
            print(f"🔁 Resuming from checkpoint: {resume_from}")
            # Prefer checkpoint model hyperparameters to ensure compatibility
            obs_hidden_dims = resume_ckpt.get('obs_hidden_dims', obs_hidden_dims)
            policy_hidden_dims = resume_ckpt.get('policy_hidden_dims', policy_hidden_dims)
            dropout_p = resume_ckpt.get('dropout_p', dropout_p)
            layernorm = resume_ckpt.get('layernorm', layernorm)
            policy_arch = resume_ckpt.get('arch', policy_arch)
            unet_base_ch = resume_ckpt.get('unet_base_ch', unet_base_ch)
            unet_depth = resume_ckpt.get('unet_depth', unet_depth)

            # Derive starting epoch if available
            ti = resume_ckpt.get('training_info', {}) if isinstance(resume_ckpt, dict) else {}
            if isinstance(ti, dict):
                if 'epoch' in ti:
                    start_epoch = int(ti.get('epoch', 0)) + 1
                elif 'epochs' in ti:
                    # Saved final model; continue after full previous run
                    start_epoch = int(ti.get('epochs', 0))
            # Cap start_epoch to requested total epochs
            start_epoch = max(0, min(start_epoch, epochs))
        except Exception as e:
            print(f"⚠️  Failed to load resume checkpoint '{resume_from}': {e}")
            resume_ckpt = None

    # Initialize the flow matching policy (use possibly updated hyperparameters)
    policy = FlowMatchingPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon=1,
        obs_hidden=obs_hidden_dims,
        policy_hidden=policy_hidden_dims,
        dropout_p=dropout_p,
        layernorm=layernorm,
        arch=policy_arch,
        unet_base_ch=unet_base_ch,
        unet_depth=unet_depth,
    ).to(device)

    # Load weights from resume checkpoint if provided
    if resume_ckpt is not None and isinstance(resume_ckpt, dict) and 'model_state_dict' in resume_ckpt:
        try:
            policy.load_state_dict(resume_ckpt['model_state_dict'], strict=True)
            print(f"✅ Loaded model weights from checkpoint. Starting at epoch {start_epoch} / {epochs}")
        except Exception as e:
            print(f"⚠️  Could not strictly load state dict ({e}); trying non-strict load...")
            missing, unexpected = policy.load_state_dict(resume_ckpt['model_state_dict'], strict=False)
            print(f"   Non-strict load done. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # Create a DataLoader for mini-batch training
    data_loader, training_info = create_data_loader(
        data_path=data_path,
        batch_size=batch_size,
        seed=seed,
        num_train=num_train,
        action_mode=action_mode,
    )

    # Instantiate optimizer: use Adam for mini-batch updates
    # Use AdamW to decouple weight decay from the gradient update (true L2 regularization)
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=l2)
    print(f"📊 Policy training AdamW optimizer: lr={lr}, weight_decay={l2}")

    # Prepare a SummaryWriter for logging
    writer = SummaryWriter(log_dir=log_dir)
    if start_epoch > 0:
        print(f"🚀 Resumed training: total epochs={epochs}, start at epoch={start_epoch}")
    else:
        print(f"🚀 Starting flow matching training for {epochs} epochs...")
    
    # Time sampling configuration
    beta_alpha1, beta_alpha2 = beta[0], beta[1]
    selected_time_sampling = None
    # Prepare discrete t choices tensor if provided
    discrete_t_tensor = None
    if t_choices is not None and len(t_choices) > 0:
        vals = [float(x) for x in t_choices]
        for v in vals:
            if not (0.0 <= v <= 1.0):
                raise ValueError("All --t_choices values must be within [0,1]")
        # Warn if values are exactly 0/1 which may reduce gradient signal for x_t construction
        if any(v in (0.0, 1.0) for v in vals):
            print("⚠️  t_choices contains 0 or 1; ensure this is intended.")
        discrete_t_tensor = torch.tensor(vals, dtype=torch.float32, device=device)
        selected_time_sampling = 'discrete'
        print(f"⏱️  Discrete t sampling enabled: {vals} (overrides other modes)")
    else:
        td = (t_dist.lower() if isinstance(t_dist, str) else 'auto')
        if td in ('poly_x2p1', 'poly_x2_plus_1', 'x2p1', 'x2_plus_1'):
            selected_time_sampling = 'poly_x2p1'
            print("⏰ Time sampling: Custom f(t) = 3/4 * (t^2 + 1)")
            print("Time sampling: Custom f(t) = (12/13) * (t^2 - t + 5/4)")
        elif td == 'uniform':
            selected_time_sampling = 'uniform'
            print("⏰ Time sampling: Uniform distribution")
        elif td in ('mix_uniform_beta', 'mix_ubeta', 'mix_u_beta'):
            selected_time_sampling = 'mix_uniform_beta'
            print(f"⏰ Time sampling: 50% Uniform + 50% Beta(α1={beta_alpha1}, α2={beta_alpha2})")
        elif td in ('ploy0.75', 'ploy075', 'ploy0p75', 'poly0.75'):
            selected_time_sampling = 'ploy0.75'
            print("Time sampling: ploy0.75 with pdf (t-3/4)^2 + 41/48")
        elif td == 'beta' or td == 'auto':
            if beta_alpha1 == 1.0 and beta_alpha2 == 1.0:
                selected_time_sampling = 'uniform'
                print("⏰ Time sampling: Uniform distribution (beta α1=1.0, α2=1.0)")
            else:
                selected_time_sampling = 'beta'
                print(f"⏰ Time sampling: Beta distribution (α1={beta_alpha1}, α2={beta_alpha2})")

    # Ensure left_low selection if not matched above due to special characters in prints
    if selected_time_sampling is None and (isinstance(t_dist, str) and t_dist.lower() in ('left_low','leftlow')):
        selected_time_sampling = 'left_low'
        print("Time sampling: 1/4 Uniform + 3/4 Beta(3,1) (left_low)")
    print(f"Training with full precision")

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        epoch_flow_loss = 0.0
        num_batches = 0

        # Iterate over the DataLoader, which yields mini-batches of normalized data
        for batch_idx, (obs_batch, action_batch) in enumerate(data_loader):
            # Send data to device with non_blocking for async transfer
            obs_batch = obs_batch.to(device, non_blocking=True)
            action_batch = action_batch.view(action_batch.size(0), -1).to(device, non_blocking=True)
            # Zero gradients (set_to_none is more efficient)
            policy_optimizer.zero_grad(set_to_none=True)

            # Compute flow matching loss
            try:
                # Try new version with beta parameters
                flow_loss = policy.flow_matching_loss(
                    obs_batch,
                    action_batch,
                    beta_alpha1=beta_alpha1,
                    beta_alpha2=beta_alpha2,
                    discrete_t_choices=discrete_t_tensor,
                    t_dist=(None if selected_time_sampling in (None, 'discrete') else selected_time_sampling),
                )
            except TypeError as e:
                msg = str(e)
                if "unexpected keyword argument" in msg and ("beta" in msg or "discrete_t_choices" in msg or "t_dist" in msg):
                    # Fallback to old version without beta/discrete support
                    print("⚠️  WARNING: FlowMatchingPolicy without beta/discrete support, falling back to uniform sampling")
                    flow_loss = policy.flow_matching_loss(
                        obs_batch,
                        action_batch,
                    )
                else:
                    raise e
            total_loss = flow_loss
            

            # Backpropagate and update
            total_loss.backward()
            policy_optimizer.step()

            # Update epoch statistics
            epoch_loss += total_loss.item()
            epoch_flow_loss += flow_loss.item()
            num_batches += 1
            
            # Clear cache periodically to prevent memory buildup
            if num_batches % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

        # Compute and log average losses
        avg_epoch_loss = epoch_loss / max(1, num_batches)
        avg_flow_loss = epoch_flow_loss / max(1, num_batches)
        writer.add_scalar("train/epoch_flow_loss", avg_flow_loss, epoch)
        writer.add_scalar("train/epoch_total_loss", avg_epoch_loss, epoch)

        # Periodic logging and saving to console
        if epoch % 100 == 0:
            print(
                f"[Flow Matching] Epoch {epoch}/{epochs}, Loss: {avg_epoch_loss:.4f}"
            )
            
            # Save checkpoint every 100 epochs (except at epoch 0)
            if epoch > 0:
                # Extract the full filename pattern from output_path and add epoch
                base_filename = os.path.basename(output_path).replace(".pth", "")
                checkpoint_filename = f"{base_filename}_epoch{epoch}.pth"
                checkpoint_path = os.path.join(os.path.dirname(output_path), checkpoint_filename)
                policy_config = {
                    'model_state_dict': policy.state_dict(),
                    'obs_dim': obs_dim,
                    'action_dim': action_dim,
                    'horizon': 1,
                    'obs_hidden_dims': obs_hidden_dims,
                    'policy_hidden_dims': policy_hidden_dims,
                    'dropout_p': dropout_p,
                    'layernorm': layernorm,
                    'arch': policy_arch,
                    'unet_base_ch': unet_base_ch,
                    'unet_depth': unet_depth,
                    'training_info': {
                        'epoch': epoch,
                        'total_epochs': epochs,
                        'lr': lr,
                        'batch_size': batch_size,
                        'num_train': data_loader.dataset.tensors[0].shape[0],
                        'avg_loss': avg_epoch_loss,
                        'beta_alpha1': beta_alpha1,
                        'beta_alpha2': beta_alpha2,
                        'time_sampling': (
                            'discrete' if (discrete_t_tensor is not None) else (
                                selected_time_sampling if selected_time_sampling is not None else (
                                    'uniform' if beta_alpha1 == 1.0 and beta_alpha2 == 1.0 else 'beta'
                                )
                            )
                        ),
                        't_choices': (t_choices if t_choices is not None else None),
                        'weight_decay_l2': l2,
                    }
                }
                torch.save(policy_config, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")

    # Save the policy after training concludes with configuration
    policy_config = {
        'model_state_dict': policy.state_dict(),
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'horizon': 1,
        'obs_hidden_dims': obs_hidden_dims,
        'policy_hidden_dims': policy_hidden_dims,
        'dropout_p': dropout_p,
        'layernorm': layernorm,
        'arch': policy_arch,
        'unet_base_ch': unet_base_ch,
        'unet_depth': unet_depth,
        'training_info': {
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'num_train': data_loader.dataset.tensors[0].shape[0],
            'beta_alpha1': beta_alpha1,
            'beta_alpha2': beta_alpha2,
            'time_sampling': (
                'discrete' if (discrete_t_tensor is not None) else (
                    selected_time_sampling if selected_time_sampling is not None else (
                        'uniform' if beta_alpha1 == 1.0 and beta_alpha2 == 1.0 else 'beta'
                    )
                )
            ),
            't_choices': (t_choices if t_choices is not None else None),
            'weight_decay_l2': l2,
        }
    }
    torch.save(policy_config, output_path)
    print(f"✅ Flow matching policy saved to {output_path}")

    # Close the writer
    writer.close()
    print("✅ Flow matching training completed.")


def main() -> None:
    """Entry point for command-line execution of the flow matching trainer."""
    parser = argparse.ArgumentParser(description="Train flow matching policy")

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to expert data file or Minari dataset ID (e.g., D4RL/pen-expert-v2)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--obs_hidden_dims",
        type=int,
        nargs='+',
        default=[128, 128],
        help="Hidden dimensions for observation encoder in policy (default: [128, 128])",
    )
    parser.add_argument(
        "--policy_hidden_dims",
        type=int,
        nargs='+',
        default=[128, 128],
        help="Hidden dimensions for policy network (default: [128, 128])",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="Latent dimension for VAE (default: 64, auto-detected if model has config)",
    )
    parser.add_argument(
        "--layernorm",
        action="store_true",
        help="Use LayerNorm in models (default: False)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam (default: 1e-4)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size (default: 32)",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.0,
        help="L2 weight decay for policy optimizer (default: 0.0)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs/flow_matching",
        help="Directory for logs (default: ./logs/flow_matching)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./logs/flow_matching/policy.pth",
        help="Path to save trained policy (default: ./logs/flow_matching/policy.pth)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=None,
        help="Number of training samples to use (default: None, uses N//200)",
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        default=0.0,
        help="Dropout probability (default: 0.0)",
    )
    parser.add_argument(
        "--policy_arch",
        type=str,
        default="mlp",
        choices=["mlp", "state_unet", "rgb_unet"],
        help="Architecture for flow policy (default: mlp)",
    )
    parser.add_argument(
        "--unet_base_ch",
        type=int,
        default=64,
        help="Base channel width for UNet (default: 64)",
    )
    parser.add_argument(
        "--unet_depth",
        type=int,
        default=3,
        help="Depth (number of residual stages) for UNet (default: 3)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        metavar=("ALPHA1", "ALPHA2"),
        help="Beta distribution parameters (alpha1, alpha2) for time sampling (default: 1.0 1.0, uniform)",
    )
    parser.add_argument(
        "--t_dist",
        type=str,
        default="auto",
        choices=["auto", "uniform", "beta", "poly_x2p1", "mix_uniform_beta", "left_low", "ploy0.75"],
        help=(
            "Time sampling distribution for t: "
            "'auto' uses discrete if --t_choices else beta(--beta) else uniform; "
            "'poly_x2p1' uses f(t)=(12/13)*(t^2 - t + 5/4); "
            "'mix_uniform_beta' uses 50% Uniform + 50% Beta(alpha1,alpha2); "
            "'left_low' uses 1/4 Uniform + 3/4 Beta(3,1); "
            "'ploy0.75' uses ((t-3/4)^2 + 41/48)."
        ),
    )
    parser.add_argument(
        "--t", "--t_choices", dest="t_choices",
        type=str,
        default=None,
        help="Comma-separated list of discrete t values in [0,1] (e.g., '0,0.5,0.7'). Overrides --beta and uniform",
    )
    parser.add_argument(
        "--action_mode",
        type=str,
        default="normalize",
        choices=["normalize", "scale"],
        help="Action preprocessing: 'normalize' (z-score) or 'scale' to [-1,1] using tasks_paths.json bounds",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a saved checkpoint (.pth) to resume training from",
    )

    args = parser.parse_args()

    # Construct descriptive strings for seed and training sample count
    seed_str = f"seed{args.seed}" if args.seed is not None else "noseed"
    num_train_str = f"nt{args.num_train}" if args.num_train is not None else "ntauto"
    
    # Add time sampling info to output path
    beta_alpha1, beta_alpha2 = args.beta[0], args.beta[1]
    if args.t_dist == 'poly_x2p1':
        beta_str = 'tdist_polyx2p1'
    elif args.t_dist == 'left_low':
        beta_str = 'tdist_leftlow'
    elif args.t_dist == 'ploy0.75':
        beta_str = 'tdist_ploy0p75'
    elif args.t_dist == 'mix_uniform_beta':
        a1s = f"{beta_alpha1}".replace('.', 'p')
        a2s = f"{beta_alpha2}".replace('.', 'p')
        beta_str = f"tdist_mixubeta_{a1s}_{a2s}"
    else:
        if beta_alpha1 == 1.0 and beta_alpha2 == 1.0:
            beta_str = "uniform"
        else:
            # Format beta parameters for filename (replace decimals with 'p')
            alpha1_str = f"{beta_alpha1}".replace('.', 'p')
            alpha2_str = f"{beta_alpha2}".replace('.', 'p')
            beta_str = f"beta{alpha1_str}_{alpha2_str}"
    
    # Add architecture info to output path
    arch_str = args.policy_arch
    if args.policy_arch in ["state_unet", "rgb_unet"]:
        arch_str += f"_ch{args.unet_base_ch}_d{args.unet_depth}"

    # Add hidden dims info to output path
    obs_dims_str = "x".join(str(x) for x in args.obs_hidden_dims)
    pol_dims_str = "x".join(str(x) for x in args.policy_hidden_dims)
    dims_str = f"obs{obs_dims_str}_pol{pol_dims_str}"

    # Detect task name from data to separate outputs per environment
    task_name = _detect_task_name(args.data_path)
    task_dir = task_name.replace('/', '_')

    # Adjust log directory if default is used (nest under task folder)
    if args.log_dir == "./logs/flow_matching":
        args.log_dir = (
            f"./logs/flow_matching/{task_dir}/"
            f"{seed_str}_{num_train_str}_{beta_str}_{dims_str}_{arch_str}"
        )
    # Adjust output path if default is used (nest under same task folder)
    if args.output_path == "./logs/flow_matching/policy.pth":
        args.output_path = (
            f"./logs/flow_matching/{task_dir}/"
            f"{seed_str}_{num_train_str}_{beta_str}_{dims_str}_{arch_str}/policy.pth"
        )

    # Ensure directories exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Parse discrete t choices if provided
    t_choices_list = None
    if args.t_choices is not None:
        s = args.t_choices.strip()
        # Allow formats like "0,0.5,0.7" or "[0, 0.5, 0.7]"
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        parts = [p.strip() for p in s.split(',') if p.strip() != '']
        try:
            t_choices_list = [float(p) for p in parts]
        except Exception:
            raise ValueError("--t_choices must be a comma-separated list of numbers, e.g., '0,0.5,0.7'")

    # Invoke the training routine
    train_flow_matching(
        data_path=args.data_path,
        lr=args.lr,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
        output_path=args.output_path,
        seed=args.seed,
        num_train=args.num_train,
        dropout_p=args.dropout_p,
        obs_hidden_dims=args.obs_hidden_dims,
        policy_hidden_dims=args.policy_hidden_dims,
        layernorm=args.layernorm,
        beta=args.beta,
        policy_arch=args.policy_arch,
        unet_base_ch=args.unet_base_ch,
        unet_depth=args.unet_depth,
        action_mode=args.action_mode,
        t_choices=t_choices_list,
        t_dist=args.t_dist,
        l2=args.l2,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()

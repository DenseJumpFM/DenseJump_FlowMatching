import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import random
import csv
from tqdm import tqdm
from model import FlowMatchingPolicy
from train_flow import load_data
import gc
import json
import h5py 
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)
H = 1  # horizon
# Flow step T will be set by command line argument

# Load tasks paths
TASKS_PATHS_FILE = "tasks_paths.json"

# Standardize episode length across environments for fair comparison
#

def load_tasks_paths():
    """Load tasks and their corresponding data paths from JSON file."""
    if os.path.exists(TASKS_PATHS_FILE):
        with open(TASKS_PATHS_FILE, 'r') as f:
            return json.load(f)
        # Return empty dict if file doesn't exist
        return {}

# number of flow steps per update
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def calculate_success_rate(returns, steps, env_name):
    """Calculate success rate based on environment-specific criteria"""
    total_episodes = len(returns)
    if total_episodes == 0:
        return 0.0
    
    # Environment-specific success criteria
    if "Pendulum" in env_name:
        # For Pendulum, success is achieving return > -500
        successful = sum(1 for r in returns if r > -800)
    elif "Walker2d" in env_name:
        # For Walker2d, success is surviving 1000 steps with return > 1000
        successful = sum(1 for r, s in zip(returns, steps) if s >= 1000 and r > 1000)
        # Default: success is surviving for 1000 steps
        successful = sum(1 for s in steps if s >= 1000)
    
    return successful / total_episodes


def export_results_to_csv(results_data, csv_path="inference_results.csv"):
    """Export inference results to CSV file"""
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['task', 'seed', 'method', 'average_reward', 'success_rate', 'num_episodes', 'total_steps', 'standup_rate', 'standup_hold_rate', 'checkpoint', 'epoch']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the results
        writer.writerow(results_data)
    
    print(f"Results exported to CSV: {csv_path}")


def run_inference(env_name, data_path, policy_path, render_mode, num_envs, num_episodes, seed=None, flow_steps=1, csv_output=None, num_train_stats=None, dump_npz=None, dump_max=None, task=None, action_mode='normalize', checkpoint_name=None, epoch=None, jump_point=None):
    """Run inference with trained flow matching policy"""
    def _parse_flow_schedule(flow_steps_arg, jump_point_arg=None):
        """Return (ts, dts) with ts strictly increasing in [0,1], anchored at 0 and 1."""

        def _build_schedule_from_list(vals):
            interior: list[float] = []
            seen: set[float] = set()
            for raw in vals:
                v = float(raw)
                if v <= 0.0 or v >= 1.0:
                    if np.isclose(v, 0.0) or np.isclose(v, 1.0):
                        continue
                    raise ValueError(
                        "Flow schedule interior points must lie strictly between 0 and 1"
                    )
                if v not in seen:
                    seen.add(v)
                    interior.append(v)
            interior.sort()
            return [0.0] + interior + [1.0]

        ts: list[float]
        # Case 1: explicit list/tuple provided programmatically or via string literal
        if isinstance(flow_steps_arg, (list, tuple)):
            ts = _build_schedule_from_list(flow_steps_arg)
            if jump_point_arg is not None:
                print(
                    "ℹ️  Ignoring --jump_point because an explicit flow schedule list was provided."
                )
        else:
            if flow_steps_arg is None:
                s = "2"  # default to two points -> [0,1]
            else:
                s = str(flow_steps_arg).strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    parsed = json.loads(s)
                except Exception:
                    parsed = [float(x) for x in s[1:-1].split(',') if x.strip() != '']
                ts = _build_schedule_from_list(parsed)
                if jump_point_arg is not None:
                    print(
                        "ℹ️  Ignoring --jump_point because an explicit flow schedule list was provided."
                    )
            else:
                try:
                    n = int(float(s))
                except Exception as exc:
                    raise ValueError(
                        "--flow_steps must be an int like '4' or a list like '[0.25,0.75]'"
                    ) from exc
                n = max(1, n)
                ts = np.linspace(0.0, 1.0, n + 1).tolist()
                if jump_point_arg is not None:
                    jp = float(jump_point_arg)
                    if not (0.0 < jp < 1.0):
                        raise ValueError("--jump_point must be strictly between 0 and 1")
                    if n < 2:
                        raise ValueError(
                            "--jump_point requires --flow_steps >= 2 to allocate steps before the jump"
                        )
                    left_steps = n - 1
                    left = np.linspace(0.0, jp, left_steps + 1).tolist()
                    ts = left + [1.0]

        if len(ts) < 2:
            raise ValueError("Flow schedule must contain at least two time points (including 0 and 1)")
        if not np.isclose(ts[0], 0.0) or not np.isclose(ts[-1], 1.0):
            raise ValueError("Flow schedule must start at 0 and end at 1")
        for i in range(1, len(ts)):
            if not (ts[i] > ts[i - 1]):
                raise ValueError("Flow schedule must be strictly increasing")

        dts = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        return ts, dts
    # Reproducibility
    if seed is not None:
        set_seed(seed)
        print(f"🎲 Set seed to {seed} for reproducibility")
        print("ℹ️  No seed specified, using random initialization")

    # Use tasks_paths.json to resolve task/env/data when available
    tasks_paths = load_tasks_paths()
    if task and task in tasks_paths:
        task_info = tasks_paths[task]
        data_path = task_info["data_path"]
        if env_name == "Pendulum-v1":
            env_name = task_info.get("env_name", env_name)
        print(f"🎯 Using task '{task}' with data path: {data_path}")
        print(f"   Environment: {env_name}")
    elif (not task) and data_path:
        # Infer task by matching data_path from tasks_paths.json (robust to separators/case)
        def _norm(p: str) -> str:
            try:
                return os.path.normpath(str(p)).replace("\\", "/").lower()
            except Exception:
                return str(p)
        arg_norm = _norm(str(data_path))
        for k, v in tasks_paths.items():
            conf_path = v.get("data_path", "")
            if _norm(conf_path) == arg_norm or os.path.basename(_norm(conf_path)) == os.path.basename(arg_norm):
                task = k
                if env_name == "Pendulum-v1":
                    env_name = v.get("env_name", env_name)
                print(f"🎯 Inferred task '{task}' from data_path match in {TASKS_PATHS_FILE}")
                print(f"   Environment: {env_name}")
                break

    # Determine data dimensions strictly from tasks_paths.json to avoid extension-based ambiguity
    if not task:
        raise ValueError(
            f"Unable to resolve task from data_path. Provide --task matching an entry in {TASKS_PATHS_FILE} or ensure data_path matches an entry."
        )
    if task not in tasks_paths or 'obs_dim' not in tasks_paths[task] or 'action_dim' not in tasks_paths[task]:
        raise ValueError(
            f"Task '{task}' missing or incomplete in {TASKS_PATHS_FILE}. "
            f"Add obs_dim and action_dim for reliable inference."
        )
    data_obs_dim = int(tasks_paths[task]['obs_dim'])
    data_action_dim = int(tasks_paths[task]['action_dim'])
    print(f"📋 Using dims from {TASKS_PATHS_FILE} for task '{task}': obs_dim={data_obs_dim}, action_dim={data_action_dim}")
    # Use provided environment (from args or tasks_paths.json)
    print(f"📊 Using environment: {env_name} (obs_dim={data_obs_dim}, action_dim={data_action_dim})")

    # Environment setup
    if num_envs > 1:
        def make_env(env_name: str, render_mode: str = "rgb_array", env_seed: int | None = None):
            def _init():
                env_local = gym.make(env_name, render_mode=render_mode)
                if env_seed is not None:
                    env_local.reset(seed=env_seed)
                return env_local
            return _init

        env_fns = []
        for idx in range(num_envs):
            env_seed = (seed + idx) if seed is not None else None
            env_fns.append(make_env(env_name, render_mode=render_mode, env_seed=env_seed))
        env = AsyncVectorEnv(env_fns)
    else:
        env = gym.make(env_name, render_mode=render_mode)

    env_obs_dim = env.observation_space.shape[-1]
    env_action_dim = env.action_space.shape[-1]
    # Try to read environment id and time limit for context
    try:
        env_id = env.spec.id if getattr(env, "spec", None) else str(env)
        max_steps = getattr(env.spec, "max_episode_steps", None) if getattr(env, "spec", None) else None
        if max_steps is not None:
            print(f"📊 Environment: {env_id}, {num_envs} env(s), env_obs_dim={env_obs_dim}, env_action_dim={env_action_dim}, time_limit={max_steps}")
            print(f"📊 Environment: {env_id}, {num_envs} env(s), env_obs_dim={env_obs_dim}, env_action_dim={env_action_dim}")
    except Exception:
        print(f"📊 Environment: {num_envs} env(s), env_obs_dim={env_obs_dim}, env_action_dim={env_action_dim}")
    
    # Use data dimensions for policy (should match environment now)
    obs_dim = data_obs_dim
    action_dim = data_action_dim
    
    # Verify dimensions match environment
    if obs_dim != env_obs_dim or action_dim != env_action_dim:
        print(f"⚠️  Dimension mismatch: env({env_obs_dim},{env_action_dim}) vs data({obs_dim},{action_dim})")
        print("⚠️  Environment auto-detection may have failed")
        print(f"Dimensions match: obs_dim={obs_dim}, action_dim={action_dim}")

    # Load policy with auto-detection of configuration
    try:
        # Try to load policy with embedded configuration
        policy_checkpoint = torch.load(policy_path, map_location=device)
        if isinstance(policy_checkpoint, dict) and 'model_state_dict' in policy_checkpoint:
            required_cfg_keys = [
                'obs_hidden_dims',
                'policy_hidden_dims',
                'dropout_p',
                'layernorm',
                'arch',
                'unet_base_ch',
                'unet_depth',
            ]
            missing_cfg = [k for k in required_cfg_keys if k not in policy_checkpoint]
            if missing_cfg:
                raise ValueError(
                    "Policy file missing configuration keys: "
                    f"{missing_cfg}. Please retrain policy with updated train_flow.py"
                )

            detected_obs_hidden = policy_checkpoint['obs_hidden_dims']
            detected_policy_hidden = policy_checkpoint['policy_hidden_dims']
            detected_dropout_p = policy_checkpoint['dropout_p']
            detected_layernorm = policy_checkpoint['layernorm']
            detected_arch = policy_checkpoint['arch']
            detected_unet_base_ch = policy_checkpoint['unet_base_ch']
            detected_unet_depth = policy_checkpoint['unet_depth']
            
            # Use detected config from policy file
            actual_obs_hidden = detected_obs_hidden
            actual_policy_hidden = detected_policy_hidden
            
            policy = FlowMatchingPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                horizon=H,
                obs_hidden=actual_obs_hidden,
                policy_hidden=actual_policy_hidden,
                dropout_p=detected_dropout_p,
                layernorm=detected_layernorm,
                arch=detected_arch,
                unet_base_ch=detected_unet_base_ch,
                unet_depth=detected_unet_depth,
            ).to(device)
            
            # Handle backward compatibility for old model weights
            state_dict = policy_checkpoint['model_state_dict']
            
            # Check if this is an old model format and convert keys
            if 'obs_encoder.0.weight' in state_dict and 'obs_encoder_mlp.0.weight' not in state_dict:
                print("🔄 Converting old model format to new format...")
                new_state_dict = {}
                
                # Convert obs_encoder -> obs_encoder_mlp
                for key, value in state_dict.items():
                    if key.startswith('obs_encoder.'):
                        new_key = key.replace('obs_encoder.', 'obs_encoder_mlp.')
                        new_state_dict[new_key] = value
                        new_state_dict[key] = value
                
                # Add dummy time_mlp weights for MLP architecture (they won't be used)
                if detected_arch == "mlp":
                    # Initialize time_mlp with dummy weights (won't be used in MLP forward pass)
                    import torch.nn as nn
                    time_mlp = nn.Sequential(
                        nn.Linear(64, actual_obs_hidden[-1]), 
                        nn.SiLU(), 
                        nn.Linear(actual_obs_hidden[-1], actual_obs_hidden[-1])
                    )
                    time_mlp_state = time_mlp.state_dict()
                    for key, value in time_mlp_state.items():
                        new_state_dict[f'time_mlp.{key}'] = value
                
                state_dict = new_state_dict
                print("Model format conversion completed")
            
            policy.load_state_dict(state_dict)
            
            print(f"Loaded policy: arch={detected_arch}, obs_hidden={actual_obs_hidden}, policy_hidden={actual_policy_hidden}, dropout_p={detected_dropout_p}")
    except Exception as e:
        print(f"Error loading policy: {e}")
        raise
    
    policy.eval()
    print(f"Flow matching policy loaded from {policy_path}")

    # Load normalization stats using shared training loader
    data_dict = load_data(data_path)
    observations = data_dict['observations']
    actions = data_dict['actions']
    
    if num_train_stats is not None:
        print(f"📊 Recomputing normalization stats using first {num_train_stats} samples (matching training)")
        train_obs = observations[:num_train_stats]
        train_actions = actions[:num_train_stats]
        obs_mean, obs_std = np.mean(train_obs, axis=0), np.std(train_obs, axis=0)
        action_mean, action_std = np.mean(train_actions, axis=0), np.std(train_actions, axis=0)
        obs_mean, obs_std = data_dict['obs_mean'], data_dict['obs_std']
        action_mean, action_std = data_dict['action_mean'], data_dict['action_std']
        print("📊 Using normalization stats from unified loader")

    obs_std = np.maximum(obs_std, 1e-8)
    action_std = np.maximum(action_std, 1e-8)

    all_returns = []
    all_steps = []  # Track steps per episode
    term_count = 0  # Episodes that terminated (success or env-defined)
    trunc_count = 0  # Episodes that truncated due to time limit
    # Custom sparse-style termination stats (for AdroitHandPen)
    custom_success_episodes = 0
    custom_failure_episodes = 0
    timeout_without_custom = 0

    # Special standup metrics: first obs > 1.2 implies stand-up; after that, if never < 1.1 => held
    is_standup_env = ('standup' in str(env_name).lower()) or ('humanoidstandup' in str(env_name).lower())
    stood_up_count = 0
    stood_up_and_held_count = 0

    print(f"🚀 Starting inference for {num_episodes} episodes with {num_envs} environment(s)")
    episode_pbar = tqdm(range(num_episodes), desc="Inference", unit="episode")

    # Pre-compute normalization/scaling tensors outside the loop for efficiency
    obs_mean_tensor = torch.from_numpy(obs_mean).to(device, dtype=torch.float32)
    obs_std_tensor = torch.from_numpy(obs_std).to(device, dtype=torch.float32)
    action_mean_tensor = torch.from_numpy(action_mean).to(device, dtype=torch.float32)
    action_std_tensor = torch.from_numpy(action_std).to(device, dtype=torch.float32)
    # Build flow schedule (time grid ts and step sizes dts)
    ts, dts = _parse_flow_schedule(flow_steps, jump_point)
    # Resolve action bounds for scaling mode
    if action_mode == 'scale':
        try:
            with open(TASKS_PATHS_FILE, 'r') as f:
                tasks_paths = json.load(f)
        except Exception:
            tasks_paths = {}
        low = high = None
        for k, v in tasks_paths.items():
            if str(v.get('data_path','')) == str(data_path):
                # Accept scalar/len-1 shorthand and accidental trailing space key
                raw_low = v.get('action_low', v.get('action_low '))
                raw_high = v.get('action_high', v.get('action_high '))
                if raw_low is not None and raw_high is not None:
                    low = np.array(raw_low, dtype=np.float32)
                    high = np.array(raw_high, dtype=np.float32)
                break
        if low is None or high is None:
            raise ValueError("action_mode='scale' requires action_low/high in tasks_paths.json for this data_path")
        # Expand scalar/len-1 to action_dim
        if low.ndim == 0 or (low.ndim == 1 and low.shape[0] == 1):
            low = np.repeat(low.reshape(1), action_dim).astype(np.float32)
        if high.ndim == 0 or (high.ndim == 1 and high.shape[0] == 1):
            high = np.repeat(high.reshape(1), action_dim).astype(np.float32)
        low_t = torch.from_numpy(low).to(device, dtype=torch.float32)
        high_t = torch.from_numpy(high).to(device, dtype=torch.float32)
        scale_t = (high_t - low_t)

    # Optional capture buffers for KNN analysis
    capture_obs = dump_npz is not None
    captured_obs = []
    captured_actions = []

    for ep in episode_pbar:
        if num_envs == 1:
            # Single environment logic
            obs, _ = env.reset(seed=(seed + ep) if seed is not None else None)
            episode_reward = 0.0
            terminated = False
            truncated = False
            step_count = 0
            custom_ended = False
            
            # Track stand-up status for this episode (HumanoidStandup)
            stood_up = False
            dropped_after = False

            while not (terminated or truncated):
                # Vectorized normalization
                obs_tensor = torch.from_numpy(obs).to(device, dtype=torch.float32)
                obs_norm = ((obs_tensor - obs_mean_tensor) / obs_std_tensor).unsqueeze(0)
                
                with torch.no_grad():
                    action_seq = torch.randn((1, action_dim * H), device=device, dtype=torch.float32)
                    
                    # Euler integration over (possibly) non-uniform time steps
                    for i, t_val in enumerate(ts[:-1]):
                        dt = dts[i]
                        t_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.float32)
                        v = policy(obs_norm, action_seq, t_tensor)
                        action_seq.add_(v, alpha=dt)
                        if action_mode == 'scale':
                            action_seq.clamp_(-1.0, 1.0)
                
                # Efficient denormalization and reshaping
                if action_mode == 'scale':
                    action_denorm = (action_seq + 1.0) * 0.5 * scale_t + low_t
                else:
                    action_denorm = action_seq * action_std_tensor + action_mean_tensor
                action = action_denorm.view(H, action_dim)[0].cpu().numpy()
                # Clip action to env bounds to avoid invalid/out-of-range torques
                try:
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                except Exception:
                    pass

                # Update stand-up tracking using raw observation (first component)
                if is_standup_env and isinstance(obs, np.ndarray) and obs.size > 0:
                    v0 = float(obs[0])
                    if not stood_up and v0 > 1.2:
                        stood_up = True
                    elif stood_up and (not dropped_after) and v0 < 1.1:
                        dropped_after = True
                # Capture before stepping env
                if capture_obs:
                    if (dump_max is None) or (len(captured_obs) < dump_max):
                        captured_obs.append(obs.copy())
                        captured_actions.append(action.copy())
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1

                # Custom sparse-style termination for AdroitHandPen: only reward==10 implies success/termination
                if 'AdroitHandPen' in env_name:
                    if np.isclose(reward, 10.0, atol=1e-6):
                        terminated = True
                        custom_ended = True
                        custom_success_episodes += 1
                
            all_returns.append(episode_reward)
            all_steps.append(step_count)
            if is_standup_env:
                if stood_up:
                    stood_up_count += 1
                    if not dropped_after:
                        stood_up_and_held_count += 1
            # Count termination type
            if terminated:
                term_count += 1
            elif truncated:
                trunc_count += 1
                if ('AdroitHandPen' in env_name) and (not custom_ended):
                    timeout_without_custom += 1
            episode_pbar.set_postfix({'Return': f'{episode_reward:.2f}', 'Steps': step_count})
            continue

        # Multi-environment logic
        if seed is not None:
            obs, _ = env.reset(seed=[seed + ep * num_envs + i for i in range(num_envs)])
        else:
            obs, _ = env.reset()

        episode_rewards = np.zeros(num_envs)
        episode_steps = np.zeros(num_envs, dtype=int)
        episodes_collected = 0
        current_round_returns = []
        active_envs = np.ones(num_envs, dtype=bool)  # Track which environments are still active
        custom_ended = np.zeros(num_envs, dtype=bool)
        # Standup tracking per environment
        if is_standup_env:
            stood_up_flags = np.zeros(num_envs, dtype=bool)
            dropped_after_flags = np.zeros(num_envs, dtype=bool)

        while episodes_collected < num_envs and np.any(active_envs):
                # Only process active environments
                active_indices = np.where(active_envs)[0]
                if len(active_indices) == 0:
                    break
                
                # Get observations for active environments
                active_obs = obs[active_indices]
                obs_tensor = torch.from_numpy(active_obs).to(device, dtype=torch.float32)
                obs_norm = (obs_tensor - obs_mean_tensor) / obs_std_tensor
                
                with torch.no_grad():
                    action_seq = torch.randn((len(active_indices), action_dim * H), device=device, dtype=torch.float32)
                    
                    # Euler integration for active environments
                    for i, t_val in enumerate(ts[:-1]):
                        dt = dts[i]
                        t_tensor = torch.full((len(active_indices), 1), t_val,
                                              device=device, dtype=torch.float32)
                        v = policy(obs_norm, action_seq, t_tensor)
                        action_seq.add_(v, alpha=dt)
                        if action_mode == 'scale':
                            action_seq.clamp_(-1.0, 1.0)
                
                # Denormalize and reshape actions
                if action_mode == 'scale':
                    action_denorm = (action_seq + 1.0) * 0.5 * scale_t + low_t
                else:
                    action_denorm = action_seq * action_std_tensor + action_mean_tensor
                actions_for_active = action_denorm.view(len(active_indices), H, action_dim)[:, 0, :].cpu().numpy()
                # Clip actions for active envs to bounds
                try:
                    low = env.single_action_space.low
                    high = env.single_action_space.high
                    actions_for_active = np.clip(actions_for_active, low, high)
                except Exception:
                    pass
                # Capture for active envs before stepping
                if capture_obs:
                    if (dump_max is None) or (len(captured_obs) < dump_max):
                        remaining = None if dump_max is None else (dump_max - len(captured_obs))
                        if remaining is None or len(active_indices) <= remaining:
                            captured_obs.extend([o.copy() for o in active_obs])
                            captured_actions.extend([a.copy() for a in actions_for_active])
                        elif remaining > 0:
                            captured_obs.extend([o.copy() for o in active_obs[:remaining]])
                            captured_actions.extend([a.copy() for a in actions_for_active[:remaining]])
                
                # Prepare full action array for all environments (inactive ones get dummy actions)
                full_actions = np.zeros((num_envs, action_dim))
                full_actions[active_indices] = actions_for_active
                
                # Step all environments (gymnasium handles inactive ones)
                obs, reward, terminated, truncated, infos = env.step(full_actions)
                
                # Update rewards and steps for active environments only
                episode_rewards[active_indices] += reward[active_indices]
                episode_steps[active_indices] += 1

                # Update standup flags for active envs using raw obs first component
                if is_standup_env:
                    try:
                        vals = active_obs[:, 0].astype(np.float32)
                    except Exception:
                        vals = None
                    if vals is not None:
                        prev_cross = stood_up_flags[active_indices]
                        # Newly crossed
                        new_cross = (~prev_cross) & (vals > 1.2)
                        if np.any(new_cross):
                            for i_local, env_idx in enumerate(active_indices):
                                if new_cross[i_local]:
                                    stood_up_flags[env_idx] = True
                        # Dropped after crossing
                        check_drop = prev_cross & (~dropped_after_flags[active_indices]) & (vals < 1.1)
                        if np.any(check_drop):
                            for i_local, env_idx in enumerate(active_indices):
                                if check_drop[i_local]:
                                    dropped_after_flags[env_idx] = True
                
                # Custom sparse-style termination on reward (AdroitHandPen): only reward==10
                if 'AdroitHandPen' in env_name:
                    rew_act = reward[active_indices]
                    succ_mask = np.isclose(rew_act, 10.0, atol=1e-6)
                    # Immediately record custom-done envs and mark inactive
                    if np.any(succ_mask):
                        for i_local, env_idx in enumerate(active_indices):
                            if succ_mask[i_local]:
                                all_returns.append(episode_rewards[env_idx])
                                all_steps.append(episode_steps[env_idx])
                                current_round_returns.append(episode_rewards[env_idx])
                                custom_success_episodes += 1
                                custom_ended[env_idx] = True
                                active_envs[env_idx] = False
                                episodes_collected += 1
                        # Continue to next loop iteration with updated active set
                        continue

                # Check for completed episodes by env signals
                done = np.array(terminated) | np.array(truncated)
                
                completed_indices = active_indices[done[active_indices]]
                
                for env_idx in completed_indices:
                    all_returns.append(episode_rewards[env_idx])
                    all_steps.append(episode_steps[env_idx])
                    current_round_returns.append(episode_rewards[env_idx])
                    if is_standup_env:
                        if stood_up_flags[env_idx]:
                            stood_up_count += 1
                            if not dropped_after_flags[env_idx]:
                                stood_up_and_held_count += 1
                    # Count termination type per env
                    if terminated[env_idx]:
                        term_count += 1
                    elif truncated[env_idx]:
                        trunc_count += 1
                        if ('AdroitHandPen' in env_name) and (not custom_ended[env_idx]):
                            timeout_without_custom += 1
                    active_envs[env_idx] = False  # Mark environment as inactive
                    episodes_collected += 1
        # Display progress for multi-env episodes
        if current_round_returns:
            avg_ret = np.mean(current_round_returns)
            episode_pbar.set_postfix({
                'Avg_Return': f'{avg_ret:.2f}',
                'Completed': len(current_round_returns),
                'Total_Collected': len(all_returns)
            })

        # Periodic memory cleanup
        if ep % 10 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

    env.close()
    final_avg = np.mean(all_returns)
    final_std = np.std(all_returns)
    total = len(all_steps)
    
    # Calculate success rate (override for Standup with stand/hold rates)
    success_rate = calculate_success_rate(all_returns, all_steps, env_name)
    standup_rate = None
    standup_hold_rate = None
    if is_standup_env and len(all_returns) > 0:
        standup_rate = stood_up_count / len(all_returns)
        standup_hold_rate = stood_up_and_held_count / len(all_returns)
        success_rate = standup_rate  # replace success rate with stand-up rate
    
    # Method name (flow matching baseline)
    method_name = "FM"
    
    print("=" * 50)
    print("🎯 INFERENCE RESULTS")
    print(f"Environment: {env_name}")
    print(f"Episodes: {num_episodes}")
    print(f"Environments per episode: {num_envs}")
    print(f"Total environment runs: {len(all_returns)}")
    print(f"Average return: {final_avg:.2f} ± {final_std:.2f}")
    if sum(all_steps) > 0:
        avg_per_step = (sum(all_returns) / sum(all_steps))
        print(f"Avg per-step reward: {avg_per_step:.4f}")
    print(f"Min return: {np.min(all_returns):.2f}")
    print(f"Max return: {np.max(all_returns):.2f}")
    if is_standup_env and standup_rate is not None:
        print(f"Stand-up: {standup_rate:.3f} ({standup_rate*100:.1f}%), Held: {standup_hold_rate:.3f} ({standup_hold_rate*100:.1f}%)")
        print(f"Success rate: {success_rate:.3f} ({success_rate*100:.1f}%)")
    print(f"Episodes terminated: {term_count}")
    print(f"Episodes truncated (time limit): {trunc_count}")
    # Diagnostic line for suspected env-logic inconsistency
    if 'AdroitHandPen' in env_name:
        print(f"Custom sparse-style termination success: {custom_success_episodes}, failure: {custom_failure_episodes}")
        print(f"Reached max steps without custom termination: {timeout_without_custom}")
    print(f"Method: {method_name}")
    print("=" * 50)
    
    # Prepare data for CSV export
    results_data = {
        'task': env_name,
        'seed': seed if seed is not None else 'None',
        'method': method_name,
        'average_reward': round(final_avg, 4),
        'success_rate': round(success_rate, 4),
        'num_episodes': len(all_returns),
        'total_steps': sum(all_steps),
        # Optional standup metrics (None for non-standup envs)
        'standup_rate': round(standup_rate, 4) if ('standup_rate' in locals() and standup_rate is not None) else None,
        'standup_hold_rate': round(standup_hold_rate, 4) if ('standup_hold_rate' in locals() and standup_hold_rate is not None) else None,
        # Optional checkpoint metadata (for batch eval)
        'checkpoint': checkpoint_name,
        'epoch': epoch,
        # Custom sparse-style termination metrics for AdroitHandPen
        'custom_success': int(custom_success_episodes) if 'custom_success_episodes' in locals() else None,
        'custom_failure': int(custom_failure_episodes) if 'custom_failure_episodes' in locals() else None,
        'custom_timeout': int(timeout_without_custom) if 'timeout_without_custom' in locals() else None,
    }
    
    # Export to CSV
    csv_path = csv_output if csv_output is not None else "inference_results.csv"
    export_results_to_csv(results_data, csv_path)

    # Optionally dump captured (observation, action) pairs for KNN analysis
    if capture_obs and len(captured_obs) > 0:
        try:
            os.makedirs(os.path.dirname(dump_npz) or '.', exist_ok=True)
            np.savez(
                dump_npz,
                observations=np.asarray(captured_obs),
                actions=np.asarray(captured_actions),
                obs_mean=obs_mean,
                obs_std=obs_std,
                action_mean=action_mean,
                action_std=action_std,
                meta=np.array({
                    'env_name': env_name,
                    'flow_schedule': np.array(ts, dtype=np.float32),
                    'num_envs': num_envs,
                }, dtype=object)
            )
            print(f"💾 Dumped {len(captured_obs)} (obs, action) pairs to {dump_npz}")
        except Exception as e:
            print(f"⚠️  Failed to dump observations/actions to {dump_npz}: {e}")
    
    return results_data


def main():
    parser = argparse.ArgumentParser(description="Run inference with flow matching policy")
    parser.add_argument('--env_name', type=str, default='Pendulum-v1',
                        help="Environment name (default: Pendulum-v1)")
    parser.add_argument('--data_path', type=str, 
                        help="Path to expert data file or Minari dataset ID (e.g., D4RL/pen-expert-v2)")
    parser.add_argument('--policy_path', type=str, required=True,
                        help="Path to trained flow matching policy")
    parser.add_argument('--render_mode', choices=['human', 'rgb_array'], default='rgb_array',
                        help="Render mode: 'human' or 'rgb_array'")
    parser.add_argument('--num_envs', type=int, default=50,
                        help="Number of parallel environments (default: 50)")
    parser.add_argument('--num_episodes', type=int, default=10,
                        help="Number of episodes to run (default: 10)")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducibility (default: None)")
    parser.add_argument('--flow_steps', type=str, default='1',
                        help="Euler schedule: int N for uniform N steps, or a list of interior points (exclude 0 and 1) like [0.1,0.5] for non-uniform steps anchored at 0 and 1")
    parser.add_argument('--jump_point', type=float, default=None,
                        help="If set to a value in (0,1), uses one Euler step from jump_point to 1 and allocates the remaining steps uniformly from 0 to jump_point. Requires --flow_steps to be an integer >= 2.")
    parser.add_argument('--action_mode', type=str, choices=['normalize', 'scale'], default='normalize',
                        help="Action preprocessing: 'normalize' (z-score) or 'scale' to [-1,1] using tasks_paths.json bounds")
    parser.add_argument('--csv_output', type=str, default=None,
                        help="Path to CSV output file (default: inference_results.csv)")
    parser.add_argument('--num_train_stats', type=int, default=None,
                        help="Use first N samples for normalization stats to match training (default: None, use all data)")
    parser.add_argument('--dump_npz', type=str, default=None,
                        help="Optional path to save (observations, actions) pairs for KNN analysis (.npz)")
    parser.add_argument('--dump_max', type=int, default=None,
                        help="Optional cap on number of (obs, action) pairs to dump")
    # No explicit --task required; we infer from tasks_paths.json using data_path if not provided
    args = parser.parse_args()
    
    if args.render_mode == 'human':
        args.num_envs = 1
        print("ℹ️  Human rendering mode: using single environment")
        
    run_inference(
        env_name=args.env_name,
        data_path=args.data_path,
        policy_path=args.policy_path,
        render_mode=args.render_mode,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        seed=args.seed,
        flow_steps=args.flow_steps,
        jump_point=args.jump_point,
        csv_output=args.csv_output,
        num_train_stats=args.num_train_stats,
        dump_npz=args.dump_npz,
        dump_max=args.dump_max,
        task=None,
        action_mode=args.action_mode
    )

if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Build and save a reusable KNN index for training actions.

This fits a scikit‑learn NearestNeighbors model on the action corpus from a
dataset and saves it to disk (.pkl/.joblib). Later, load it in
`knn_action_distance.py` via `--index_path` to run fast queries without
rebuilding.

Example:
  python src/utils/knn_build_index.py --data_path data/pendulum_expert.npz --num_train 600
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import numpy as np

# Allow import from ICRA module path
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from knn_action_distance import load_dataset  # reuse loader


def main():
    parser = argparse.ArgumentParser(description="Build a reusable sklearn KNN index for actions")
    parser.add_argument("--data_path", type=str, required=True, help="Dataset with observations/actions (.npz or .pkl)")
    parser.add_argument("--metric", type=str, default="l2", choices=["l2", "l1", "cosine"], help="Distance metric")
    parser.add_argument("--algorithm", type=str, default="auto", choices=["auto", "ball_tree", "kd_tree", "brute"], help="NN algorithm")
    parser.add_argument("--leaf_size", type=int, default=40, help="Leaf size for BallTree/KDTree")
    parser.add_argument("--num_train", type=int, default=500, help="Use first N samples from dataset for the index (default: 500)")
    parser.add_argument("--output", type=str, default="knn_index.joblib", help="Output path (.pkl/.joblib). If left as default, will be renamed to include task name from tasks_paths.json")

    args = parser.parse_args()

    # Map metrics
    metric_map = {"l2": "euclidean", "l1": "manhattan", "cosine": "cosine"}
    metric = metric_map[args.metric]

    # Load dataset actions
    _obs, actions, *_ = load_dataset(args.data_path)
    # Deterministic training subset: first num_train samples (default: 500)
    if args.num_train is not None:
        n = int(min(max(args.num_train, 0), len(actions)))
        actions = actions[:n]
        corpus_indices = np.arange(n)   
    else:
        corpus_indices = np.arange(len(actions))

    # Fit sklearn index
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
    except Exception as e:
        raise RuntimeError("scikit-learn not available. Please install scikit-learn to build an index.") from e

    nn = NearestNeighbors(algorithm=args.algorithm, metric=metric, leaf_size=args.leaf_size)
    nn.fit(actions)

    # Attach small metadata for robustness
    nn._icra_meta = {
        "metric": args.metric,
        "algorithm": args.algorithm,
        "leaf_size": args.leaf_size,
        "num_corpus": int(actions.shape[0]),
        "dim": int(actions.shape[1]) if actions.ndim > 1 else 1,
        "data_path": args.data_path,
        "num_train": int(actions.shape[0]),
    }
    # Store mapping from index rows to dataset rows for neighbor state lookup
    try:
        import numpy as _np  # local scope guard
        nn._icra_corpus_indices = _np.asarray(corpus_indices, dtype=int)
    except Exception:
        pass

    # Resolve task name from tasks_paths.json (best-effort)
    def resolve_task_name(data_path: str) -> str:
        tasks_file = os.path.join(REPO_ROOT, 'tasks_paths.json')
        try:
            if os.path.exists(tasks_file):
                with open(tasks_file, 'r') as f:
                    tasks = json.load(f)
                dp_real = os.path.realpath(data_path)
                for k, v in tasks.items():
                    vpath = str(v.get('data_path', ''))
                    if not vpath:
                        continue
                    if os.path.realpath(os.path.join(REPO_ROOT, vpath)) == dp_real or vpath == data_path:
                        return str(k)
                # fallback: use env_name if present
                for k, v in tasks.items():
                    vpath = str(v.get('data_path', ''))
                    if vpath == data_path:
                        env = v.get('env_name', None)
                        if env:
                            return env.replace('/', '_')
        except Exception:
            pass
        # final fallback to file stem
        base = os.path.basename(data_path)
        return os.path.splitext(base)[0]

    task_name = resolve_task_name(args.data_path)

    # Decide output path (inject task name when using default)
    output_path = args.output
    if args.output == "knn_index.joblib":
        output_path = f"knn_index_{task_name}.joblib"

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        import joblib  # type: ignore
        joblib.dump(nn, output_path)
        print(f"✅ Saved KNN index to {output_path} (joblib)")
    except Exception:
        import pickle
        with open(output_path, "wb") as f:
            pickle.dump(nn, f)
        print(f"✅ Saved KNN index to {output_path} (pickle)")


if __name__ == "__main__":
    main()

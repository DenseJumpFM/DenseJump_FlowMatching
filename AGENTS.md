# Repository Guidelines

## Project Structure & Module Organization
Core policy architecture lives in `model.py`, including the `FlowMatchingPolicy` variants (MLP, state UNet, RGB UNet) and utility builders. Training orchestration, dataset ingestion, and checkpointing sit in `train_flow.py`; it reads expert trajectories from `.npz/.hdf5` files, handles device selection, and spins up TensorBoard logging under `logs/flow_matching/<env>/`. Evaluation pipelines are in `inference.py`, which registers Gymnasium robotics tasks, supports CSV exports, and consumes dataset metadata declared in `tasks_paths.json`. Optional Nearest-Neighbor tooling resides in `knn/`, reusing the shared dataset loader. Keep large datasets in `data/` (gitignored) and stage run artifacts in `logs/`.

## Build, Test, and Development Commands
Use `python train_flow.py --data_path pendulum_expert.npz --num_train 600 --num_epochs 10000 --seed 0` for baseline Pendulum training; adjust `--arch`, `--flow_steps`, or dataset flags per `tasks_paths.json`. Run `python inference.py --data_path pendulum_expert.npz --policy_path logs/flow_matching/Pendulum-v1/<run>/policy_epochXXXX.pth --num_episodes 10` to benchmark a checkpoint and record `inference_results.csv`. Generate reusable action indices with `python knn/knn_build_index.py --data_path data/pen_expert.hdf5 --num_train 500`. Add `PYTHONPATH=.` if invoking scripts from outside the repo root.

## Coding Style & Naming Conventions
Follow the existing PEP 8 style: four-space indentation, 100-character soft line limit, and snake_case for functions/variables. Classes use PascalCase (`FlowMatchingPolicy`). Prefer explicit type hints and module-level docstrings, as seen in `model.py`. Keep tensors on the device returned by `train_flow.device` and guard optional dependencies (`minari`, `h5py`, `scikit-learn`) with informative errors.

## Testing Guidelines
There is no automated test harness yet; rely on deterministic seeds (`--seed`, `train_flow.set_seed`) and smoke tests via `inference.py`. When adding tests, mirror the data-loading helpers in `train_flow.py` and target critical flows: time-sampling utilities, loss computation, and KNN tooling. Name new tests `test_<feature>.py` under a `tests/` directory and invoke them with `pytest`.

## Commit & Pull Request Guidelines
Recent history uses terse messages; please adopt imperative present-tense subjects (`Add flow schedule sampler`). Reference tasks or datasets touched, e.g., `Update walker2d loader for Minari`. For pull requests, include: scope summary, reproduced command (`python train_flow.py ...`), dataset prerequisites, and before/after metrics or charts when altering policy behaviour. Attach CSV snippets or TensorBoard screenshots when results change.

## Data & Configuration Tips
Maintain `tasks_paths.json` entries when introducing new trajectories so `inference.py` can auto-map environments. Store credentials-sensitive paths via environment variables rather than committing absolute paths. Before large runs, clear GPU cache with `torch.cuda.empty_cache()` hooks already present in training loops instead of manual `gc.collect()` calls.

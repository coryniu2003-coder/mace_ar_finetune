# MACE Fine‑Tuning Log for Solid Ar

This note consolidates everything that went into fine‑tuning the public MACE MP‑0b3 checkpoint on the professor’s FCC Ar reference data (provided by Prof. Andreas Hermann, University of Edinburgh), plus the way the model is wired into the LAMMPS decks. It is meant to be copy‑ready for GitHub so future work can build on (or reproduce) the pipeline without digging through terminal history.

## Repository layout

- `DATASET_PREP.md` – original data conversion notes (VASP → ASE → XYZ).
- `T-{300,900,1500,2500}/` – professor’s raw runs (vasprun.xml, XDATCAR, etc.).
- `T-*.traj` – ASE trajectories (5 k frames each, 108‑atom cell).
- `ar_train|val|test.xyz` – merged datasets with metadata tags.
- `run_ar_finetune.sh` – resilient training wrapper (single‑epoch scheduling + resume logic).
- `mace_runs/ar_finetune/` – logs, checkpoints, compiled models.
- `~/lammps/model-setups/mace_mp0_new/` – TorchScript export + LAMMPS decks + helper script.
- `~/Ar_result/mace_new/` – archived production trajectories (each run stored in its own folder).

## 1. Dataset preparation (summary)

Documented in `DATASET_PREP.md`, but the essentials are:

```bash
python vasp2ase.py T-300/vasprun.xml  -o T-300.traj
python vasp2ase.py T-900/vasprun.xml  -o T-900.traj
python vasp2ase.py T-1500/vasprun.xml -o T-1500.traj
python vasp2ase.py T-2500/vasprun.xml -o T-2500.traj
```

Each `.traj` (5 k frames) was stratified into 4 k/500/500 splits and written to XYZ with per‑frame tags (`config_type=Ar_<temp>K`, `temperature`, `frame_source`). The resulting dataset sizes:

| File           | Frames | Contents                                           |
|----------------|--------|----------------------------------------------------|
| `ar_train.xyz` | 16,000 | 4 k frames per temperature, sorted                 |
| `ar_val.xyz`   | 2,000  | 500 frames per temperature                         |
| `ar_test.xyz`  | 2,000  | 500 frames per temperature (held out for metrics)  |

## 2. Environment & tooling

- Conda env: `mace-env` (PyTorch 2.4 CUDA 12.4 + `mace-torch==0.3.14`, ASE, pymatgen).
- GPU: RTX 3080 (10 GB VRAM, single card used for *all* training and inference runs); CPU: Threadripper 3990X (only 1% CPU usage, can be replaced by any other CPU).
- Key env vars before training (or add to `.bashrc`):

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mace-env
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The custom script also forces `TORCH_LOAD_WEIGHTS_ONLY=0` to keep `torch.load` compatible with the MP‑0b3 checkpoint.

## 3. Training workflow (`run_ar_finetune.sh`)

The wrapper (`fcc-p-10/run_ar_finetune.sh`) prevents line‑wrap mistakes, tracks the next epoch target, and automatically resumes from the latest checkpoint (`--restart_latest`). Highlights:

- Explicit logs, checkpoints, and results roots under `fcc-p-10/mace_runs/ar_finetune/`.
- `next_epoch.txt` records the next epoch index (starting at 1 because epoch‑0 is the initial fine‑tune pass).
- Accepts environment overrides:
  - `MAX_EPOCHS` – upper bound per launch (defaults to the value in `next_epoch.txt`).
  - `MACE_BATCH_SIZE` – reduce to 2 if VRAM is tight.
  - `AUTO_LOOP=1` + `SLEEP_BETWEEN=60` – optional “keep going” mode (disabled by default after repeated filesystem hiccups).

### Running one epoch at a time

```bash
cd ~/fcc-p-10
MAX_EPOCHS=$(<mace_runs/ar_finetune/next_epoch.txt) \
    ./run_ar_finetune.sh
```

Each successful epoch bumps `next_epoch.txt` by one and writes:

- checkpoint: `mace_runs/ar_finetune/checkpoints/ar_finetune_run-123_epoch-<N>.pt`
- compiled model: `mace_runs/ar_finetune/ar_finetune_compiled.model`
- JSONL log: `mace_runs/ar_finetune/results/ar_finetune_run-123_train.txt`

If the job crashes (common when `/home` flips read‑only), `next_epoch.txt` is **not** incremented, so simply rerun the script after fixing the underlying I/O issue (see below).

### Common pitfalls & fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: CUDA out of memory ... cat` | Fragmented VRAM during resume | Keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and drop `MACE_BATCH_SIZE` to 2 if needed. |
| `OSError: [Errno 5/30] Input/output error / Read-only file system` | NVMe hiccup remounting `/home` read-only | Capture `journalctl -k -b | tail -n 200`, plan a `sudo touch /forcefsck && sudo reboot`, and rerun once the filesystem is writable again (check with `mount | grep nvme0n1p4`). |
| `Unknown scheduler: 'cosine'` | `mace_run_train` ≥0.3.14 removed the cosine scheduler | Use `--scheduler ReduceLROnPlateau` (already baked into the script). |
| Command-line JSON errors (`{'Ar_300K':...} not found`) | Shell wrapped arguments | Always run through the script; it handles escaping. |

## 4. Training history & metrics

After two clean epochs (epoch indices 0 and 1 completed), the evaluation block in `results/ar_finetune_run-123_train.txt` shows:

| Split / Config | RMSE(E) [meV/atom] | RMSE(F) [meV/Å] | Relative F RMSE [%] |
|----------------|--------------------|-----------------|---------------------|
| Train (all)    | 0.3                | **7.7**         | 1.04                |
| Valid (all)    | 0.3                | **7.7**         | 1.03                |
| Test 300 K     | 0.3                | **6.8**         | 2.20                |
| Test 900 K     | 0.3                | **6.9**         | 1.22                |
| Test 1500 K    | 0.3                | **7.1**         | 0.95                |
| Test 2500 K    | 0.3                | **9.3**         | 0.85                |

Force MAE at epoch 1 is ~6 meV/Å, down from the ~0.53 eV/Å baseline of the untouched MP‑0b3 model. Additional epochs can be scheduled by re‑running the script (the next target in `next_epoch.txt` is currently 3).

## 5. Exported artifacts

Located under `~/lammps/model-setups/mace_mp0_new/`:

| File | Description |
|------|-------------|
| `ar_finetune_epoch1.model` | Copy of `mace_runs/ar_finetune/checkpoints/ar_finetune_run-123_epoch-1.pt`. |
| `ar_finetune_epoch1_compiled.model` | Serialized metadata for quick reload/testing. |
| `ar_finetune_epoch1.model-lammps.pt` | TorchScript export produced via `mace_create_lammps_model ... --head Default --dtype float32`. |
| `mace_mp0.in`, `mace_mp0_test.in` | NVT input decks already pointing at the new TorchScript file. |
| `run_mace_mp0.sh` | GPU wrapper that streams dumps via `/dev/shm`, then archives them. |

The original MP‑0b3 checkpoint (`mace-mp-0b3-medium.model`) and its TorchScript export remain in the same folder for A/B testing.

## 6. LAMMPS integration & evaluation

### Default run (100k steps @ 300 K)

```bash
cd ~/lammps/model-setups/mace_mp0_new
./run_mace_mp0.sh            # uses mace_mp0.in (NVT 300 K)
```

Outputs are automatically organized under `/tmp/mace_mp0_runs/<timestamp>/` and then manually copied into `~/Ar_result/mace_new/<timestamp>/`. The sample run you asked to archive now lives in `~/Ar_result/20251121-112158/`.

Key script overrides (export before running):

| Variable | Meaning | Default |
|----------|---------|---------|
| `TOTAL_STEPS` | Number of MD steps | 100000 |
| `CHUNK_STEPS` | Dump chunk size | 2000 |
| `OUTPUT_DIR`  | Destination (e.g. `~/Ar_result/mace_new`) | `/tmp/mace_mp0_runs` |
| `RUN_LABEL`   | Folder name instead of timestamp | autogenerated |

Example storing directly under `~/Ar_result/mace_new/300K_latest`:

```bash
OUTPUT_DIR=~/Ar_result/mace_new \
RUN_LABEL=300K_latest \
./run_mace_mp0.sh
```

### Multi-temperature decks

Create two copies of the input with different target temperatures:

```bash
cd ~/lammps/model-setups/mace_mp0_new
cp mace_mp0.in mace_mp0_900K.in
cp mace_mp0.in mace_mp0_1500K.in
```

Edit the two lines in each file:

```
velocity      all create <TEMP> 12345
fix           myEnse all nvt temp <TEMP> <TEMP> 0.1
```

Launch both simultaneously (in separate shells) so they share the GPU:

```bash
# Terminal 1 – 900 K
OUTPUT_DIR=~/Ar_result/mace_new RUN_LABEL=900K ./run_mace_mp0.sh -in mace_mp0_900K.in

# Terminal 2 – 1500 K
OUTPUT_DIR=~/Ar_result/mace_new RUN_LABEL=1500K ./run_mace_mp0.sh -in mace_mp0_1500K.in
```

The RTX 3080 typically sits around 85 W for one job, so in theory, running two in parallel is fine (but in practice, my computer crash when running two in parallel). Just ensure `dump_root` (handled by the wrapper) and `RUN_LABEL` differ so outputs do not overwrite.

### Direct LAMMPS invocation

For custom workflows, call the wrapper executable directly (see `README.md` in the same folder) and point `-var log_file` and `-var dump_root` at writable directories, ideally on `/dev/shm` or `/tmp` to avoid stressing `/home`.

## 7. Suggested GitHub structure

When mirroring to GitHub, include:

1. `MACE_AR_FINETUNE.md` (this file) – top-level documentation.
2. `DATASET_PREP.md` – raw data lineage.
3. `run_ar_finetune.sh` – training driver.
4. `lammps/model-setups/mace_mp0_new/` – ready-to-run inference deck + TorchScript model (if licensing permits; otherwise document the export command).
5. Sample results (e.g., `Ar_result/20251121-112158/`) or plots derived from them.

Anyone cloning the repo can then follow the exact steps: convert data → activate `mace-env` → run the script epoch by epoch → export the model → launch the LAMMPS deck at any temperature.

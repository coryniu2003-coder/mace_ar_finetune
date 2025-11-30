# MACE MPA-0 (MP0) + LAMMPS Setup (fine-tuned for Ar)

Files:
- `mace_mp0.in`: 100,000-step NVT example using the fine-tuned Ar model exported to TorchScript (chunked into 50 × 2,000-step segments for dump rotation).
- `mace_mp0_test.in`: 200-step smoke test (single chunk) for validating the setup.
- `ar_finetune_epoch1.model`: fine-tuned checkpoint (epoch 1) generated under `~/fcc-p-10/mace_runs/ar_finetune`.
- `ar_finetune_epoch1.model-lammps.pt`: TorchScript file produced via `mace_create_lammps_model ar_finetune_epoch1.model --head Default --dtype float32` and referenced by `mace_mp0.in`.
- `mace-mp-0b3-medium.model` / `mace-mp-0b3-medium.model-lammps.pt`: the original MP-0b3 weights are kept here for reference.
- `dat.lammps`: FCC Ar structure (108 atoms).
- `archive/`: legacy outputs from earlier experiments.

## GPU wrapper (`run_mace_mp0.sh`)

Default launch (from this directory):

```bash
cd ~/lammps/model-setups/mace_mp0_new
./run_mace_mp0.sh
```

What the wrapper does now:

1. Activates the `mace-env` Conda environment and forces single-threaded BLAS/PyTorch inside `build-mace_env/lmp_mace_env.sh`.
2. Streams all dump chunks to `/dev/shm/...` during the run to keep GPU kernels fed.
3. Copies the finished `xyz_<segment>.lammpstrj` files into a timestamped results folder under `/tmp/mace_mp0_runs/<YYYYmmdd-HHMMSS>/dumps/` (override with `OUTPUT_DIR=/path/you/control`).
4. Concatenates those chunks into `dumps/xyz_full.lammpstrj` and stores both `log.mace_mp0` and the full stdout (`run_mace_mp0.out`) alongside them.

You will see the final path echoed at the end of the run, e.g.

```
MACE run complete. Outputs stored under /tmp/mace_mp0_runs/20251104-162530
```

Copy that directory back into your project if you need to archive results permanently.

### Useful overrides

All overrides are simple environment variables set before calling the script:

| Variable | Meaning | Default |
|----------|---------|---------|
| `TOTAL_STEPS` | Total MD steps | `100000` |
| `CHUNK_STEPS` | Steps per dump chunk | `2000` |
| `OUTPUT_DIR` | Parent directory that will receive the timestamped run folder | `/tmp/mace_mp0_runs` |
| `RUN_LABEL` | Override the timestamp-based folder name | current timestamp |
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0` |

Example (short GPU smoke on Si with 20 chunks of 5,000 steps written to `/scratch/mace_runs/Si_test`):

```bash
OUTPUT_DIR=/scratch/mace_runs \
RUN_LABEL=Si_test \
TOTAL_STEPS=100000 \
CHUNK_STEPS=5000 \
CUDA_VISIBLE_DEVICES=1 \
./run_mace_mp0.sh
```
The same chunking logic works for the original Ar deck and for the Si structure under `../model-setups_Si/mace_mp0`.

Required Python bits inside `mace-env`:
1. Install `mace` (already present) so that `/home/cory/anaconda3/envs/mace-env/lib/python3.12/site-packages/mace/cli/create_lammps_model.py` is available.
2. (When updates are needed) regenerate the TorchScript file via  
   `mace_create_lammps_model ar_finetune_epoch1.model --head Default --dtype float32`  
   which overwrites `ar_finetune_epoch1.model-lammps.pt`.

Logs end up inside the run directory described above:

- `log.mace_mp0` – the LAMMPS log file
- `run_mace_mp0.out` – stdout/stderr captured from the executable
- `dumps/xyz_<segment>.lammpstrj` – individual chunk dumps
- `dumps/xyz_full.lammpstrj` – concatenated trajectory for quick viewing

### Manual execution

If you want to call LAMMPS directly (e.g., for experimentation), you must supply all variables that the input deck expects and pick writable locations for logs/dumps:

```bash
mkdir -p /dev/shm/mace_mp0_manual /tmp/mace_mp0_manual
../../build-mace_env/lmp_mace_env.sh \
  -log none \
  -var log_file /tmp/mace_mp0_manual/log.mace_mp0 \
  -var dump_root /dev/shm/mace_mp0_manual \
  -var total_steps 100000 \
  -var chunk_steps 2000 \
  -in mace_mp0.in
mv /dev/shm/mace_mp0_manual/xyz_*.lammpstrj /tmp/mace_mp0_manual/
```
Adjust the numeric arguments as needed, and remember to keep both the log and dump destinations on a filesystem that can handle sustained writes (tmpfs or `/tmp` works best on this workstation).

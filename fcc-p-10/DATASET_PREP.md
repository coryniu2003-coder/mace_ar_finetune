# Ar Dataset Preparation

This folder now contains ASE/XYZ conversions of the professor's reference VASP runs and ready-made splits for fine-tuning ML potentials.

## Source trajectories

Each temperature subdirectory (`T-300`, `T-900`, `T-1500`, `T-2500`) carried a `vasprun.xml` with 5,000 MD frames (108-atom FCC Ar cell). They were converted into ASE `.traj` files via:

```bash
python vasp2ase.py T-300/vasprun.xml -o T-300.traj
python vasp2ase.py T-900/vasprun.xml -o T-900.traj
python vasp2ase.py T-1500/vasprun.xml -o T-1500.traj
python vasp2ase.py T-2500/vasprun.xml -o T-2500.traj
```

These trajectories preserve energies, forces, and stress tensors as parsed by ASE.

## Train/validation/test splits

Each 5,000-frame trajectory was stratified into 4,000/500/500 frames for train/val/test respectively while keeping metadata that downstream tools can consume:

- `config_type` → `Ar_<temp>K` (e.g., `Ar_300K`)
- `temperature` → scalar temperature tag
- `frame_source` → original directory (e.g., `T-1500`)

The splits were written as extended XYZ files using `ase.io.write`, resulting in:

| File            | Frames | Notes                                        |
|-----------------|--------|----------------------------------------------|
| `ar_train.xyz`  | 16,000 | 4k frames per temperature (sorted order)     |
| `ar_val.xyz`    | 2,000  | 500 frames per temperature                   |
| `ar_test.xyz`   | 2,000  | 500 frames per temperature                   |

All files live in `/home/cory/fcc-p-10/`. Each frame contains lattice vectors, per-atom positions/forces, cell stresses, and total energy (`energy` & `free_energy` keys).

## Next steps

1. Activate `mace-env`.
2. Point `mace_run_train` at `ar_train.xyz`/`ar_val.xyz` using the `run_ar_finetune.sh` helper.
3. After convergence, export the fine-tuned model and convert it to a TorchScript file for the LAMMPS deck.

See `MACE_AR_FINETUNE.md` for the full training workflow.

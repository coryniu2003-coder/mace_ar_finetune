# Published Ar Results (300 K, 10 GPa)

This folder provides curated outputs for the 300 K / 10 GPa solid Ar simulations
used to benchmark the MP-0b3 foundation model (pre-train) versus the fine-tuned
MACE model described in `fcc-p-10/MACE_AR_FINETUNE.md`.

## Layout

- `300k_pretrained/` – Original MP-0b3 outputs taken before fine-tuning
  (`~/Ar_result/mace/300k unwrap`).
- `300k_finetuned/` – Outputs produced after fine-tuning with the VASP dataset
  (`~/Ar_result/mace_new/300k`).

Each directory contains the same artifact set:

| File | Notes |
|------|-------|
| `log.mace_mp0` | LAMMPS log file (100k steps, chunked dumps). |
| `run_mace_mp0.out` | Stdout captured by the wrapper (fine-tuned run only). |
| `xyz_full.lammpstrj` | Full trajectory concatenation. |
| `dumps/xyz_*.lammpstrj` | Per-chunk dumps (fine-tuned run). |
| `lattice_vs_time.dat` | a-lattice vs MD step. |
| `msd-all.dat` / `.pdf` | Mean-square displacement analysis. |
| `pdf-X-X.dat` / `.pdf` | Radial distribution function. |
| `PROBCAR_Ar.vasp`, `PROBPRIM_Ar.vasp` | Probability density snapshots. |
| `ase_pp.py` | Helper script used for post-processing (fine-tuned run). |

## Why this directory exists

The raw `~/Ar_result/...` folders are intentionally ignored in git due to size.
This directory provides a git-friendly snapshot so the GitHub repo captures the
same validation evidence without depending on the local scratch area.

#!/usr/bin/env bash
# Launch the MACE trajectory with the GPU-enabled environment.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LAMMPS_BIN="${SCRIPT_DIR}/../../build-mace_env/lmp_mace_env.sh"
INPUT="${SCRIPT_DIR}/mace_mp0.in"
ENV_NAME="mace-env"

# Runtime controls
TOTAL_STEPS=${TOTAL_STEPS:-100000}
CHUNK_STEPS=${CHUNK_STEPS:-2000}
RUN_LABEL=${RUN_LABEL:-$(date +%Y%m%d-%H%M%S)}
OUTPUT_BASE=${OUTPUT_DIR:-/home/cory/Ar_result/mace_new}
RUN_DIR="${OUTPUT_BASE}/${RUN_LABEL}"
FINAL_DUMP_DIR="${RUN_DIR}/dumps"
LOG_FILE="${RUN_DIR}/log.mace_mp0"
STDOUT_FILE="${RUN_DIR}/run_mace_mp0.out"

mkdir -p "${FINAL_DUMP_DIR}"

# Clean up stale chunked dump files in the run directory if they exist.
rm -f "${FINAL_DUMP_DIR}"/xyz_*.lammpstrj "${FINAL_DUMP_DIR}/xyz_full.lammpstrj" 2>/dev/null || true

# Stage chunk dumps under tmpfs for the active run.
if TMP_OUTPUT_DIR=$(mktemp -d -p /dev/shm mace_mp0.XXXXXX 2>/dev/null); then
    :
else
    TMP_OUTPUT_DIR=$(mktemp -d)
fi

cleanup_tmp() {
    rm -rf "${TMP_OUTPUT_DIR}"
}
trap cleanup_tmp EXIT

# Allow the caller to override GPU selection, default to the first device.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Build the LAMMPS command.
CMD=(
    "${LAMMPS_BIN}"
    -log none
    -var log_file "${LOG_FILE}"
    -var dump_root "${TMP_OUTPUT_DIR}"
    -var total_steps "${TOTAL_STEPS}"
    -var chunk_steps "${CHUNK_STEPS}"
    -in "${INPUT}"
)

# Activate the environment if needed.
if [ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda command not found; cannot activate ${ENV_NAME}" >&2
        exit 1
    fi
    CONDA_BASE=$(conda info --base)
    # shellcheck disable=SC1091
    set +u
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
    set -u
fi

# Run LAMMPS and capture stdout/stderr.
set +e
"${CMD[@]}" 2>&1 | tee "${STDOUT_FILE}"
CMD_STATUS=${PIPESTATUS[0]}
set -e
if [ ${CMD_STATUS} -ne 0 ]; then
    exit ${CMD_STATUS}
fi

# Move chunk files into the persistent run directory and aggregate.
if compgen -G "${TMP_OUTPUT_DIR}/xyz_*.lammpstrj" > /dev/null; then
    mv "${TMP_OUTPUT_DIR}"/xyz_*.lammpstrj "${FINAL_DUMP_DIR}/"
    if compgen -G "${FINAL_DUMP_DIR}/xyz_*.lammpstrj" > /dev/null; then
        python3 - <<'PY' "${FINAL_DUMP_DIR}"
import glob, os, re, sys
root = sys.argv[1]
pattern = os.path.join(root, 'xyz_*.lammpstrj')
files = glob.glob(pattern)
if not files:
    raise SystemExit(0)
def sort_key(path):
    name = os.path.basename(path)
    match = re.search(r'_(\d+)\.lammpstrj$', name)
    return int(match.group(1)) if match else name
files.sort(key=sort_key)
out_path = os.path.join(root, 'xyz_full.lammpstrj')
with open(out_path, 'wb') as out_f:
    for chunk in files:
        with open(chunk, 'rb') as src:
            out_f.write(src.read())
PY
    fi
fi

echo "MACE run complete. Outputs stored under ${RUN_DIR}"

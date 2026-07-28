#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

: "${CODA_REMOTE:?Set CODA_REMOTE to your SSH host, for example user@gpu-server or an SSH alias.}"

REMOTE_DIR="${CODA_REMOTE_DIR:-~/CoDA}"
REMOTE_SCRIPT="${CODA_REMOTE_SCRIPT:-scripts/CoDA.sh}"
CONDA_ENV="${CODA_CONDA_ENV:-coda}"
CUDA_DEVICES="${CODA_CUDA_DEVICES:-0}"
SYNC_FIRST="${CODA_SYNC_FIRST:-1}"

if [[ "$SYNC_FIRST" == "1" ]]; then
    "$SCRIPT_DIR/remote_sync.sh"
fi

remote_command=$(cat <<EOF
set -euo pipefail
cd "$REMOTE_DIR"
if command -v conda >/dev/null 2>&1; then
    eval "\$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif [[ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "\$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
elif [[ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "\$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    echo "Cannot find conda on the remote host. Set CODA_CONDA_ENV after installing the environment." >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export IPC="${IPC:-10}"
export N_NEIGHBORS="${N_NEIGHBORS:-85}"
export SIZE_MIN="${SIZE_MIN:-55}"
export STEP1="${STEP1:-true}"
export FEATURES="${FEATURES:-true}"
export CLUSTER="${CLUSTER:-true}"
export GENERATE="${GENERATE:-true}"
export STEP2="${STEP2:-true}"
export REAL_IMAGES="${REAL_IMAGES:-true}"
bash "$REMOTE_SCRIPT"
EOF
)

ssh "$CODA_REMOTE" "bash -lc $(printf '%q' "$remote_command")"

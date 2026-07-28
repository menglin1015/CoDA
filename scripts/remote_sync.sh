#!/usr/bin/env bash
set -euo pipefail

: "${CODA_REMOTE:?Set CODA_REMOTE to your SSH host, for example user@gpu-server or an SSH alias.}"

REMOTE_DIR="${CODA_REMOTE_DIR:-~/CoDA}"

ssh "$CODA_REMOTE" "REMOTE_DIR=$(printf '%q' "$REMOTE_DIR"); mkdir -p \"\$REMOTE_DIR\""

rsync_args=(
    -az
    --human-readable
    --info=progress2
    --exclude ".git/"
    --exclude ".claude/"
    --exclude ".codex/"
    --exclude "__pycache__/"
    --exclude ".pytest_cache/"
    --exclude "results/"
    --exclude "trained_results/"
    --exclude "_in1k_run/"
    --exclude "*.pyc"
    --exclude "*.log"
    --exclude "CoDA_full.tar.zst"
)

if [[ "${CODA_RSYNC_DELETE:-0}" == "1" ]]; then
    rsync_args+=(--delete)
fi

rsync "${rsync_args[@]}" ./ "$CODA_REMOTE:$REMOTE_DIR/"

echo "Synced $(pwd) to $CODA_REMOTE:$REMOTE_DIR"

#!/usr/bin/env bash
# Source this file before running scripts manually:
#   source setup_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

if [ -f "$REPO_DIR/configs/paths.local.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_DIR/configs/paths.local.env"
  set +a
fi

export PAIAD_SCORE_ROOT="${PAIAD_SCORE_ROOT:-$REPO_DIR/outputs/score}"
export PAIAD_AGG_ROOT="${PAIAD_AGG_ROOT:-$REPO_DIR/outputs/aggregates/eva350}"
export PAIAD_OUTPUT_ROOT="${PAIAD_OUTPUT_ROOT:-$REPO_DIR/outputs}"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/third_party/TSB-AD:$REPO_DIR/third_party/ts2vec:$REPO_DIR/third_party/KDD2023-DCdetector:${PYTHONPATH:-}"

export PAIAD_THREAD_CAPS="OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 BLIS_NUM_THREADS=2"
export PAIAD_STRICT_THREAD_CAPS="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 TBB_NUM_THREADS=1"

#!/usr/bin/env bash
# Reproduce the paper tables from model prediction anomaly scores.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_DIR/configs/paths.local.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_DIR/configs/paths.local.env"
  set +a
elif [ -f "$SCRIPT_DIR/../configs/paths.local.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/../configs/paths.local.env"
  set +a
fi

PYBIN="python"
NUM_PROCS="${PAIAD_NUM_PROCS:-16}"
SCORE_ROOT="${PAIAD_SCORE_ROOT:-$REPO_DIR/outputs/score}"
AGG_ROOT="${PAIAD_AGG_ROOT:-$REPO_DIR/outputs/aggregates/eva350}"
OUTPUT_ROOT="${PAIAD_OUTPUT_ROOT:-$REPO_DIR/outputs}"
ABLATION_ROOT="${PAIAD_ABLATION_ROOT:-$(dirname "$AGG_ROOT")/weight_ablation}"
LOG_ROOT="${PAIAD_LOG_ROOT:-$OUTPUT_ROOT/logs}"
export PAIAD_SCORE_ROOT="$SCORE_ROOT"
export PAIAD_AGG_ROOT="$AGG_ROOT"
export PAIAD_OUTPUT_ROOT="$OUTPUT_ROOT"
if [ -z "${PAIAD_TSB_U_DATASET_DIR:-}" ]; then
  for p in \
    "$REPO_DIR/data/TSB-AD-U/TSB-AD-U" \
    "$REPO_DIR/data/TSB-AD/Datasets/TSB-AD-U/TSB-AD-U" \
    "$REPO_DIR/data/TSB-AD/TSB-AD-U/TSB-AD-U"
  do
    if [ -d "$p" ]; then
      export PAIAD_TSB_U_DATASET_DIR="$p"
      break
    fi
  done
  export PAIAD_TSB_U_DATASET_DIR="${PAIAD_TSB_U_DATASET_DIR:-$REPO_DIR/data/TSB-AD-U/TSB-AD-U}"
fi
if [ -z "${PAIAD_TSB_U_EVA_CSV:-}" ]; then
  for p in \
    "$REPO_DIR/data/TSB-AD/Datasets/File_List/TSB-AD-U-Eva.csv" \
    "$REPO_DIR/data/TSB-AD/File_List/TSB-AD-U-Eva.csv" \
    "$REPO_DIR/data/File_List/TSB-AD-U-Eva.csv" \
    "$REPO_DIR/data/TSB-AD-U-Eva.csv"
  do
    if [ -f "$p" ]; then
      export PAIAD_TSB_U_EVA_CSV="$p"
      break
    fi
  done
  export PAIAD_TSB_U_EVA_CSV="${PAIAD_TSB_U_EVA_CSV:-$REPO_DIR/data/TSB-AD/Datasets/File_List/TSB-AD-U-Eva.csv}"
fi
export PYTHONPATH="$REPO_DIR/code:$REPO_DIR/third_party/TSB-AD:$REPO_DIR/third_party/ts2vec:$REPO_DIR/third_party/KDD2023-DCdetector:${PYTHONPATH:-}"

usage() {
  cat <<EOF
Usage: ./reproduce.sh <target>

Targets:
  validate_data           optional sanity check for raw TSB-AD-U Eva inputs
  generate_anomaly_scores run all models and write anomaly-score files for metrics
  main_table              regenerate full_comparison_table.csv (and FULL_COMPARISON_TABLE.md)
  ablation_table          regenerate fusion-weight ablation table

Put the TSB-AD dataset under data/. Then run generate_anomaly_scores to produce
model prediction anomaly-score files, and run main_table or ablation_table to
calculate metrics.
EOF
}

if [ $# -ne 1 ]; then
  usage
  exit 1
fi

cd "$SCRIPT_DIR"
mkdir -p "$SCORE_ROOT" "$AGG_ROOT" "$OUTPUT_ROOT" "$ABLATION_ROOT" "$LOG_ROOT"

case "$1" in
  validate_data)
    MANIFEST_ARGS=()
    if [ -n "${PAIAD_TSB_U_MANIFEST_CSV:-}" ]; then
      MANIFEST_ARGS=(--out_csv "$PAIAD_TSB_U_MANIFEST_CSV")
    fi
    "$PYBIN" data/validate_tsbad.py \
      --dataset_dir "$PAIAD_TSB_U_DATASET_DIR" \
      --file_list "$PAIAD_TSB_U_EVA_CSV" \
      "${MANIFEST_ARGS[@]}"
    ;;
  generate_anomaly_scores)
    echo "[generate_anomaly_scores] writing anomaly-score files to $SCORE_ROOT"
    echo "[generate_anomaly_scores] writing logs to $LOG_ROOT"

    "$PYBIN" runners/run_ts2vec_eva.py \
      --score_dir "$SCORE_ROOT/TS2Vec" \
      --log_path "$LOG_ROOT/ts2vec_original.log"

    "$PYBIN" runners/run_ts2vec_uniform_score.py \
      --score_dir "$SCORE_ROOT/UNIFORM_TS2Vec" \
      --log_path "$LOG_ROOT/ts2vec_pai.log"

    "$PYBIN" runners/run_dcdetector_eva.py \
      --score_dir "$SCORE_ROOT/DCdetector" \
      --log_path "$LOG_ROOT/dcdetector_original.log"

    "$PYBIN" runners/run_dcdetector_uniform_score.py \
      --score_dir "$SCORE_ROOT/UNIFORM_DCdetector" \
      --log_path "$LOG_ROOT/dcdetector_pai.log"

    "$PYBIN" runners/run_classical_baseline_eva350.py \
      --score_dir "$SCORE_ROOT/TSPulse_ZS" \
      --AD_Name TSPulse_ZS

    "$PYBIN" runners/run_tspulse_uniform_score.py \
      --score_dir "$SCORE_ROOT/UNIFORM_TSPulse" \
      --log_path "$LOG_ROOT/tspulse_pai.log"

    "$PYBIN" runners/run_tspulse_ft_pmlookup.py \
      --score_dir "$SCORE_ROOT" \
      --AD_Name TSPulse_FT

    "$PYBIN" runners/run_paano_tsb.py \
      --score_dir "$SCORE_ROOT/PaAno_baseline" \
      --variant original

    "$PYBIN" runners/run_paano_tsb.py \
      --score_dir "$SCORE_ROOT/PaAno_PAI" \
      --variant pai

    echo "[generate_anomaly_scores] done"
    ;;
  main_table)
    env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        NUMEXPR_NUM_THREADS=2 BLIS_NUM_THREADS=2 \
      "$PYBIN" aggregators/build_full_table.py \
        --num_procs "$NUM_PROCS" \
        --out_csv "$AGG_ROOT/pool_means_main_table.csv"
    "$PYBIN" aggregators/finalize_table.py \
      --csv     "$AGG_ROOT/pool_means_main_table.csv" \
      --out_md  "$OUTPUT_ROOT/FULL_COMPARISON_TABLE.md"
    ;;
  ablation_table)
    env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        NUMEXPR_NUM_THREADS=2 BLIS_NUM_THREADS=2 \
      "$PYBIN" aggregators/build_weight_ablation.py \
        --num_procs "$NUM_PROCS" \
        --out_dir "$ABLATION_ROOT" \
        --out_csv "$ABLATION_ROOT/sweep_eva350_all.csv" \
        --out_md  "$OUTPUT_ROOT/WEIGHT_ABLATION_TABLE.md"
    ;;
  *)
    usage
    exit 1
    ;;
esac

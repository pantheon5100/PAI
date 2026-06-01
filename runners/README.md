# Runners

Runners generate model prediction anomaly-score files for each TSB-AD-U Eva
series. These files are later used for metric calculation. By default, all
scripts read the dataset from the local release layout:

```text
data/TSB-AD-U/TSB-AD-U/*.csv
data/TSB-AD/Datasets/File_List/TSB-AD-U-Eva.csv
```

Use `--file_list` and `--dataset_dir` only if your local layout is different.

Use `--start` and `--end` to run a one-file smoke test or shard the 350 files.
To run every model sequentially and write all anomaly-score files, use:

```bash
cd code
./reproduce.sh generate_anomaly_scores
```

## Method Map

| Method | Original runner | PAI runner/export |
|---|---|---|
| TS2Vec | `run_ts2vec_eva.py` -> `TS2Vec/<fid>.npy` | `run_ts2vec_uniform_score.py` -> `UNIFORM_TS2Vec/<fid>_{eu,magG,T2,train_stats}` |
| DCdetector | `run_dcdetector_eva.py` -> `DCdetector/<fid>.npy` | `run_dcdetector_uniform_score.py` -> `UNIFORM_DCdetector/<fid>_{eu,magG,T2,train_stats}` |
| TSPulse | `run_classical_baseline_eva350.py --AD_Name TSPulse_ZS` -> `TSPulse_ZS/<fid>.npy` | `run_tspulse_uniform_score.py` -> `UNIFORM_TSPulse/<fid>_{eu,magG,T2,train_stats}` |
| PaAno | `run_paano_tsb.py --variant original` -> `PaAno_baseline/<fid>/cos_score.npy` | `run_paano_tsb.py --variant pai` -> `PaAno_PAI/<fid>/eucl_score.npy` |

`run_tspulse_ft_pmlookup.py` is kept as a paper-protocol TSPulse_FT baseline
runner. It is not needed for the original-vs-PAI four-method table unless you
choose to report that extra baseline.

## Smoke Commands

```bash
cd code
source setup_env.sh

# TS2Vec
python runners/run_ts2vec_eva.py \
  --score_dir "$PAIAD_SCORE_ROOT/TS2Vec" --log_path /tmp/ts2vec_native.log \
  --start 0 --end 1 --n_iters 1

python runners/run_ts2vec_uniform_score.py \
  --score_dir "$PAIAD_SCORE_ROOT/UNIFORM_TS2Vec" --log_path /tmp/ts2vec_pai.log \
  --start 0 --end 1 --n_iters 1

# DCdetector
python runners/run_dcdetector_eva.py \
  --score_dir "$PAIAD_SCORE_ROOT/DCdetector" --log_path /tmp/dcdet_native.log \
  --start 0 --end 1 --n_epochs 1

python runners/run_dcdetector_uniform_score.py \
  --score_dir "$PAIAD_SCORE_ROOT/UNIFORM_DCdetector" --log_path /tmp/dcdet_pai.log \
  --start 0 --end 1 --n_epochs 1

# TSPulse
python runners/run_classical_baseline_eva350.py \
  --score_dir "$PAIAD_SCORE_ROOT/TSPulse_ZS" --AD_Name TSPulse_ZS \
  --start 0 --end 1

python runners/run_tspulse_uniform_score.py \
  --score_dir "$PAIAD_SCORE_ROOT/UNIFORM_TSPulse" --log_path /tmp/tspulse_pai.log \
  --start 0 --end 1

# PaAno
python runners/run_paano_tsb.py \
  --score_dir "$PAIAD_SCORE_ROOT/PaAno_baseline" --variant original \
  --start 0 --end 1 --num_iters 1 --device cpu

python runners/run_paano_tsb.py \
  --score_dir "$PAIAD_SCORE_ROOT/PaAno_PAI" --variant pai \
  --start 0 --end 1 --num_iters 1 --device cpu
```

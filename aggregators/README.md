# Aggregation

The release workflow has two aggregation paths:

```bash
cd code
./reproduce.sh main_table
./reproduce.sh ablation_table
```

`build_full_table.py` reads model prediction anomaly scores for TS2Vec,
DCdetector, TSPulse, and PaAno original/PAI variants, evaluates them with
TSB-AD metrics, and writes:

```text
$PAIAD_AGG_ROOT/pool_means_main_table.csv
```

`finalize_table.py` renders that CSV as:

```text
$PAIAD_OUTPUT_ROOT/FULL_COMPARISON_TABLE.md
```

`build_weight_ablation.py` reads anomaly-score components for TS2Vec,
DCdetector, and TSPulse_ZS, sweeps fixed fusion weights over encoder/native,
magG, and T2 components, and writes:

```text
$PAIAD_ABLATION_ROOT/sweep_eva350_all.csv
$PAIAD_OUTPUT_ROOT/WEIGHT_ABLATION_TABLE.md
```

Each score can be either full-series or test-only. The aggregator detects the
length and aligns labels accordingly.

# Dataset Validation

This directory contains optional dataset validation scripts for the release
workflow. The default dataset location is:

```text
data/TSB-AD-U/TSB-AD-U/*.csv
data/TSB-AD/Datasets/File_List/TSB-AD-U-Eva.csv
```

## TSB-AD-U Eva

```bash
cd code
./reproduce.sh validate_data
```

The command validates the local dataset and file list. If `--out_csv` is passed,
it also writes a per-file manifest with:

- row counts before and after `dropna`
- feature count
- train/test split length parsed from the file name
- anomaly point count

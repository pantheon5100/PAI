# Vendored Baseline Implementations

This directory contains the source subsets required for end-to-end reproduction.
Large datasets, generated scores, caches, images, and `.git` metadata are not included.

- `TSB-AD/`: TSB-AD metric implementation, classical baseline wrappers, and TSPulse wrapper utilities.
- `ts2vec/`: TS2Vec model, losses, utilities, and task code required by the TS2Vec runners.
- `KDD2023-DCdetector/`: DCdetector model modules required by the DCdetector runners.
- `PaAno/`: PaAno baseline core and scoring utilities used by PaAno reproduction and comparison.

The runner scripts prefer these vendored paths by default. Set `PAIAD_TSB_AD_REPO`,
`PAIAD_TS2VEC_REPO`, `PAIAD_DCDETECTOR_REPO`, or `PAIAD_PAANO_REPO` to override
with external checkouts.

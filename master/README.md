# Master PCA Structure

## Files
- `pca_data_prep.py`: Build/normalize probe tables and event metadata, then save/load processed bundles.
- `pca_plotting.py`: All PCA plotting functions.
- `PCA_master_runner.ipynb`: Main plotting notebook that loads processed data and runs plot cells.
- `processed_data/bundle_latest/`: Default processed bundle location used by the notebook.

## Typical Flow
1. Use `pca_data_prep.py` to build and save a processed bundle.
2. Open `PCA_master_runner.ipynb`.
3. Run the load + verify cells.
4. Run the plotting cells you want.

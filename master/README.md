# Master PCA Structure

## Files
- `pca_data_prep.py`: Build/normalize probe tables and event metadata, then save/load processed bundles.
- `pca_plotting.py`: All PCA plotting functions.
- `PCA_master_runner.ipynb`: Main plotting notebook that loads processed data and runs plot cells.
- `processed_data/bundle_latest/`: Default processed bundle location used by the notebook.

## Typical Flow
1. Use `pca_data_prep.py` to build and save a processed bundle.
2. Open `PCA_master_runner.ipynb`.
3. In the load cell, either:
   - load an existing `processed_data/bundle_latest`, or
   - set `NWB_PATH_FOR_AUTO_BUILD` and (optionally) `BOMBCELL_ROOT_FOR_AUTO_BUILD` to auto-build/rebuild with Bombcell columns.
4. Run single-probe grouped cells (each group is one larger figure with related subplots).
5. Run the grouped batch cell to save all grouped figure sets for all probes/brain regions.

## Grouped Plot Sets Included
- Group 1 (base trial-level): 2D scatter + 3D scatter + PC1-PC2-Time.
- Group 2 (epoch-mean family): 2D + 3D + PC1-PC2-Time points + PC1-PC2-Time lines.
- Group 3 (per-trial time-bin family): 2D + 3D + PC1-PC2-Time.
- Group 4 (one-line-per-epoch population family): 2D + 3D + PC1-PC2-Time.

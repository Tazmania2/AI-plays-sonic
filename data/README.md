# Data Directory

This directory is for storing data files used by the AI Sonic project.

## Structure

- `raw/` - Raw data files (ignored by git)
- `processed/` - Processed data files (ignored by git)
- `external/` - External datasets (ignored by git)

## Notes

- Large data files (*.csv, *.json, *.parquet, *.h5, *.npz, *.npy) are ignored by git
- Place your training data, game states, and other datasets here
- Use relative paths in your code to reference data files 
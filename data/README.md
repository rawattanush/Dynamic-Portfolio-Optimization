# Data

The dataset used in this project is not stored directly in the repository because of size limitations.

Data is downloaded using `yfinance` in `extractor.ipynb` and saved to Google Drive.

Final processed dataset:

NIFTY50_CLEAN_LONG_FORMAT.csv

Google Drive location used in the notebooks:

/Nifty50_RL_Project/finance_data/processed/

The dataset contains:
- Historical OHLCV data for NIFTY50 constituents
- Technical indicators generated using `stockstats` and `ta`
- Data formatted in long format for FinRL environment compatibility

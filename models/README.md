# Models

Trained reinforcement learning models are not stored in this repository.

Models are generated during notebook training and saved to Google Drive.

Saved models:

- V1.zip → PPO agent with time_window = 1
- V2.zip → PPO agent with time_window = 10 (best performing model)
- V3.zip → PPO agent with additional features and drawdown penalty

Google Drive location:

/Nifty50_RL_Project/models/

These models are loaded during the final backtesting step in `trainercopy (V3 - BAD).ipynb`.

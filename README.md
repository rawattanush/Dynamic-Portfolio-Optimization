
# Deep Reinforcement Learning for Adaptive Portfolio Optimization

**Status:** `Complete`

> A project to develop and evaluate a Deep RL agent (PPO) for dynamic asset allocation in the NIFTY50 stock market. The final V2 model demonstrates a robust, defensive policy that outperforms market benchmarks on a risk-adjusted basis, especially during bear markets.

---

## 1. Problem Statement

Traditional portfolio management strategies (like static "buy-and-hold" or market-cap weighting) are often sub-optimal. They fail to adapt to changing market regimes, leaving portfolios exposed to high volatility and significant drawdowns.

This project investigates whether a Reinforcement Learning agent can learn a dynamic trading policy by directly interacting with market data. The agent's goal is to **maximize the risk-adjusted return (Sharpe Ratio)** by learning when to aggressively seek returns and, more importantly, when to defensively rotate into a "risk-free" cash position to preserve capital.

## 2. Technology Stack

* **Core Model:** `Python 3.x`, `PyTorch`
* **RL Framework:** `Stable-Baselines3` (PPO Algorithm)
* **Environment:** `Gymnasium` (Custom FinRL Environment)
* **Data & Analysis:** `Pandas`, `NumPy`, `yfinance` (Data Ingestion)
* **Feature Engineering:** `stockstats`, `ta`
* **Environment:** `Google Colab` & `Google Drive`

---

## 3. Key Findings & Analysis: The V2 Model

The final analysis revealed a critical and nuanced insight: **V2 (10-Day Window) is the optimal model, but its strength is highly specific.**

The model's performance is defined by its 10-day "memory," which creates a trade-off:

* **It Fails in "Black Swan" Events:** V2 performed the *worst* of all models during the **2020 Crash** (Sharpe -1.470). The crash was too sudden and violent; by the time the 10-day window confirmed a problem, the portfolio had already suffered massive losses.

* **It Masters "Slow-Grind" Downturns:** V2 was the *undisputed winner* of the **2022 Bear Market** (Sharpe +0.734). This slow, protracted downturn was the perfect environment for its 10-day memory to identify the bearish regime, adapt, and defensively move to cash, preserving capital while all other models failed.

### Comparative Performance (Sharpe Ratio by Market Regime)

| Model / Regime | 2020 Crash<br/>(Fast) | 2020-21 Bull Run<br/>(Up) | **2022 Bear Market**<br/>(Slow) | 2023-25 Recovery<br/>(Up) |
| :--- | :---: | :---: | :---: | :---: |
| **🏆 RL Model V2 (10-Day)** | -1.470 | 2.769 | **+0.734** | 1.357 |
| **Equal-Weight (Benchmark)** | -1.114 | **2.923** | 0.592 | **1.534** |
| **NIFTY50 (Benchmark)** | -1.203 | 2.094 | 0.335 | 1.103 |
| **RL Model V3 (Overfit)** | -1.506 | 2.480 | 0.049 | 0.907 |
| **RL Model V1 (Memoryless)**| -1.378 | 2.246 | -0.123 | 0.643 |

### Core Insight: A Tool for a Specific Job

The V2 model is not a "silver bullet." It is a specialized tool that learned to identify and defend against **protracted, slow-moving bear markets**, making it an excellent risk-management agent. It was not, however, equipped to handle sudden, "black swan" crashes, demonstrating the critical importance of `time_window` selection in model design.
---

## 4. Model Architecture & Iterations

The project's success was driven by iterating on the agent's "state" (what it can see).

* **V1: The "Memoryless" Agent**
    * **Architecture:** `time_window = 1`. The agent only saw the current day's data.
    * **Result:** **Failure.** It was purely reactive, had no context, and could not distinguish a one-day dip from a market-wide crash.

* **V2: The "Informed" Agent (Optimal Model)**
    * **Architecture:** `time_window = 10`. The agent could see a 10-day sequence of market data.
    * **Result:** **Success.** This 10-day "memory" was the critical feature. It allowed the PPO agent to learn complex patterns (e.g., "volatility rising for 3 days while RSI stays low") that signaled a regime change, allowing it to move to cash.

* **V3: The "Over-Complicated" Agent**
    * **Architecture:** `time_window = 10` + *new features* (VIX, NIFTY indicators) + a *custom drawdown penalty* in the reward function.
    * **Result:** **Failure.** The new features likely added noise, and the custom penalty made the agent *too* defensive ("overfit" to risk), causing it to hide in cash and miss all returns.

---

## 5. Project Workflow (How to Run)

The notebooks **must be run in the following order**, as they are codependent.

1.  **`extractor.ipynb`**
    * **What it does:** Connects to Google Drive, downloads all NIFTY50 ticker data from Yahoo Finance, cleans it, formats it into a "long" CSV, and saves it to the Drive.
    * **Output:** `/Nifty50_RL_Project/finance_data/processed/_NIFTY50_CLEAN_LONG_FORMAT.csv`

2.  **`trainercopy (V1).ipynb`**
    * **What it does:** Loads the clean CSV from Step 1. Trains and evaluates the V1 model (`time_window=1`).
    * **Output:** Saves the trained model (`V1.zip`) to `/Nifty50_RL_Project/models/`

3.  **`trainercopy (v2).ipynb`**
    * **What it does:** Loads the clean CSV from Step 1. Trains and evaluates the V2 model (`time_window=10`).
    * **Output:** Saves the trained model (`V2.zip`) to `/NIFTY50_RL_Project/models/`

4.  **`trainercopy (V3 - BAD).ipynb`**
    * **What it does:** This is the final analysis notebook. It loads the clean CSV, trains the V3 model, and *also* loads the `V1.zip` and `V2.zip` models from the previous steps to run the final comparative backtest.
    * **Output:** The final comparison tables and graphs.

---

## 6. Setup & Environment

This project is configured for Google Colab.

### 1. Google Drive Folder Structure

Before running, you must create the following folder structure in the root of your Google Drive:

```

/MyDrive/
└── Nifty50\_RL\_Project/
├── finance\_data/
│   └── processed/
├── logs/
└── models/

```

### 2. Requirements

All notebooks include installation cells. The main libraries can also be installed locally via a `requirements.txt` file:

```

# Core ML

torch
stable-baselines3[extra]
gymnasium

# Data & Finance

pandas
numpy
yfinance
stockstats
ta
FinRL



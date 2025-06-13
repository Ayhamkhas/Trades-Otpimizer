# 💸 Crypto Strategy Optimizer 🤖📈  
_Evolving better trades — one generation at a time._

---

## 🧠 Overview

This project is a **multi-objective trading strategy optimizer** built in **C++**, designed to evolve and evaluate moving average crossover strategies on historical crypto market data.

Leveraging the power of **evolutionary algorithms** like **NSGA-II** and **SPEA2**, the optimizer searches for trading strategies that strike the best balance between **profitability**, **stability**, and **win rate** — because in trading, one metric never tells the whole story.

---

## ⚙️ What It Does

- 📥 Loads real **OHLCV** data from CSV files
- 📊 Represents each trading strategy as a combination of **short/long MA periods**
- 🧬 Evolves populations using:
  - **NSGA-II** (Non-dominated Sorting Genetic Algorithm II)
  - **SPEA2** (Strength Pareto Evolutionary Algorithm 2)
- 🎯 Evaluates individuals across three conflicting objectives:
  - ✅ **ROI** (Total Return)
  - ✅ **Win Rate**
  - ✅ **Risk** (drawdown, volatility, or other)
- 📈 Outputs **Pareto fronts** and logs hypervolume indicators
- 🧪 Performs **statistical analysis** comparing the results of both algorithms (Mann-Whitney U, Wilcoxon, etc.)

---

## 🔬 Why This Matters

Traditional trading bots often optimize for a single goal: profit.  
But in reality, traders care about **multiple things at once** — not just how much you make, but how consistently and how safely.

This project tries to **evolve smarter, more balanced strategies** by treating trading like a multi-objective optimization problem.
![Comparison_ParetoFront](https://github.com/user-attachments/assets/a663cbc4-0239-466b-90ae-7395992dc3a0)

---

## 🛠 How to Use

### 🔧 Build
```bash
git clone https://github.com/yourusername/crypto-strategy-optimizer.git
cd crypto-strategy-optimizer
mkdir build && cd build
cmake ..
make

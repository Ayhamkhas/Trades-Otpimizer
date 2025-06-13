# 💸 Crypto Strategy Optimizer 🤖📈  
_Evolving better trades — one generation at a time._

---

## 🧠 Overview

This project is a **multi-objective trading strategy optimizer** built in **C++**, designed to evolve and evaluate moving average crossover strategies on Ethereum market data.

Leveraging the power of **evolutionary algorithms** like **NSGA-II** and **SPEA2**, the optimizer searches for trading strategies that strike the best balance between **profitability**, **stability**, and **win rate** — because in trading, one metric never tells the whole story.

---

## ⚙️ What It Does

- 📥 Loads real **OHLCV** data of Ethereum from CSV files
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


## 📷 Results Overview

### 📌 Hypervolume Comparison

> NSGA-II significantly outperforms SPEA2 in hypervolume, suggesting better convergence and diversity.

![Hypervolume Comparison](https://github.com/user-attachments/assets/04996a24-962e-4f80-80b0-12c469406da6)

---

### 📌 Convergence over Generations

> NSGA-II converges more smoothly over 300 generations, while SPEA2 fluctuates and plateaus earlier.

![Hypervolume Convergence](https://github.com/user-attachments/assets/ebfc032e-e380-4e9b-9daa-b961225d6488)


---

### 📌 Final Pareto Fronts (3D View)

> Blue: NSGA-II | Red: SPEA2  
> NSGA-II produced a broader front with stronger trade-offs across ROI, win rate, and risk.

![Pareto Front Comparison](https://github.com/user-attachments/assets/b78a9a4e-fb20-458f-8247-34696b21ec7c)


---

### 📌 Win Rate Distribution

> Both algorithms reached similar win rate medians, but NSGA-II showed tighter bounds and fewer outliers.

![WinRate Boxplot](https://github.com/user-attachments/assets/b5a31905-4aa0-4b8e-aed9-373ffb44bf57)


---

## 🛠 How to Use

### 🔧 Build
```bash
git clone https://github.com/yourusername/crypto-strategy-optimizer.git
cd crypto-strategy-optimizer
mkdir build && cd build
cmake ..
make

# Prediction Market Trading Algorithm

## The Challenge: State Space Explosion
A direct Dynamic Programming formulation tracking absolute wealth `W_t` and contract counts `x_t` leads to an **unbounded state space**:
- Wealth `W_t` can grow without bound
- Contract counts `x_t` can be arbitrarily large
- Result: DP table size `O(T × |W| × |x|)` with `|W|, |x| → ∞` → **computationally infeasible**

## Key Innovation: Proportional Allocation

We show that for log utility with multiplicative returns, optimal trading depends only on the **proportion** of wealth allocated to contracts, not on absolute wealth levels. Defining:

`θ_t = (x_t × c_t) / (W_t + x_t × c_t) ∈ [-1, 1]`

where:
- `θ > 0`: Net long YES contracts
- `θ < 0`: Net long NO contracts  
- `|θ|`: Fraction of portfolio value in contracts

reduces the state space from unbounded `(W, x, c)` to bounded `(θ, c)` while yielding the same optimal policy as the unbounded formulation. This is true **even in the case of asymmetric spreads for YES and NO contracts**.

## Solution: Bounded Dynamic Programming
- **Policy Matrix**: `(T, θ, c)` where θ ∈ [-1,1]
- **Space Complexity**: $O(T*N^2)$ vs previous $O(T * \infty)$
- **Time Complexity**: $O(T * N^3)$
- **Realism**: Includes asymmetric spreads

## Evaluation & Results
We evaluate on 997 real [Polymarket price paths](https://www.kaggle.com/datasets/sandeepkumarfromin/full-market-data-from-polymarket) with 5% buy / 10% sell spreads (for YES and NO contracts, though they can be different in practice)

### Strategy Robustness:
- **Uniform regime**: Survives ±0.4 probability errors
- **Random walk**: Tolerates ±0.2 errors with regime fit  
- **Mean reverting**: Requires accurate probabilities AND regime

### Key Finding:
Optimal allocation & performance depends on:
- Outcome probability estimates
- Regime identification
- Buy Spread / Sell Spread

## Implications
- **Traders**: Optimal position sizing with real costs
- **Platforms**: Better AMM design using bounded inventory θ
- **Research**: Tractable DP for prediction market equilibrium, and enables future work on trader-market maker interactions

## Repository Structure
- `trading_dp.ipynb` - DP implementation with three regimes
- `evaluation.ipynb` - Sensitivity analysis on Polymarket data
- `trading_dp.py` - Source code for DP functions
- `docs/FORMULATION.md` - DP derivation & runtime analysis
- `docs/EVALUATION.md` - Evaluation methods overview
- `data/` - Polymarket price paths

## Quick Start
```python
from trading_dp import run_dp, make_transition_matrix
T = 31
n_prices = 21
psub = .7
regime = make_transition_matrix(psub,.2,mean_reversion=0.8,n_prices=n_prices)
V, policy = run_dp(model_trans=regime,
                    p_subj=psub,
                    T = T,
                    gamma_b = .05,
                    gamma_s = .10,
                    n_theta = 51,
                    n_b = 51,
                    n_prices = n_prices
                )
```
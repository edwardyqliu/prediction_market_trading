# Optimal Trading for Prediction Markets (Binary Options)

**Problem**: How should you size positions when market noise varies?

**Finding**: DP reveals fundamentally different optimal policies per regime:

1. **High noise** (uniform regime): Conservative allocation strategy - trade only extreme discounts  
2. **Trending** (random walk regime): Moderate allocation strategy; momentum optionality - allocate past fair value
3. **Predictable** (mean-reverting regime): Most aggressive allocation strategy, but rarely above fair value  
4. **Transaction costs**: Create path dependence - optimal quotes depend on current inventory
5. **Model Evaluation**: Evaluation on 100,975 unseen [Polymarket market price paths](https://www.kaggle.com/datasets/ismetsemedov/polymarket-prediction-markets/) shows most conservative policy is most robust to misspecification risk with expected wealth multiplier of [blank] and probability of 70% drawdown of [blank]
6. **Strategy Selection Decision Boundary**: Decision boundaries derived from likelihood surfaces enable optimal strategy selection

**Trading implication**: 
* Spreads change optimal policy dramatically
* Regime detection & confidence in current regime -> corresponding position sizing and timing

---
`trading_dp_outline.ipynb` - DP formulation & optimizations  

`trading_dp_implementation.ipynb` - Implementation & analysis

`trading_dp_evaluation.ipynb` - Evaluation on Polymarket data

`data/` - Polymarket price trajectories (Kaggle)

`results/` - Performance metrics 
# DP Evaluation Overview
We evaluate robustness of three DP strategies each trained on different regimes and tested across 453 [Polymarket price paths](https://www.kaggle.com/datasets/sandeepkumarfromin/full-market-data-from-polymarket): 

## DP Strategies
We use the same DP algorithm except vary the price transition matrix in accordance to these 3 market regimes, and describe the resulting optimal policy below (See Visualizations):
1. **Uniform Regime:** Conservative Policy - assumes no predictable price dynamics
2. **Random Walk Regime:** Moderate Policy - assumes momentum/drift optionality  
3. **Mean Reverting Regime:** Aggressive Policy - assumes prices revert to trader's subjective price (alpha decay)

For each of these strategies, we also try varying $p_{subj} \in [0.1,0.3,0.5,0.7,0.9]$ to reflect varying level's of a trader's (possibly misplaced) confidence about an event occuring. This allows us to see which regime is most robust to misspecification of $p_{subj}$.

## Non-DP Baselines [TODO]
We also test the following baselines:
1. **Buy and Hold**
2. **Kelly w/ Rebalancing**
3. **Spread Aware Kelly w/ Rebalancing**

## Metrics
For each of these strategies, we use the following metrics to evaluate performance:

**Average Wealth Return $~~e^{V_T}-1$**:
* How much long-run return is expected per $1 originally put into the trade
* This is the certainty equivalent growth rate for a log-utility investor

**Drawdown Risk $~~Pr(W_0 < 0.7 * W_T)$**:
* \# runs with ≥30% wealth drawdown  / total \# of runs

**Pre-resolution Average Wealth Return $e^{V_{T-1}}-1$:**
* How much long-run return is expected per $1 originally put into trade immediately before resolution date (How much wealth from trading vs betting on outcome)
* Considers how well strategies perform with early stopping

## Challenges
In increasing order of difficulty!
#### **1. Data Imbalance:** 
* Of the 453 Polymarket price paths, more than 90% of YES contracts resolve to 1. This could bias our results to favor high $p_{subj}$. Assuming the events of each price path are all independent of one another; we can flip outcomes randomly such that 50% of YES contracts resolve to 1 and 50% resolve to 0. 
#### **2: Non Stationarity of prices:**
* For each price path in the dataset, prices tend to trend towards 1 and 0. This is the case for many duration dependent markets (ex: which team makes the NBA playoffs), which means our trader's subjective price should also converage towards 1 and 0. We can resolve this by creating a time varying model that updates the trader's subjective price over time, or focus on markets that are not duration dependent.
#### **3. Data Dependence: [TODO]**
* In order to guarantee independent runs, we randomly and uniformly assign balanced price paths without replacement (see above) to different $p_{subj}$ such that we have 110 unique runs per $p_{subj}$ bucket. 
#### **4. Comparing $p_{subj}$ with binary (0/1) event resolution: [TODO]**
* We want to create a metric that measures how off our subjective probability $p_{subj}$ is with the actual event resolution. Let $outcome \in \{0,1\}$ represent the binary event resolution. Naively ($p_{subj} - outcome$) takes $|p_{subj}|$ different values. 
* However is there really a difference between predicting 10% probability for a YES contract that resolves to 1 and 90% probability for a YES contract that resolves 0? They are actually equally unlikely in log likelihood space; more generally for $LL(p_{subj},outcome)$:
    * $LL(x, 0) = log(1-p_{subj}) = log(x)$
    * $LL(1-x, 1) = log(p_{subj}) = log(x)$
* Hence we use $$|p_{subj}-outcome|$$ as our miscalibration metric. Runs with equal $|p_{subj}-outcome|$ have equal log-likelihood scores, representing equally "surprising" outcomes given the prediction. As this metric increases in value this misspecification is strictly larger. 

## Analysis [TODO]
First we randomize Polymarket event outcomes such that 50% of YES contracts resolve to 1 and 50% resolve to 0. For each strategy (DP and baselines) and for each $p_{subj} \in [0.1,0.2...,0.9]$, we randomly and uniformly assign price paths to each $strategy \times p_{subj}$ combination (about 110 paths per combination). Then we plot $|p_{subj}-outcome|$ vs our metrics (Average Wealth Return, Drawdown Ruin, Pre-resolution Average Wealth Return) to show our model performances at varying levels of robustness requirements.  

## Future Work [TODO]
#### **1. Comparing regime with price path transitions:**
* Since misspecification of expected vs actual price transitions can be both good and bad for long-run growth (unlike mis-valuing event resolution which is generally bad), further work needs to be done in this area to develop better metrics. It should be noted that if our DP model is trained perfectly on out-of-sample data, that any misspecification would reduce model performance.  
#### **2. Distribution Shift:**
* It's possible that these price paths are not representative of all Polymarket events. It's also possible that participants in different prediction markets behave differently. Further work needs to be done to create datasets that cover more markets are needed to cross-compare across more dimensions and ensure out of sample performance. 
#### **3. Multi outcome generalization**
* Apply this approach to prediction market events that aren't only binary; since at the end of the day a trader can only by YES or NO contracts per outcome, the bulk of work here is creating better models to capture a trader's subjective price.
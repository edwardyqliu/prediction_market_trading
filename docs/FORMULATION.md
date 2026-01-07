# DP Formulation Overview
## Problem Statement

Typically in prediction market binary options, for any given event/outcome with probability `c`, you can purchase like so:
- Buy 1 YES contract with price `c * (1 + buy spread)` that pays 1 if outcome occurs, 0 otherwise
- Buy 1 NO contract with price `(1 - c) * (1 + buy spread)` that pays 0 if outcome occurs, 1 otherwise

For vast majority of events, a resolution date is set such that the YES and NO contracts resolve at that time.
Note that in some you can't hold YES and NO simultaneously (buy NO = sell existing YES and vice versa).

Once contracts are owned, they can also be sold:
- Sell 1 YES contract with price `c * (1 - sell spread)`
- Sell 1 NO contract with price `(1 - c) * (1 - sell spread)`

## Variable Definitions

**Time Horizon:**
* $t \in [0, T]$ be the trading time steps ($T$ = last trading time step, resolve immediately after)

**Cash wealth:**
* $W_t \in [0, \infty)$ be the wealth level at time $t$

**Contracts:**
* $x_t \in [0, \infty)$ be the number of contracts held at time $t$ 
    * $x_t < 0$ = holding $|x_t|$ NO contracts
    * $x_t > 0$ = holding $|x_t|$ YES contracts
* $c_t \in [0, 1]$ be the contract price at time $t$
    
**Portfolio Value:**
* $P_{t,start}$ be the portfolio value at the start of time $t$
    * $P_{t,start} = 
    \begin{cases}
        W_{t-1}  + x_{t-1} c_t & x_{t-1} ≥ 0 \\
         W_{t-1} - x_{t-1} (1-c_t) & x_{t-1} < 0 \\
    \end{cases}
    $
* $P_{t,end}$ be the portfolio value at the end of time $t$ ($P_{t,start}$ minus spread costs)
    * $P_{t,end} = 
    \begin{cases}
        W_t + x_t c_t & & x_t ≥ 0 \\
         W_t - x_t (1-c_t) & & x_t < 0 \\
    \end{cases}
    $
    
**Portfolio Allocation at start of time $t$ (State Variable):**
* $\theta_t \in [-1,1] = 
    \begin{cases}
        \frac{x_{t-1} c_t} {P_{t,start}} & x_{t-1} ≥ 0 \\
        \frac{x_{t-1} (1 - c_t)} {P_{t,start}}  & x_t < 0
    \end{cases}
    $

**Portfolio Allocation at end of time $t$ (Action Variable):**
* Defined in terms of $P_{t,end}$:
    * $b'_t \in [-1,1] = 
        \begin{cases}
            \frac{x_t c_t} {P_{t,end}} & x_t ≥ 0 \\
            \frac{x_t (1 - c_t)} {P_{t,end}} & x_t < 0
        \end{cases}
        $

* Defined in terms of $P_{t,start}$:
    * $b_t = b'_t * \frac{P_{t,end}}{P_{t,start}} = 
        \begin{cases}
            \frac{x_t c_t} {P_{t,start}} & x_t ≥ 0 \\
            \frac{x_t (1 - c_t)} {P_{t,start}} & x_t < 0
        \end{cases}
        $

**Mark to Market (Price transition from t-1 to t)**
* YES Contract: 
    $R_{\text{YES},t} = \frac{c_t}{c_{t-1}}$
* NO Contract: $R_{\text{NO},t}=\frac{1-c_t}{1-c_{t-1}}$

**Market Spreads**
* YES Sell Spread:$~\gamma_{\text{YES},s}$
* YES Buy Spread:$~\gamma_{\text{YES},b}$
* NO Sell Spread:$~\gamma_{\text{NO},s}$
* NO Buy Spread:$~\gamma_{\text{NO},b}$

## Goal
Since DP generates a Policy and Value table, in order to leverage the properties of a bounded DP with $\theta_t$ and $b'_t$, we need to express everything in terms of $\theta_t$,$b'_t$, $c_t$.

Let our value function we seek to maximize be $V_T = \mathbb{E}[log(W_T)]$. In other words we seek to maximize the terminal expected log wealth at resolution time, since this maximizes long-term growth while penalizing excessively risky bets leading to drawdowns.

## Initial Condition
* $t=0,c_0 \in [0,1], W_0$
* $x_0 = 0, P_{t,start} = W_0, \theta_0=0, b'_0=0, b_0=0$

## Terminal Condition:
Let $p$ be the subjective (trader's) probability of outcome occuring; equivalently trader's value of YES contract.

**Case 1 (Own YES)**: With $p$ probability $|X_T|$ YES contracts resolve to 1, but in any case keep $W_T$ cash wealth
$$V_T = plog(W_T + |X_T|) +(1-p)log(W_T)$$
$$V_T= plog(W_T + \frac{|\theta_T|}{1-|\theta_T|}*W_T/C_T) +(1-p)log(W_T)$$
$$V_T= log(W_T) + plog(1 + \frac{|\theta_T|}{c_T(1-|\theta_T|)})$$

**Case 2 (Own NO)**: With $1-p$ probability $|X_T|$ NO contracts resolve to 1, but in any case keep $W_T$ cash wealth
$$V_T = plog(W_T)+(1-p)log(W_T + |X_T|)$$
$$V_T= plog(W_T) +(1-p)log(W_T + \frac{|\theta_T|}{1-|\theta_T|}*W_T/(1-c_T))$$
$$V_T= log(W_T) + (1-p)log(1+\frac{|\theta_T|}{(1-c_T)(1-|\theta_T|)})$$

## $\theta_{t+1}$ Update (Portfolio Mark to Market):
**Case 1 (Own YES)**
$$\theta_{t+1} = 
    \frac{x_tc_{t+1}}
         {W_t + x_t(c_{t+1})}$$
$$\theta_{t+1} = 
    \frac{x_tc_{t+1}/P_{t,end}} 
         {W_t/P_{t,end} + x_tc_{t+1}/P_{t,end}}$$
$$\theta_{t+1} = 
    \frac{b'_t * R_{\text{YES},t}} 
         {(1-b'_t) + b'_t * R_{\text{YES},t}}$$
**Case 2 (Own NO)**
$$\theta_{t+1} = 
    \frac{x_t(1-c_{t+1})}
         {W_t - x_t(1-c_{t+1})}$$
$$\theta_{t+1} = 
    \frac{x_t(1-c_{t+1})/P_{t,end}} 
         {W_t/P_{t,end} + x_t(1-c_{t+1})/P_{t,end}}$$
$$\theta_{t+1} = 
    \frac{b'_t * R_{\text{NO},t}} 
         {(1+b'_t) - b'_t * R_{\text{NO},t}}$$

**More Generally:**
$$\theta_{t+1} = 
    \frac{b'_t * R_{\text{NO},t}} 
         {(1-|b'_t|) + |b'_t| * R_{\text{NO},t}}$$

## Portfolio Value Update

$P_{t,end} * (1-|b'_t|) = P_{t,start}*(1-|\theta_t|)*(\frac{W_t}{W_{t-1}})$

$\frac{P_{t,end}}{P_{t,start}}=\frac{(1-|\theta_t|) * \frac{W_t}{W_{t-1}}}{1-|b'_t|}$

## Wealth Update (with Spread):
Let $\Beta=\frac{b'_t}{1-|b'_t|},~~~\Theta=\frac{\theta_t}{1-|\theta_t|},~~~y=\frac{W_t}{W_{t-1}}$.

And note that

$b_t = (\frac{b'_t}{1-|b'_t|})(1-|\theta_t|)(\frac{W_t}{W_{t-1}})$.

**Case 1:** 

$x_t ≥ x_{t-1}, ~ x_{t-1} < 0, ~ x_t < 0$

We sell $x_t - x_{t-1}$ NO contracts
    $$W_t = W_{t-1} + (x_t - x_{t-1})(1-c_t)(1-\gamma_{NO,s})$$
    $$W_t = W_{t-1} + \frac{W_{t-1}}{1-|\theta_t|} *(b_t - \theta_t)* (1-\gamma_{NO,s})$$
    $$W_t = W_{t-1} * (1 + \frac{b_t - \theta_t}{1 - |\theta_t|} * (1-\gamma_{NO,s}))$$
    $$y=1+(\Beta y-\Theta)*(1-\gamma_{NO,s})$$
    $$y-y \Beta(1-\gamma_{NO,s})=1-\Theta(1-\gamma_{NO,s})$$
    $$y=\frac{1-\Theta(1-\gamma_{NO,s})}{1-\Beta(1-\gamma_{NO,s})}$$

**Case 2:** 

$x_t ≥ x_{t-1}, ~ x_{t-1} < 0, ~ x_t ≥ 0$

We sell $x_{t-1}$ NO contracts and buy $x_t$ YES contracts
    $$W_t = W_{t-1} - x_{t-1}(1-c_t)(1-\gamma_{NO,s}) - x_tc_t(1+\gamma_{\text{YES},b})$$
    $$W_t = W_{t-1} - \frac{W_{t-1}}{1-|\theta_t|}*\theta_t*(1-\gamma_{NO,s}) - \frac{W_{t-1}}{1-|\theta_t|}*b_t*(1 + \gamma_{\text{YES},b})$$
    $$W_t = W_{t-1}(1-\frac{\theta_t (1-\gamma_{NO,s}) + b_t(1+\gamma_{\text{YES,b}})}{1-|\theta_t|})$$
    $$y=1-\Theta(1-\gamma_{NO,s})-\Beta y(1+\gamma_{\text{YES},b})$$
    $$y(1+B(1-\gamma_{\text{YES},b}))=1-\Theta(1-\gamma_{NO,s})$$
    $$y=\frac{1-\Theta(1-\gamma_{NO,s})}{1+\Beta(1+\gamma_{\text{YES},b})}$$

**Case 3:** 

$x_t ≥ x_{t-1}, ~ x_{t-1} ≥ 0, ~ x_t ≥ 0$

We buy $x_t - x_{t-1}$ YES contracts
    $$W_t = W_{t-1} - (x_t - x_{t-1})(c_t)(1+\gamma_{\text{YES},b})$$
    $$W_t = W_{t-1} - \frac{W_{t-1}}{1-|\theta_t|} *(b_t - \theta_t) * (1+\gamma_{\text{YES},b})$$
    $$W_t = W_{t-1} * (1 - \frac{b_t - \theta_t}{1 - |\theta_t|} * (1+\gamma_{\text{YES},b}))$$
    $$y=1-(\Beta y - \Theta)*(1+\gamma_{\text{YES},b})$$
    $$y=\frac{1+\Theta(1+\gamma_{\text{YES},b})}{1+\Beta(1+\gamma_{\text{YES},b})}$$

**Case 4:** 

$x_t < x_{t-1}, ~ x_{t-1} ≥ 0, ~ x_t ≥ 0$

We sell $-(x_t - x_{t-1})$ YES contracts
    $$W_t = W_{t-1} - (x_t - x_{t-1})(c_t)(1-\gamma_{\text{YES},s})$$
    $$W_t = W_{t-1} - \frac{W_{t-1}}{1-|\theta_t|} *(b_t - \theta_t) * (1-\gamma_{\text{YES},s})$$
    $$W_t = W_{t-1} * (1 - \frac{b_t - \theta_t}{1 - |\theta_t|} * (1-\gamma_{\text{YES},s}))$$

By symmetry to Case 3:

$$y=\frac{1+\Theta(1-\gamma_{\text{YES},s})}{1+\Beta(1-\gamma_{\text{YES},s})}$$

**Case 5:** 

$x_t < x_{t-1}, ~ x_{t-1} ≥ 0, ~ x_t < 0$

We sell $x_{t-1}$ YES contracts and buy $x_t$ NO contracts
    $$W_t = W_{t-1} + x_{t-1}(c_t)(1-\gamma_{\text{YES},s}) + x_t(1-c_t)(1+\gamma_{NO,b})$$
    $$W_t = W_{t-1} + \frac{W_{t-1}}{1-|\theta_t|}*\theta_t*(1-\gamma_{\text{YES},s}) + \frac{W_{t-1}}{1-|\theta_t|}*b_t*(1 + \gamma_{NO,b})$$
    $$W_t = W_{t-1} * (1+\frac{\theta_t (1-\gamma_{\text{YES},s}) + b_t(1+\gamma_{NO,b})}{1-|\theta_t|})$$
    $$y=1+\Theta(1-\gamma_{\text{YES,s}})+\Beta y(1+\gamma_{NO,b})$$
    $$y=\frac{1+\Theta(1-\gamma_{\text{YES},s})}{1-\Beta(1+\gamma_{NO,b})}$$
    

**Case 6:** 

$x_t < x_{t-1}, ~ x_{t-1} < 0, ~ x_t < 0$

We buy $-(x_t - x_{t-1})$ NO contracts
    $$W_t = W_{t-1} + (x_t - x_{t-1})(1-c_t)(1+\gamma_{NO,b})$$
    $$W_t = W_{t-1} + \frac{W_{t-1}}{1-|\theta_t|} *(b_t - \theta_t)* (1+\gamma_{NO,b})$$
    $$W_t = W_{t-1} * (1 + \frac{b_t - \theta_t}{1 - |\theta_t|} * (1+\gamma_{NO,b}))$$

By symmetry to Case 1:
    $$y=\frac{1-\Theta(1+\gamma_{NO,b})}{1-\Beta(1+\gamma_{NO,b})}$$

Note: $\Theta$ and $\Beta$ can be thought of as starting and ending coverage ratio (contract value / cash wealth value). 

More generally we have:

$$y=\frac{1+|\Theta|(1+\gamma_{\Theta})}{1+|\Beta|(1+\gamma_{\Beta})}$$

or

$$y=\frac{\frac{1}{1+\gamma_{\Theta}}+|\Theta|}{\frac{1}{1+\gamma_{\Beta}}+|\Beta|}$$
where $\gamma$ is positive if buying and negative if selling. So long as $|\Theta|,|\Beta| \in[0,\infty)$ and $\gamma \in (-1,1)$, $y$ is always positive ($W_t$ will never flip negative).

## Recursion Step
At each $t \in [0..T]$,
$$V_t = log(W_{t-1}) + v_t(\theta_t, c_t)$$
Where $v_t(\theta_t,c_t)$ is the growth function of log wealth between the start and end of time $t$, $log(W_{t-1})$ is the cash wealth at start of time $t$.

$$V_{t+1} = log(W_t) + v_{t+1}(\theta_{t+1}, c_{t+1})$$

Recursion Step (Bellman Equation):
$$V_{t+1} = log(W_{t-1}) + \max_{b'_t \in[-1,1]}\mathbb{E_{c_t|c_{t-1}}}[log(y)+v_{t+1}(\theta_{t+1},c_{t+1})]$$
$$v_t(\theta_t, c_t) = \max_{b'_t \in[-1,1]}\mathbb{E_{c_t|c_{t-1}}}[log(y) + v_{t+1}(\theta_{t+1}, c_{t+1})]$$

## Runtime Analysis
### Space Complexity
Hence our space complexity for policy (and value) grid is
$$ O(T * |\theta| * |c|)$$
where $|\theta|$ and $|c|$ are bounded between -1 and 1, and can be represented as a constant number of grid points; so really space complexity is $$O(T * N^2)$$ if both values are represented with the same granularity for some constant $N$. 

### Time Complexity
And our time complexity for generating such a policy is 
$$ O(T * |\theta| * |b'| * |c|^2)$$
Since for each element in the DP grid we try out |$b'$| different allocations and taking expected value over $|c|$ different prices. $|b'|$ is also bounded by -1 and 1, and if it also is represented with the same granularity as the other bounded values we have runtime complexity of
$$ O(T * N^4)$$
which makes the problem tractable. Note that this is with a full price transition matrix; if we were to swap out with a binomial/trinomial price transition model runtime would be less. 

### Optimized Implementation
We observe that the expectation term $\mathbb{E}_{c_{t+1}|c_t}[v_{t+1}(\theta_{t+1}, c_{t+1})]$ depends on $b'_t$ and $c_t$ but **not on $\theta_t$**. This enables factorization:

**Precompute expectations**: For all $(b'_t, c_t)$ pairs, compute:
   $$F(b'_t, c_t) = \mathbb{E}_{c_{t+1}|c_t}[v_{t+1}(\theta_{t+1}(b'_t, c_t, c_{t+1}), c_{t+1})]$$
   Complexity: $O(N_b × N_c × N_c) = O(N^3)$

**Optimize allocations**: For all $(\theta_t, c_t)$ pairs:
   $$v_t(\theta_t, c_t) = \max_{b'} [\log(y_t(\theta_t, b'_t)) + F(b'_t, c_t)]$$
   Complexity: $O(N_\theta × N_b) = O(N^2)$

Total per time step: $O(N^3)$, giving overall complexity of:

$$O(T × N^3)$$

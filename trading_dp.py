import numpy as np
from typing import Tuple
from tqdm import tqdm
from scipy.stats import norm


def make_transition_matrix(fair_value, volatility, mean_reversion=0.2, n_prices=99):
    """
    Random walk in log-odds space with mean reversion:
      L_t = log(c_t / (1 - c_t))
      L_{t+1} = L_t + k*(L_fair - L_t) + e, e ~ N(0, sigma^2)
      c_{t+1} = 1 / (1 + exp(-L_{t+1}))
    
    Parameters:
    -----------
    fair_value : float
        Long-term mean in probability space
    volatility : float
        Std dev of shocks in log-odds space
    mean_reversion : float
        Speed of mean reversion (0 = no reversion, 1 = instant)
    n_prices : int
        Number of price grid points
    """
    # Price grid
    prices = np.linspace(0, 1, n_prices)
    prices_safe = np.clip(prices, .01, .99)
    
    # Convert fair_value to log-odds
    L_fair = np.log(fair_value / (1 - fair_value))
    
    # Build transition matrix
    M = np.zeros((n_prices, n_prices))
    
    for i, c_t in enumerate(prices_safe):
        # Current log-odds
        L_t = np.log(c_t / (1 - c_t))
        
        L_next_mean = L_t + mean_reversion * (L_fair - L_t)
        
        # Next price distribution in log-odds space
        L_grid = np.log(prices_safe / (1 - prices_safe))
        pdf_logit = norm.pdf(L_grid, loc=L_next_mean, scale=volatility)
        
        # Jacobian: dL/dc = 1/(c*(1-c))
        jacobian = 1 / (prices_safe * (1 - prices_safe))
        pdf_prob = pdf_logit * jacobian
        
        # Normalize
        pdf_prob = np.maximum(pdf_prob, 1e-10)
        M[i, :] = pdf_prob / pdf_prob.sum()
    
    return M


def run_dp(
    model_trans: np.ndarray,
    p_subj: float,
    gamma_b: float = 0.05,
    gamma_s: float = 0.10,
    T: int = 20,
    n_theta: int = 51,
    n_b: int = 51,
    n_prices: int = 99
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dynamic programming for optimal binary options trading with transaction costs.
    
    Args:
        model_trans: Transition matrix [n_prices, n_prices], model_trans[i,j] = P(price_j | price_i)
        p_subj: Subjective probability of contract paying out (0-1)
        gamma_b: Transaction cost for buying (fraction of mid price)
        gamma_s: Transaction cost for selling (fraction of mid price)  
        T: Number of time steps (including terminal)
        n_theta: Grid size for portfolio fraction in contracts
        n_b: Grid size for target allocation
        n_prices: Grid size for contract prices
        
    Returns:
        V: Value function [T+1, n_theta, n_prices]
        policy: Optimal target allocation [T, n_theta, n_prices]
    """
    
    # ========== STATE/ACTION GRIDS ==========
    theta_grid = np.linspace(0, 0.99, n_theta)      # Portfolio fraction in contracts
    price_grid = np.linspace(0.01, 0.99, n_prices)  # Contract mid prices
    b_grid = np.linspace(0, 0.99, n_b)              # Target allocation
    
    # ========== PRE-COMPUTE THETA_NEXT INDICES ==========
    # R_matrix[i,j] = price_grid[j] / price_grid[i] = c_next / c_current
    R_matrix = price_grid[None, :] / price_grid[:, None]
    
    # theta_next_idx[b, current_price, next_price] -> theta grid index
    theta_next_idx = np.zeros((n_b, n_prices, n_prices), dtype=np.int16)
    
    for i_b, b in enumerate(b_grid):
        b_safe = np.clip(b, 1e-12, 1 - 1e-12)
        numerator = R_matrix * b_safe
        denominator = 1 - b_safe + numerator + 1e-12
        theta_next_vals = np.clip(numerator / denominator, 0, 1)
        
        # Convert to grid indices
        idxs = (theta_next_vals * (n_theta - 1) + 0.5).astype(int)
        theta_next_idx[i_b] = np.clip(idxs, 0, n_theta - 1)
    
    # ========== IMMEDIATE REWARDS WITH TRANSACTION COSTS ==========
    theta_2d = theta_grid[:, None]  # (n_theta, 1)
    b_2d = b_grid[None, :]          # (1, n_b)
    
    # Buying: b > theta
    buy_mask = b_2d > theta_2d + 1e-8
    buy_reward = np.log(np.clip(1 - theta_2d - (b_2d - theta_2d) * (1 + gamma_b), 1e-12, np.inf))
    
    # Selling: b < theta  
    sell_mask = b_2d < theta_2d - 1e-8
    sell_reward = np.log(np.clip(1 - theta_2d - (b_2d - theta_2d) * (1 - gamma_s), 1e-12, np.inf))
    
    # No trade: b = theta
    hold_reward = np.log(np.clip(1 - theta_2d, 1e-12, np.inf))
    
    immediate = np.where(buy_mask, buy_reward, 
                        np.where(sell_mask, sell_reward, hold_reward))
    
    # ========== DP TABLES ==========
    V = np.zeros((T + 1, n_theta, n_prices))
    policy = np.zeros((T, n_theta, n_prices))
    
    # ========== TERMINAL CONDITION ==========
    Theta, C = np.meshgrid(theta_grid, price_grid, indexing='ij')
    mask = Theta > 1e-20
    V_terminal = np.zeros_like(Theta)
    V_terminal[mask] = p_subj * np.log(1 + Theta[mask] / (C[mask] * (1 - Theta[mask])))
    V[T] = V_terminal
    
    # ========== CURRENT PERIOD COST ==========
    current_cost = -np.log(np.clip(1 - theta_grid, 1e-12, None))
    
    # ========== BACKWARD INDUCTION ==========
    for t in tqdm(reversed(range(T)), total=T, desc="DP backward induction"):
        if t == T - 1:
            # Last trading period: next period is settlement
            settle_idx = np.clip((b_grid * (n_theta - 1) + 0.5).astype(int), 0, n_theta - 1)
            
            for i_c in range(n_prices):
                settle_vals = V[T, settle_idx, i_c]  # Terminal values for each b
                total_vals = current_cost[:, None] + immediate + settle_vals[None, :]
                best_b_idx = np.argmax(total_vals, axis=1)
                V[t, :, i_c] = total_vals[np.arange(n_theta), best_b_idx]
                policy[t, :, i_c] = b_grid[best_b_idx]
                
        else:
            # Normal periods: precompute expected future values
            V_next = V[t + 1]
            expected_future = np.zeros((n_b, n_prices))
            
            # Precompute E[V_next | b, current_price] for all (b, price) pairs
            for i_b in range(n_b):
                for i_c in range(n_prices):
                    trans_probs = model_trans[i_c, :]
                    theta_idx = theta_next_idx[i_b, i_c, :]
                    v_vals = V_next[theta_idx, np.arange(n_prices)]
                    expected_future[i_b, i_c] = np.dot(trans_probs, v_vals)
            
            # Optimized inner loop: vectorize over theta for each price
            for i_c in range(n_prices):
                exp_future_for_price = expected_future[:, i_c]
                total_vals = current_cost[:, None] + immediate + exp_future_for_price[None, :]
                best_b_idx = np.argmax(total_vals, axis=1)
                V[t, :, i_c] = total_vals[np.arange(n_theta), best_b_idx]
                policy[t, :, i_c] = b_grid[best_b_idx]
    
    return V, policy
import numpy as np
from typing import Tuple
from scipy.stats import norm
import matplotlib.pyplot as plt

def make_transition_matrix(fair_value: float = 0.5, 
                          volatility: float = 0.2, 
                          mean_reversion: float = 0.2, 
                          dist: str = None, 
                          n_prices: int = 99,
                          C_MIN : float = .01,
                          C_MAX : float = .99):
    """
    Random walk in log-odds space with mean reversion.
    
    Avoid p = 0 or 1 exactly by using eps.
    
    dist == Uniform returns uniform transition matrix
    """
    if dist == "uniform":
        M = np.ones((n_prices, n_prices)) / n_prices
        return M
    elif dist != None:
        raise NotImplementedError(f"Distribution {dist} not implemented")
    
    # Avoid 0 and 1 exactly
    p_grid = np.linspace(C_MIN, C_MAX, n_prices)
    
    # Convert fair_value to log-odds
    L_fair = np.log(fair_value / (1 - fair_value))
    
    # Build transition matrix
    M = np.zeros((n_prices, n_prices))
    L_grid = np.log(p_grid / (1 - p_grid))
    
    for i, p_t in enumerate(p_grid):
        # Current log-odds
        L_t = np.log(p_t / (1 - p_t))
        
        # Mean in log-odds space after mean reversion
        L_next_mean = L_t + mean_reversion * (L_fair - L_t)
        
        # Density in log-odds space evaluated on grid
        pdf_logit = norm.pdf(L_grid, loc=L_next_mean, scale=volatility)
        
        # Convert density to probability space using Jacobian
        # f_p(p) = f_L(logit(p)) * |dL/dp| = f_L * 1/(p*(1-p))
        jacobian = 1 / (p_grid * (1 - p_grid))
        pdf_prob = pdf_logit * jacobian
        
        # Normalize (sum over possible next prices = 1)
        M[i, :] = pdf_prob / pdf_prob.sum()
    
    return M

def get_theta_bounds(c_min, c_max, b_tick_min, b_tick_max):
    """
    Get the bounds of theta given the price and allocation bounds
    
    Returns:
        x0: float
            Lower bound of theta
        x1: float
            Upper bound of theta
    """
    x0 = None
    x1 = None
    
    if b_tick_min < 0:
        R_min = (1 - c_min) / (1 - c_max)
    else:
        R_min = c_min / c_max
    
    if b_tick_max < 0:
        R_max = (1 - c_max) / (1 - c_min)
    else:
        R_max = c_max / c_min
    
    x0 = b_tick_min * R_min / (1 - abs(b_tick_min) + abs(b_tick_min) * R_min)
    x1 = b_tick_max * R_max / (1 - abs(b_tick_max) + abs(b_tick_max) * R_max)
    return x0, x1
    

def simulate_one_path(path, policy, T_MAX, C_MIN, C_MAX, THETA_MIN, THETA_MAX, 
    n_prices, n_theta, gamma_yes_b, gamma_yes_s, gamma_no_b, gamma_no_s, debug=False):
    eps = 1e-12
        
    T_start = T_MAX - len(path)
    
    W0 = 100
    W = W0
    x = 0   # Num contracts
    contract_value = 0
    max_pct_drawdown_curr = 0
    
    portfolio_value_prev = None
    portfolio_value_curr = W + contract_value
    
    theta = 0
    b = 0
    
    for t in range(T_start, T_MAX - 1):
        # Mark to Market Update
        c_curr = path[t - T_start]
        price_idx = np.clip(int((c_curr - C_MIN) / (C_MAX - C_MIN) * (n_prices - 1)), 0, n_prices - 1)
        
        if abs(b - theta) < eps:
            pass
        elif b >= 0:
            contract_value = c_curr * x
            theta = contract_value / (W + contract_value)
        else:
            contract_value = (1 - c_curr) * x
            theta = -contract_value / (W + contract_value)
            
        portfolio_value_curr = W + contract_value
        
        if portfolio_value_prev is not None and portfolio_value_prev > 0:
            max_pct_drawdown_curr = min(max_pct_drawdown_curr, 
                                        (portfolio_value_curr - portfolio_value_prev) / portfolio_value_prev)
        elif portfolio_value_prev == 0:
            max_pct_drawdown_curr = -1
            
        theta = np.clip(theta, THETA_MIN, THETA_MAX)
        theta_idx = np.clip(((theta + THETA_MAX) / (2 * THETA_MAX) * (n_theta - 1)).astype(int), 0, n_theta - 1)
        
        # Optimal Allocation
        b = policy[t, theta_idx, price_idx]
        Beta = abs(b) / (1 - abs(b))
        Theta = abs(theta) / (1 - abs(theta))
        if debug:   
            print(f"t: {t}, theta: {theta} c_curr: {c_curr}")

        
        # Trading Update
        if abs(b - theta) < eps:    # Do nothing
            pass
        
        elif b >= theta:  # Go Long outcome
            if theta < 0 and b < 0: # Sell NO
                W_new = W * (1 + Theta * (1 - gamma_no_s)) / (1 + Beta * (1 - gamma_no_s))                        
                x_new = (W_new * abs(b) / (1 - abs(b))) // (1 - c_curr)
                W += abs(x_new - x) * (1 - c_curr) * (1 - gamma_no_s)
                x = x_new
            
            elif theta < 0 and b >= 0:  # Sell NO then buy YES
                W_new = W * (1 + Theta * (1 - gamma_no_s)) / (1 + Beta * (1 + gamma_yes_b))
                W += x * (1 - c_curr) * (1 - gamma_no_s)
                
                x_new = (W_new * abs(b) / (1 - abs(b))) // c_curr
                x_new = min(x_new, W // (c_curr * (1 + gamma_yes_b)))
                W -= x_new * c_curr * (1 + gamma_yes_b)
                x = x_new
            
            elif theta >= 0 and b >= 0: # Buy YES
                W_new = W * (1 + Theta * (1 + gamma_yes_b)) / (1 + Beta * (1 + gamma_yes_b))    
                x_new = (W_new * abs(b) / (1 - abs(b))) // c_curr
                x_new = min(x_new, x + W // (c_curr * (1 + gamma_yes_b)))
                W -= abs(x_new - x) * c_curr * (1 + gamma_yes_b)
                x = x_new
        
        else:   # Go Short outcome
            if theta >= 0 and b >= 0:   # Sell YES
                W_new = W * (1 + Theta * (1 - gamma_yes_s)) / (1 + Beta * (1 - gamma_yes_s))                        
                x_new = (W_new * abs(b) / (1 - abs(b))) // c_curr
                W += abs(x_new - x) * c_curr * (1 - gamma_yes_s)
                x = x_new
            
            elif theta >= 0 and b < 0:  # Sell YES then buy NO 
                W_new = W * (1 + Theta * (1 - gamma_yes_s)) / (1 + Beta * (1 + gamma_no_b))
                W += x * c_curr * (1 - gamma_yes_s)
                
                x_new = (W_new * abs(b) / (1 - abs(b))) // (1 - c_curr)
                x_new = min(x_new, W // ((1 - c_curr) * (1 + gamma_no_b)))
                W -= x_new * (1 - c_curr) * (1 + gamma_no_b)
                x = x_new
            
            elif theta <= 0 and b < 0:   # Buy NO
                W_new = W * (1 + Theta * (1 + gamma_no_b)) / (1 + Beta * (1 + gamma_no_b))          
                x_new = (W_new * abs(b) / (1 - abs(b))) // (1 - c_curr)
                x_new = min(x_new, x + W // ((1 - c_curr) * (1 + gamma_no_b)))
                W -= abs(x_new - x) * (1 - c_curr) * (1 + gamma_no_b)
                x = x_new
        
        portfolio_value_prev = portfolio_value_curr
        if debug:
            if b >= 0:
                print(f"t: {t}, W_new: {W}, x_new: {x}, b: {b}, b_act: {(x * c_curr) / (W + x * c_curr)}\n")
            else:
                print(f"t: {t}, W_new: {W}, x_new: {x}, b: {b}, b_act: {(-x * (1 - c_curr)) / (W + x * (1 - c_curr))}\n")
            
            
        
    # Resolve after time T_MAX
    W_prev = W
    c_curr = path[-1]
    if abs(b - theta) < eps:
        if theta >= 0:
            W += x * c_curr
        else:
            W += x * (1 - c_curr)
    elif b >= 0:
        W += x * c_curr
    else:
        W += x * (1 - c_curr)
    
    x = 0
    theta=0
    b=0
    if debug:
        print(f"t: {t + 1}, W: {W}, c_curr: {c_curr}, x: {x}, theta: {theta}, b: {b}")
    
    if portfolio_value_prev is not None and portfolio_value_prev > 0:
        max_pct_drawdown_curr = min(max_pct_drawdown_curr, 
                                    (W - portfolio_value_prev) / portfolio_value_prev)
    elif portfolio_value_prev == 0:
        max_pct_drawdown_curr = -1
    
    e_log_wT = np.log(W) - np.log(W0)
    e_log_wTm1 = np.log(W_prev) - np.log(W0)
    pct_drawdown = min((W - W0) / W0, 0)
    max_pct_drawdown = max_pct_drawdown_curr
    return e_log_wT, max_pct_drawdown, pct_drawdown, e_log_wTm1
    
@staticmethod
def simulate_trading_static(
    policy: np.ndarray,
    paths: list,
    T_MAX: int,
    C_MIN: float,
    C_MAX: float,
    THETA_MIN: float,
    THETA_MAX: float,
    n_prices: int,
    n_theta: int,
    gamma_yes_b: float = 0.05,
    gamma_yes_s: float = 0.10,
    gamma_no_b: float = 0.05,
    gamma_no_s: float = 0.10,
    debug: bool = False,
    n_jobs: int = -1
):
    from tqdm.contrib.concurrent import process_map
    from functools import partial
    
    # Create a worker function with all parameters bound except path
    worker = partial(simulate_one_path,
                    policy=policy,
                    T_MAX=T_MAX,
                    C_MIN=C_MIN,
                    C_MAX=C_MAX,
                    THETA_MIN=THETA_MIN,
                    THETA_MAX=THETA_MAX,
                    n_prices=n_prices,
                    n_theta=n_theta,
                    gamma_yes_b=gamma_yes_b,
                    gamma_yes_s=gamma_yes_s,
                    gamma_no_b=gamma_no_b,
                    gamma_no_s=gamma_no_s,
                    debug=debug)
    
    # Run in parallel with progress bar
    results = process_map(
        worker,
        paths,
        max_workers=n_jobs if n_jobs > 0 else None,
        desc="Simulating paths",
        chunksize=1
    )
    
    # Convert to arrays
    e_log_wT, max_pct_drawdown, pct_drawdown, e_log_wTm1 = zip(*results)
    return (np.array(e_log_wT), np.array(max_pct_drawdown),
            np.array(pct_drawdown), np.array(e_log_wTm1))
        
class TradingDP():
    """
    TradingDP class for prediction market trading
    
    Args:
        n_prices (int): Number of prices
        n_b_tick (int): Number of b ticks
        n_theta (int): Number of theta
        C_MIN (float): Minimum contract price
        C_MAX (float): Maximum contract price
        B_TICK_MIN (float): Minimum b tick
        B_TICK_MAX (float): Maximum b tick
        THETA_MIN (float): Minimum theta
        THETA_MAX (float): Maximum theta
    """
    
    def __init__(self, 
                 n_prices : int = 99, 
                 n_b_tick : int = 100,
                 n_theta : int = 3000,
                 C_MIN : float = .01, 
                 C_MAX : float = .99,
                 B_TICK_MIN : float = -.80, 
                 B_TICK_MAX : float = .80, 
                 THETA_MIN : float = -.999, 
                 THETA_MAX : float = .999):
        
        self.n_prices = n_prices
        self.n_b_tick = n_b_tick
        self.n_theta = n_theta
        
        self.C_MIN = C_MIN
        self.C_MAX = C_MAX
        
        self.B_TICK_MIN = B_TICK_MIN
        self.B_TICK_MAX = B_TICK_MAX
        
        self.THETA_MIN = THETA_MIN
        self.THETA_MAX = THETA_MAX
    
    def run_dp(
        self,
        model_trans: np.ndarray,
        p_subj: float,
        gamma_yes_b: float = 0.05,
        gamma_yes_s: float = 0.10,
        gamma_no_b: float = 0.05,
        gamma_no_s: float = 0.10,
        T: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        DP Algorithm for prediction market trading

        Args:
            model_trans (np.ndarray): Trader's price transition matrix
            p_subj (float): Trader's subjective probability of event outcome occuring
            gamma_yes_b (float, optional): YES contract buy spread. Defaults to 0.05.
            gamma_yes_s (float, optional): YES contract sell spread. Defaults to 0.10.
            gamma_no_b (float, optional): NO contract buy spread. Defaults to 0.10.
            gamma_no_s (float, optional): NO contract sell spread. Defaults to 0.05.
            T (int, optional): Time horizon. Defaults to 20.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                V: np.ndarray
                    Value function
                policy: np.ndarray
                    Optimal contract allocation policy
        """
        
        # ========== STATE/ACTION GRIDS ==========
        theta_grid = np.linspace(self.THETA_MIN, self.THETA_MAX, self.n_theta)
        price_grid = np.linspace(self.C_MIN, self.C_MAX, self.n_prices)
        b_tick_grid = np.linspace(self.B_TICK_MIN, self.B_TICK_MAX, self.n_b_tick)
        
        # ========== PRECOMPUTE MARK TO MARKET UPDATE ==========
        R_yes_matrix = price_grid[None, :] / price_grid[:, None]
        R_no_matrix = (1 - price_grid[None, :]) / (1 - price_grid[:, None])
        
        theta_next_idx = np.zeros((self.n_b_tick, self.n_prices, self.n_prices), dtype=np.int16)
        
        for i_b_tick, b_tick in enumerate(b_tick_grid):
            if b_tick >= 0:
                R_matrix = R_yes_matrix
                numerator = R_matrix * b_tick
                denominator = 1 - b_tick + numerator
            else:
                R_matrix = R_no_matrix
                numerator = R_matrix * b_tick
                denominator = 1 + b_tick - numerator
            
            theta_next_vals = numerator / denominator
            
            idxs = ((theta_next_vals + self.THETA_MAX) / (2 * self.THETA_MAX) * (self.n_theta - 1)).astype(int)
            idxs = np.clip(idxs, 0, self.n_theta - 1)
            theta_next_idx[i_b_tick] = idxs
        
        # ========== PRECOMPUTE WEALTH UPDATE ==========
        theta_abs = np.abs(theta_grid)
        b_tick_abs = np.abs(b_tick_grid)
        
        # Avoid division by zero
        epsilon = 1e-12
        
        Theta = theta_abs / np.clip(1 - theta_abs, epsilon, np.inf)
        Beta = b_tick_abs / np.clip(1 - b_tick_abs, epsilon, np.inf)
        
        immediate = np.zeros((self.n_theta, self.n_b_tick))
        
        for i in range(self.n_theta):
            for j in range(self.n_b_tick):
                theta = theta_grid[i]
                b_tick = b_tick_grid[j]
                
                if abs(b_tick - theta) < epsilon:
                    continue
                    
                if theta >= 0:
                    if b_tick >= 0:
                        if b_tick >= theta:
                            spread_theta = 1 + gamma_yes_b
                            spread_beta = 1 + gamma_yes_b
                        else:
                            spread_theta = 1 - gamma_yes_s
                            spread_beta = 1 - gamma_yes_s
                    else:
                        spread_theta = 1 - gamma_yes_s
                        spread_beta = 1 + gamma_no_b
                else:
                    if b_tick <= 0:
                        if abs(b_tick) >= abs(theta):
                            spread_theta = 1 + gamma_no_b
                            spread_beta = 1 + gamma_no_b
                        else:
                            spread_theta = 1 - gamma_no_s
                            spread_beta = 1 - gamma_no_s
                    else:
                        spread_theta = 1 - gamma_no_s
                        spread_beta = 1 + gamma_yes_b
                
                num = np.clip(1 + Theta[i] * spread_theta, epsilon, np.inf)
                den = np.clip(1 + Beta[j] * spread_beta, epsilon, np.inf)
                immediate[i, j] = np.log(num) - np.log(den)
        
        # ========== DP TABLES ==========
        V = np.zeros((T + 1, self.n_theta, self.n_prices))
        policy = np.zeros((T, self.n_theta, self.n_prices))
        
        # ========== TERMINAL CONDITION ==========
        theta_T, C = np.meshgrid(theta_grid, price_grid, indexing='ij')
        theta_T_abs = np.abs(theta_T)
        mask = theta_T >= 0
        
        # Avoid numerical issues
        theta_T_abs_clipped = np.clip(theta_T_abs, 0, 1 - epsilon)
        denom_yes = np.clip(C * (1 - theta_T_abs_clipped), epsilon, np.inf)
        denom_no = np.clip((1 - C) * (1 - theta_T_abs_clipped), epsilon, np.inf)
        
        V_terminal = np.zeros_like(theta_T)
        V_terminal[mask] = p_subj * np.log(1 + theta_T_abs_clipped[mask] / denom_yes[mask])
        V_terminal[~mask] = (1 - p_subj) * np.log(1 + theta_T_abs_clipped[~mask] / denom_no[~mask])
        V[T] = V_terminal
        
        expected_future = np.zeros((self.n_b_tick, self.n_prices))
        # ========== BACKWARD INDUCTION ==========
        for t in reversed(range(T)):
                V_next = V[t + 1]
                
                for i_b_tick in range(self.n_b_tick):
                    for i_c in range(self.n_prices):
                        trans_probs = model_trans[i_c, :]
                        theta_idx = theta_next_idx[i_b_tick, i_c, :]
                        v_vals = V_next[theta_idx, np.arange(self.n_prices)]
                        expected_future[i_b_tick, i_c] = np.dot(trans_probs, v_vals)
                
                for i_c in range(self.n_prices):
                    total_vals = immediate + expected_future[:, i_c][None, :]
                    best_b_idx = np.argmax(total_vals, axis=1)
                    V[t, :, i_c] = total_vals[np.arange(self.n_theta), best_b_idx]
                    policy[t, :, i_c] = b_tick_grid[best_b_idx]
        
        return V, policy    
    
    def plot_policy(self, policy: np.ndarray):
        """
        Plot the optimal allocation policy over time
        
        Args:
            policy (np.ndarray): Policy
        """
        T = policy.shape[0]
        # Create heatmaps for key time points
        key_times = [0, T//4, T//2, 3*T//4, T-2, T-1]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, t in enumerate(key_times):
            ax = axes[idx]
            
            # Create heatmap
            im = ax.imshow(policy[t], 
                        aspect='auto',
                        origin='lower',
                        extent=[self.C_MIN, self.C_MAX, self.THETA_MIN, self.THETA_MAX],
                        vmin=self.B_TICK_MIN, vmax=self.B_TICK_MAX,
                        cmap='RdYlBu')
            
            ax.set_xlabel('Contract Price')
            ax.set_ylabel('Current theta')
            ax.set_title(f't = {t}' + (' (First)' if t == 0 else ' (Last)' if t == T-1 else ''))
            
            # Add colorbar only to last plot
            if idx == len(key_times) - 1:
                fig.colorbar(im, ax=ax, label='Optimal b\'')

        plt.suptitle("Optimal Contract Allocation across varying T, p, theta")
        plt.show()
        
    
    def simulate_trading(self, policy, paths, T_MAX, gamma_yes_b=0.05, gamma_yes_s=0.10,
                        gamma_no_b=0.05, gamma_no_s=0.10, debug: bool = False, n_jobs=-1):
        """Instance method wrapper for the static method"""
        return simulate_trading_static(
            policy=policy,
            paths=paths,
            T_MAX=T_MAX,
            C_MIN=self.C_MIN,
            C_MAX=self.C_MAX,
            THETA_MIN=self.THETA_MIN,
            THETA_MAX=self.THETA_MAX,
            n_prices=self.n_prices,
            n_theta=self.n_theta,
            gamma_yes_b=gamma_yes_b,
            gamma_yes_s=gamma_yes_s,
            gamma_no_b=gamma_no_b,
            gamma_no_s=gamma_no_s,
            debug = debug,
            n_jobs=n_jobs
        )
    
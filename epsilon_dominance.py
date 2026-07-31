"""
epsilon_dominance.py  —  Epsilon-Dominance Engine (Optimized)
=============================================================

Optimized version with faster bootstrap computation.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class EpsilonDominanceEngine:
    """
    Model-Free Stochastic Dominance with E-Processes.
    Optimized for speed with vectorized operations.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.threshold = config.get("threshold", 20.0)
        self.epsilon = config.get("epsilon", 0.01)
        self.test_type = config.get("test_type", "first_order")
        self.n_bootstrap = config.get("n_bootstrap", 1000)
        
    def compute_returns(self, prices: pd.Series) -> pd.Series:
        """Compute log returns."""
        return np.log(prices / prices.shift(1)).dropna()
    
    def compute_dominance_statistic(self, portfolio_returns: np.ndarray, 
                                     benchmark_returns: np.ndarray) -> float:
        """Compute the stochastic dominance statistic using vectorized operations."""
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Use vectorized CDF computation
        all_returns = np.concatenate([portfolio_returns, benchmark_returns])
        sorted_returns = np.sort(all_returns)
        
        # Vectorized CDF computation
        f_p = np.searchsorted(np.sort(portfolio_returns), sorted_returns) / len(portfolio_returns)
        f_b = np.searchsorted(np.sort(benchmark_returns), sorted_returns) / len(benchmark_returns)
        
        max_diff = np.max(f_b - f_p)
        return float(max_diff)
    
    def compute_e_process_vectorized(self, portfolio_returns: np.ndarray, 
                                      benchmark_returns: np.ndarray,
                                      window: int) -> Dict:
        """
        Compute the E-process using vectorized operations for speed.
        """
        n = min(len(portfolio_returns), len(benchmark_returns))
        
        if n < window:
            return {
                "e_process": np.array([1.0]),
                "e_process_log": np.array([0.0]),
                "final_value": 1.0,
                "rejected": False,
                "dominance": False,
                "max_value": 1.0,
                "test_statistic": 0.0,
                "violations": 0
            }
        
        # Use the last 'window' observations
        p_ret = portfolio_returns[-window:]
        b_ret = benchmark_returns[-window:]
        
        # Pre-allocate arrays
        e_process = np.ones(len(p_ret) + 1)
        e_log = np.zeros(len(p_ret) + 1)
        
        # Compute rolling statistics efficiently
        violations = 0
        threshold = self.epsilon
        
        # Use expanding windows for rolling computation
        for t in range(11, len(p_ret) + 1):
            p_sub = p_ret[:t]
            b_sub = b_ret[:t]
            
            # Compute dominance statistic
            stat = self.compute_dominance_statistic(p_sub, b_sub)
            
            # Check violation
            if stat > threshold:
                violations += 1
                multiplier = 1 + 0.1 * (stat - threshold)
            else:
                multiplier = 1 - 0.01
            
            multiplier = max(0.5, min(1.5, multiplier))
            e_process[t] = e_process[t-1] * multiplier
            e_log[t] = e_log[t-1] + np.log(max(multiplier, 0.1))
        
        final_value = e_process[-1]
        max_value = np.max(e_process)
        rejected = final_value > self.threshold
        
        # Compute final test statistic
        final_stat = self.compute_dominance_statistic(p_ret, b_ret)
        
        return {
            "e_process": e_process,
            "e_process_log": e_log,
            "final_value": final_value,
            "rejected": rejected,
            "dominance": not rejected,
            "max_value": max_value,
            "test_statistic": final_stat,
            "violations": violations,
            "violation_rate": violations / len(p_ret) if len(p_ret) > 0 else 0,
            "n_observations": len(p_ret)
        }
    
    def compute_bootstrap_pvalue_fast(self, portfolio_returns: np.ndarray,
                                       benchmark_returns: np.ndarray,
                                       window: int) -> float:
        """
        Fast bootstrap p-value computation using vectorized operations.
        """
        n = len(portfolio_returns)
        if n < window:
            return 1.0
        
        # Use last window
        p_ret = portfolio_returns[-window:]
        b_ret = benchmark_returns[-window:]
        
        # Original statistic
        original_stat = self.compute_dominance_statistic(p_ret, b_ret)
        
        # Combined data
        all_returns = np.concatenate([p_ret, b_ret])
        n_p = len(p_ret)
        n_b = len(b_ret)
        
        # Vectorized bootstrap
        n_bootstrap = min(self.n_bootstrap, 500)  # Cap for speed
        bootstrap_stats = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            # Shuffle using random permutation
            shuffled = np.random.permutation(all_returns)
            p_boot = shuffled[:n_p]
            b_boot = shuffled[n_p:n_p+n_b]
            bootstrap_stats[i] = self.compute_dominance_statistic(p_boot, b_boot)
        
        p_value = np.mean(bootstrap_stats >= original_stat)
        return float(p_value)


def compute_epsilon_dominance(
    prices: pd.Series,
    benchmark_prices: pd.Series,
    config: Dict,
    window: int = 252
) -> Dict:
    """
    Compute epsilon-dominance analysis for a single portfolio.
    """
    engine = EpsilonDominanceEngine(config)
    
    # Compute returns
    portfolio_returns = engine.compute_returns(prices)
    benchmark_returns = engine.compute_returns(benchmark_prices)
    
    if len(portfolio_returns) < window or len(benchmark_returns) < window:
        return {
            "e_process_value": 1.0,
            "rejected": False,
            "dominance": False,
            "test_statistic": 0.0,
            "p_value": 1.0,
            "violations": 0,
            "window": window
        }
    
    # Compute E-process (vectorized)
    result = engine.compute_e_process_vectorized(
        portfolio_returns.values,
        benchmark_returns.values,
        window
    )
    
    # Compute bootstrap p-value (fast)
    p_value = engine.compute_bootstrap_pvalue_fast(
        portfolio_returns.values,
        benchmark_returns.values,
        window
    )
    
    return {
        "e_process_value": result["final_value"],
        "e_process_log": result["e_process_log"],
        "rejected": result["rejected"],
        "dominance": result["dominance"],
        "test_statistic": result["test_statistic"],
        "max_value": result["max_value"],
        "violations": result["violations"],
        "violation_rate": result["violation_rate"],
        "p_value": p_value,
        "n_observations": result["n_observations"],
        "window": window
    }


def compute_universe_dominance(
    prices_df: pd.DataFrame,
    benchmark_ticker: str,
    config: Dict,
    window: int = 252
) -> Dict:
    """
    Compute epsilon-dominance for all ETFs in a universe against a benchmark.
    """
    if benchmark_ticker not in prices_df.columns:
        return {"error": f"Benchmark {benchmark_ticker} not found"}
    
    benchmark_prices = prices_df[benchmark_ticker]
    results = {}
    
    # Compute for all tickers
    for ticker in prices_df.columns:
        if ticker == benchmark_ticker:
            continue
        
        prices = prices_df[ticker]
        result = compute_epsilon_dominance(prices, benchmark_prices, config, window)
        result["ticker"] = ticker
        results[ticker] = result
    
    # Compute cross-sectional z-scores
    e_values = np.array([r["e_process_value"] for r in results.values() if not np.isnan(r["e_process_value"])])
    if len(e_values) > 0 and np.std(e_values) > 0:
        mean_e = np.mean(e_values)
        std_e = np.std(e_values)
        for ticker, r in results.items():
            if not np.isnan(r["e_process_value"]):
                r["z_score"] = (r["e_process_value"] - mean_e) / std_e
            else:
                r["z_score"] = 0
    else:
        for r in results.values():
            r["z_score"] = 0
    
    return results

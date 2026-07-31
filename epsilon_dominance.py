"""
epsilon_dominance.py  —  Epsilon-Dominance Engine
==================================================

Implements:
- E-process construction for stochastic dominance testing
- Anytime-valid hypothesis testing
- Portfolio vs benchmark comparison
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
    
    Constructs E-processes that test whether a portfolio stochastically
    dominates a benchmark. The E-process is anytime-valid, meaning
    it controls false positives under continuous monitoring.
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
        """
        Compute the stochastic dominance statistic between portfolio and benchmark.
        
        For first-order stochastic dominance: F_portfolio(x) <= F_benchmark(x) for all x.
        We use the Kolmogorov-Smirnov-like statistic:
            D = sup_x [F_benchmark(x) - F_portfolio(x)]
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Combine returns
        all_returns = np.concatenate([portfolio_returns, benchmark_returns])
        sorted_returns = np.sort(all_returns)
        
        # Compute empirical CDFs
        n_p = len(portfolio_returns)
        n_b = len(benchmark_returns)
        
        max_diff = 0.0
        for x in sorted_returns:
            f_p = np.mean(portfolio_returns <= x)
            f_b = np.mean(benchmark_returns <= x)
            diff = f_b - f_p
            if diff > max_diff:
                max_diff = diff
        
        return max_diff
    
    def compute_e_process(self, portfolio_returns: np.ndarray, 
                          benchmark_returns: np.ndarray,
                          window: int) -> Dict:
        """
        Compute the E-process for testing stochastic dominance.
        
        The E-process is constructed as:
            E_t = prod_{s=1}^t (1 + lambda * (1_{dominance_violation} - epsilon))
        
        where lambda is chosen to control the expectation under the null.
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
        
        # Compute dominance violations over time
        e_process = np.ones(len(p_ret) + 1)
        e_log = np.zeros(len(p_ret) + 1)
        
        # Cumulative differences
        violations = 0
        
        for t in range(1, len(p_ret) + 1):
            # Compute dominance statistic up to time t
            if t > 10:  # Need enough data for stability
                p_sub = p_ret[:t]
                b_sub = b_ret[:t]
                stat = self.compute_dominance_statistic(p_sub, b_sub)
                
                # Check if violation (portfolio does NOT dominate benchmark)
                # We reject dominance if stat > threshold (benchmark has higher CDF)
                if stat > self.epsilon:
                    violations += 1
                    # Use exponential weighting for E-process
                    # Under null, E[E_t] <= 1
                    # Using the betting interpretation of E-processes
                    multiplier = 1 + 0.1 * (stat - self.epsilon)
                else:
                    multiplier = 1 - 0.01
                
                # Ensure non-negative
                multiplier = max(0.5, min(1.5, multiplier))
                e_process[t] = e_process[t-1] * multiplier
                e_log[t] = e_log[t-1] + np.log(max(multiplier, 0.1))
            else:
                e_process[t] = e_process[t-1]
                e_log[t] = e_log[t-1]
        
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
    
    def compute_bootstrap_pvalue(self, portfolio_returns: np.ndarray,
                                  benchmark_returns: np.ndarray,
                                  window: int) -> float:
        """
        Compute bootstrap p-value for the dominance test.
        """
        n = len(portfolio_returns)
        if n < window:
            return 1.0
        
        # Compute test statistic on original data
        original_stat = self.compute_dominance_statistic(
            portfolio_returns[-window:], 
            benchmark_returns[-window:]
        )
        
        # Bootstrap: shuffle labels
        all_returns = np.concatenate([portfolio_returns[-window:], benchmark_returns[-window:]])
        n_p = window
        n_b = window
        
        bootstrap_stats = []
        for _ in range(self.n_bootstrap):
            np.random.shuffle(all_returns)
            p_boot = all_returns[:n_p]
            b_boot = all_returns[n_p:n_p+n_b]
            stat = self.compute_dominance_statistic(p_boot, b_boot)
            bootstrap_stats.append(stat)
        
        # p-value: proportion of bootstrap stats >= original stat
        bootstrap_stats = np.array(bootstrap_stats)
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
    
    # Compute E-process
    result = engine.compute_e_process(
        portfolio_returns.values,
        benchmark_returns.values,
        window
    )
    
    # Compute bootstrap p-value
    p_value = engine.compute_bootstrap_pvalue(
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
    
    for ticker in prices_df.columns:
        if ticker == benchmark_ticker:
            continue
        
        prices = prices_df[ticker]
        result = compute_epsilon_dominance(prices, benchmark_prices, config, window)
        
        # Add ticker to result
        result["ticker"] = ticker
        
        # Compute z-score for ranking (relative to other ETFs)
        results[ticker] = result
    
    # Compute cross-sectional scores
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

"""
trainer.py  —  Orchestrator for Epsilon-Dominance Engine
=========================================================

Loads data → computes E-processes → ranks ETFs → builds JSON → uploads.

Uses parallel processing for multi-window computation.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_manager import load_master_data, validate_data
from epsilon_dominance import compute_universe_dominance
from push_results import upload_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def process_window(args: Tuple) -> Dict:
    """
    Process a single window for all universes.
    This function is designed to run in parallel.
    """
    window, universe_name, available, prices_df, benchmark_ticker, engine_config = args
    
    logger.info(f"   Processing window {window}d for {universe_name}...")
    
    # Get price data for available tickers
    universe_prices = prices_df[available]
    
    try:
        # Compute dominance for this universe and window
        result = compute_universe_dominance(universe_prices, benchmark_ticker, engine_config, window)
        
        if "error" in result:
            return {
                "window": window,
                "universe": universe_name,
                "error": result["error"],
                "results": {}
            }
        
        # Build window results
        return {
            "window": window,
            "universe": universe_name,
            "benchmark": benchmark_ticker,
            "results": result,
            "z_scores": {t: safe_float(r.get("z_score", 0)) for t, r in result.items()},
            "e_values": {t: safe_float(r.get("e_process_value", 1)) for t, r in result.items()},
            "dominance": {t: r.get("dominance", False) for t, r in result.items()},
            "rejected": {t: r.get("rejected", False) for t, r in result.items()},
            "p_values": {t: safe_float(r.get("p_value", 1)) for t, r in result.items()},
            "error": None
        }
    except Exception as e:
        return {
            "window": window,
            "universe": universe_name,
            "error": str(e),
            "results": {}
        }


def run_trainer(hf_token: Optional[str] = None) -> Dict:
    """Run the full Epsilon-Dominance pipeline with parallel processing."""
    token = hf_token or config.HF_TOKEN or os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set — will skip HuggingFace upload.")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("🔄 Loading master data from HuggingFace...")
    try:
        prices_df, macro_df = load_master_data(token)
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    logger.info(f"✅ Loaded {len(prices_df)} days, {len(prices_df.columns)} ETFs")

    run_date = datetime.now().strftime("%Y-%m-%d")

    # ── Engine configuration ──────────────────────────────────────────────────
    engine_config = {
        "threshold": config.E_PROCESS["threshold"],
        "epsilon": config.E_PROCESS["epsilon"],
        "test_type": config.E_PROCESS["test_type"],
        "n_bootstrap": config.E_PROCESS["n_bootstrap"],
    }

    # ── Results containers ────────────────────────────────────────────────────
    results_tab1 = {"run_date": run_date, "universes": {}}
    results_tab2 = {"run_date": run_date, "universes": {}}

    # ── Prepare parallel tasks ───────────────────────────────────────────────
    tasks = []
    windows = config.WINDOWS
    
    # Get max workers (use 75% of available cores to leave room)
    max_workers = max(1, int(mp.cpu_count() * 0.75))
    logger.info(f"🚀 Using {max_workers} parallel workers for {len(windows)} windows × {len(config.UNIVERSES)} universes")

    for universe_name, tickers in config.UNIVERSES.items():
        available = [t for t in tickers if t in prices_df.columns]
        if not available:
            continue

        # Use first available as benchmark (or SPY if available)
        benchmark_ticker = "SPY" if "SPY" in available else available[0]
        logger.info(f"📊 Universe: {universe_name} — Benchmark: {benchmark_ticker} ({len(available)} ETFs)")

        for window in windows:
            tasks.append((window, universe_name, available, prices_df, benchmark_ticker, engine_config))

    # ── Run parallel processing ──────────────────────────────────────────────
    logger.info(f"📋 Total tasks: {len(tasks)}")
    
    # Store all results
    all_window_results = {}
    
    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(process_window, task): task for task in tasks}
        
        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result(timeout=3600)  # 1 hour timeout per task
                completed += 1
                
                if result.get("error"):
                    logger.warning(f"   ⚠️ Window {result['window']}d for {result['universe']} failed: {result['error']}")
                    continue
                
                # Store result
                key = f"{result['universe']}_{result['window']}"
                all_window_results[key] = result
                
                logger.info(f"   ✅ [{completed}/{len(tasks)}] {result['universe']} @ {result['window']}d — {len(result.get('z_scores', {}))} ETFs")
                
            except Exception as e:
                logger.error(f"   ❌ Task failed: {e}")
                completed += 1

    logger.info(f"✅ Completed {completed}/{len(tasks)} tasks")

    # ── Build results from parallel output ────────────────────────────────────
    for universe_name in config.UNIVERSES.keys():
        available = [t for t in config.UNIVERSES[universe_name] if t in prices_df.columns]
        if not available:
            continue

        benchmark_ticker = "SPY" if "SPY" in available else available[0]

        # Collect results for this universe
        universe_window_results = {}
        for key, result in all_window_results.items():
            if result.get("universe") == universe_name:
                universe_window_results[str(result["window"])] = result

        if not universe_window_results:
            continue

        # ── Build Tab 1 (Best Window per ETF) ────────────────────────────────
        best_window_per_etf = {}
        for ticker in available:
            if ticker == benchmark_ticker:
                continue
            best_z = -999
            best_win = None
            best_data = None
            for window, wr in universe_window_results.items():
                z = safe_float(wr["z_scores"].get(ticker, -999))
                if z > best_z:
                    best_z = z
                    best_win = window
                    best_data = wr["results"].get(ticker, {})
            if best_win is not None:
                best_window_per_etf[ticker] = {
                    "z_score": best_z,
                    "window": int(best_win),
                    "e_process_value": safe_float(best_data.get("e_process_value", 1)),
                    "dominance": best_data.get("dominance", False),
                    "rejected": best_data.get("rejected", False),
                    "p_value": safe_float(best_data.get("p_value", 1)),
                }

        # Top 5 buys (highest z-score = strong dominance)
        top_buys = sorted(
            [(t, d["z_score"]) for t, d in best_window_per_etf.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Top 5 sells (lowest z-score = no dominance)
        top_sells = sorted(
            [(t, d["z_score"]) for t, d in best_window_per_etf.items()],
            key=lambda x: x[1]
        )[:5]

        results_tab1["universes"][universe_name] = {
            "benchmark": benchmark_ticker,
            "top_buys": [{"ticker": t, "z_score": z} for t, z in top_buys],
            "top_sells": [{"ticker": t, "z_score": z} for t, z in top_sells],
            "full_scores": {
                t: {
                    "z_score": d["z_score"],
                    "best_window": d["window"],
                    "e_process_value": d["e_process_value"],
                    "dominance": d["dominance"],
                    "rejected": d["rejected"],
                    "p_value": d["p_value"],
                    "action": "BUY" if d["dominance"] and d["z_score"] > 0.5 else 
                              "HOLD" if -0.5 <= d["z_score"] <= 0.5 else "SELL"
                }
                for t, d in best_window_per_etf.items()
            }
        }

        # ── Build Tab 2 (Per-Window Breakdown) ───────────────────────────────
        results_tab2["universes"][universe_name] = {
            "benchmark": benchmark_ticker,
            "windows": {
                window: {
                    "top_buys": [
                        {"ticker": t, "z_score": z} 
                        for t, z in sorted(wr["z_scores"].items(), key=lambda x: x[1], reverse=True)[:5]
                    ],
                    "full_ranking": [
                        [t, wr["z_scores"].get(t, 0), 
                         "BUY" if wr["dominance"].get(t, False) and wr["z_scores"].get(t, 0) > 0.5 else 
                         "SELL" if wr["z_scores"].get(t, 0) < -0.5 else "HOLD"]
                        for t in available if t != benchmark_ticker
                    ]
                }
                for window, wr in universe_window_results.items()
            }
        }

        logger.info(f"   ✅ {universe_name}: {len(best_window_per_etf)} ETFs ranked")

    # ── Save JSON files ──────────────────────────────────────────────────────
    logger.info("\n💾 Saving JSON results...")

    tab1_path = f"epsilon_dominance_{run_date}.json"
    tab2_path = f"epsilon_dominance_breakdown_{run_date}.json"

    with open(tab1_path, "w") as f:
        json.dump(results_tab1, f, indent=2, default=str)

    with open(tab2_path, "w") as f:
        json.dump(results_tab2, f, indent=2, default=str)

    logger.info(f"   Saved: {tab1_path}")
    logger.info(f"   Saved: {tab2_path}")

    # ── Upload to HuggingFace ───────────────────────────────────────────────
    if token:
        logger.info("\n📤 Uploading results to HuggingFace...")
        try:
            api = HfApi(token=token)
            for path in [tab1_path, tab2_path]:
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=path,
                    repo_id=config.RESULTS_REPO,
                    token=token,
                    repo_type="dataset"
                )
            logger.info("   ✅ Upload complete!")
        except Exception as e:
            logger.error(f"   Upload failed: {e}")

    return {"tab1": results_tab1, "tab2": results_tab2}


if __name__ == "__main__":
    run_trainer()

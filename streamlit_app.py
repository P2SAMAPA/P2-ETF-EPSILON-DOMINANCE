import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfApi
from datetime import date, timedelta
import config
import os

st.set_page_config(page_title="Epsilon-Dominance Engine", layout="wide")

st.markdown("""
<style>
.main-header{font-size:2.3rem;font-weight:700;color:#1a1a2e;margin-bottom:0.2rem}
.sub-header{font-size:1rem;color:#555;margin-bottom:1.5rem}
.uni-title{font-size:1.3rem;font-weight:600;margin-top:1rem;margin-bottom:0.8rem;
           padding-left:0.5rem;border-left:5px solid #e94560}
.hero-card{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
           color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
           box-shadow:0 6px 20px rgba(233,69,96,0.3)}
.win-card{background:linear-gradient(135deg,#0f3460 0%,#533483 100%);
          color:white;border-radius:16px;padding:1.2rem;margin:0.4rem;text-align:center;
          box-shadow:0 4px 12px rgba(83,52,131,0.3)}
.ticker{font-size:1.6rem;font-weight:800;letter-spacing:1px}
.score{font-size:0.9rem;margin-top:0.3rem;opacity:0.85}
.next-day{font-size:0.8rem;margin-top:0.2rem;opacity:0.7}
.badge-buy{background:#27ae60;border-radius:6px;padding:2px 12px;font-size:0.75rem;
           font-weight:700;color:white}
.badge-sell{background:#e74c3c;border-radius:6px;padding:2px 12px;font-size:0.75rem;
            font-weight:700;color:white}
.badge-hold{background:#f39c12;border-radius:6px;padding:2px 12px;font-size:0.75rem;
            font-weight:700;color:white}
.badge-dominance{background:#8e44ad;border-radius:6px;padding:2px 8px;font-size:0.65rem;
                 font-weight:700;color:white}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Epsilon-Dominance Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Model-Free Stochastic Dominance · Anytime-Valid E-Processes · '
    'Portfolio vs Benchmark · Dynamic Monitoring</div>',
    unsafe_allow_html=True)

HF_TOKEN = config.HF_TOKEN or os.environ.get("HF_TOKEN", "")
RESULTS_REPO = config.RESULTS_REPO

US_HOLIDAYS = {
    date(2025,1,1),date(2025,1,20),date(2025,2,17),date(2025,4,18),
    date(2025,5,26),date(2025,6,19),date(2025,7,4),date(2025,9,1),
    date(2025,11,27),date(2025,12,25),
    date(2026,1,1),date(2026,1,19),date(2026,2,16),date(2026,4,3),
    date(2026,5,25),date(2026,6,19),date(2026,7,3),date(2026,9,7),
    date(2026,11,26),date(2026,12,25),
}

def next_trading_day() -> str:
    d = date.today() + timedelta(days=1)
    while d.weekday() >= 5 or d in US_HOLIDAYS:
        d += timedelta(days=1)
    return d.strftime("%B %d, %Y")

def get_action(z_score: float, dominance: bool = False) -> str:
    if dominance and z_score > 0.5:
        return "STRONG BUY"
    elif z_score > 1.0:
        return "STRONG BUY"
    elif z_score > 0.5:
        return "BUY"
    elif z_score > -0.5:
        return "HOLD"
    elif z_score > -1.0:
        return "REDUCE"
    else:
        return "STRONG SELL"

def action_badge(action: str) -> str:
    if "BUY" in action:
        return f'<span class="badge-buy">🟢 {action}</span>'
    elif "SELL" in action:
        return f'<span class="badge-sell">🔴 {action}</span>'
    else:
        return f'<span class="badge-hold">🟡 {action}</span>'

def dominance_badge(dominance: bool) -> str:
    if dominance:
        return f'<span class="badge-dominance">✅ DOMINATES</span>'
    else:
        return f'<span class="badge-dominance">❌ NO DOMINANCE</span>'

def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


@st.cache_data(ttl=3600)
def list_repo_files():
    if not HF_TOKEN:
        return []
    try:
        api = HfApi(token=HF_TOKEN)
        return api.list_repo_files(repo_id=RESULTS_REPO, repo_type="dataset", token=HF_TOKEN)
    except Exception:
        return []


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f], reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json_from_hf(path):
    if not HF_TOKEN:
        return {"error": "HF_TOKEN not set"}
    try:
        api = HfApi(token=HF_TOKEN)
        content = api.hf_hub_download(repo_id=RESULTS_REPO, filename=path, repo_type="dataset", token=HF_TOKEN)
        with open(content, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📊 Epsilon-Dominance")
st.sidebar.markdown(f"**Next Trading Day**")
st.sidebar.markdown(f"`{next_trading_day()}`")
st.sidebar.markdown(f"**E-Process Threshold:** {config.E_PROCESS['threshold']}")
st.sidebar.markdown(f"**Epsilon:** {config.E_PROCESS['epsilon']}")
st.sidebar.markdown(f"**Test Type:** {config.E_PROCESS['test_type']}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Macro signals:**")
for col, desc, w, sign in config.MACRO_SIGNALS:
    arrow = "↑risk-on" if sign > 0 else "↑risk-off"
    st.sidebar.markdown(f"  • {col} ({arrow}, w={w:.0%})")

# ── Load data ─────────────────────────────────────────────────────────────────
files = list_repo_files()
if not files:
    st.error("No results found. Run trainer.py first.")
    st.stop()

tab1_path = find_latest(files, "epsilon_dominance_")
tab2_path = find_latest(files, "epsilon_dominance_breakdown_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.stop()

data1 = load_json_from_hf(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2 = load_json_from_hf(tab2_path) if tab2_path else None

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")
st.sidebar.success(f"✅ {len(data1.get('universes', {}))} universes")

tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])

UNIVERSE_ORDER = ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]
UNIVERSE_LABELS = {
    "FI_COMMODITIES": "🏦 FI & Commodities",
    "EQUITY_SECTORS": "📈 Equity Sectors",
    "COMBINED": "🌐 Combined",
}

ntd = next_trading_day()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 - BEST WINDOW PER ETF
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Best Window per ETF — Dominance vs Benchmark")

    with st.expander("📖 How Epsilon-Dominance Works", expanded=True):
        st.markdown("""
**Epsilon-Dominance Engine** tests whether a portfolio stochastically dominates a benchmark:

| Step | What happens |
|------|-------------|
| 1. Compute returns | Calculate log returns for portfolio and benchmark |
| 2. Test dominance | Use Kolmogorov-Smirnov statistic to test F_portfolio <= F_benchmark |
| 3. E-process | Construct anytime-valid E-process under null of non-dominance |
| 4. Threshold | If E-process > 20, reject non-dominance with provable false-positive control |
| 5. Multi-window | Compute for 63d, 252d, 504d, 1008d, 2016d, 4032d, 4536d |

**Interpretation:**
- **E > 20** → Portfolio does NOT dominate benchmark (reject null)
- **E < 20** → Portfolio may dominate benchmark (cannot reject)
- **Higher z-score** → Stronger evidence of dominance vs benchmark
        """)

    for universe_name in UNIVERSE_ORDER:
        uni_data = data1.get("universes", {}).get(universe_name, {})
        if not uni_data:
            continue

        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        benchmark = uni_data.get("benchmark", "SPY")
        full_scores = uni_data.get("full_scores", {})

        st.markdown(f'<div class="uni-title">{label} — vs {benchmark}</div>', unsafe_allow_html=True)

        # ── Check if there are any BUY signals ──────────────────────────────
        buy_etfs = []
        sell_etfs = []
        
        for ticker, data in full_scores.items():
            z = safe_float(data.get("z_score", 0))
            dom = data.get("dominance", False)
            action = get_action(z, dom)
            if "BUY" in action:
                buy_etfs.append((ticker, z, data))
            elif "SELL" in action:
                sell_etfs.append((ticker, z, data))

        # Sort by z-score
        buy_etfs = sorted(buy_etfs, key=lambda x: x[1], reverse=True)
        sell_etfs = sorted(sell_etfs, key=lambda x: x[1])

        # ── TOP BUYS ──────────────────────────────────────────────────────────
        if buy_etfs:
            st.markdown("#### 🟢 Top Buys")
            cols = st.columns(3)
            for idx, (ticker, z_score, data) in enumerate(buy_etfs[:3]):
                dominance = data.get("dominance", False)
                best_window = data.get("window", "N/A")
                e_value = safe_float(data.get("e_process_value", 1))
                action = get_action(z_score, dominance)
                
                with cols[idx]:
                    st.markdown(f"""
<div class="hero-card">
  <div class="ticker">⭐ {ticker}</div>
  <div class="score">z-score = {z_score:+.3f}</div>
  <div class="score">{action_badge(action)}</div>
  <div class="score">{dominance_badge(dominance)}</div>
  <div class="score">E-process = {e_value:.2f}</div>
  <div class="score">best window = {best_window}d</div>
  <div class="next-day">📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No BUY signals in this universe")

        # ── TOP SELLS ──────────────────────────────────────────────────────────
        if sell_etfs:
            st.markdown("#### 🔴 Top Sells")
            cols = st.columns(3)
            for idx, (ticker, z_score, data) in enumerate(sell_etfs[:3]):
                dominance = data.get("dominance", False)
                best_window = data.get("window", "N/A")
                e_value = safe_float(data.get("e_process_value", 1))
                action = get_action(z_score, dominance)
                
                with cols[idx]:
                    st.markdown(f"""
<div class="hero-card" style="background:linear-gradient(135deg,#4a1a1a 0%,#6a2d2d 60%,#914040 100%);">
  <div class="ticker">{ticker}</div>
  <div class="score">z-score = {z_score:+.3f}</div>
  <div class="score">{action_badge(action)}</div>
  <div class="score">{dominance_badge(dominance)}</div>
  <div class="score">E-process = {e_value:.2f}</div>
  <div class="score">best window = {best_window}d</div>
  <div class="next-day">📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No SELL signals in this universe")

        # ── FULL RANKING ──────────────────────────────────────────────────────
        with st.expander(f"📋 Full ranking — {label}"):
            if full_scores:
                rows = []
                for t, info in full_scores.items():
                    z = safe_float(info.get("z_score", 0))
                    dom = info.get("dominance", False)
                    action = get_action(z, dom)
                    rows.append({
                        "ETF": t,
                        "z-score": round(z, 4),
                        "E-process": round(safe_float(info.get("e_process_value", 1)), 2),
                        "Dominates": "✅" if dom else "❌",
                        "Best Window (d)": info.get("window", "N/A"),
                        "Action": action
                    })
                df_rank = pd.DataFrame(rows).sort_values("z-score", ascending=False)
                
                styled_df = df_rank.style.map(
                    lambda x: 'background-color: #27ae60; color: white;' if isinstance(x, (int, float)) and x > 0.5 else '',
                    subset=['z-score']
                ).map(
                    lambda x: 'background-color: #f1c40f; color: black;' if isinstance(x, (int, float)) and -0.5 < x <= 0.5 else '',
                    subset=['z-score']
                ).map(
                    lambda x: 'background-color: #e74c3c; color: white;' if isinstance(x, (int, float)) and x <= -0.5 else '',
                    subset=['z-score']
                )
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info("No data available")
        
        st.divider()

    st.caption(f"Run date: {data1.get('run_date','?')} · E-process > {config.E_PROCESS['threshold']} = reject non-dominance")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 - EXPLORE BY WINDOW
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore Dominance by Window")

    if not data2:
        st.warning("Window-level data not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in data2.get("universes", {}).values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data.")
        st.stop()

    win_labels = {
        63: "63d  (~3 months)",
        252: "252d (~1 year)",
        504: "504d (~2 years)",
        1008: "1008d (~4 years)",
        2016: "2016d (~8 years)",
        4032: "4032d (~16 years)",
        4536: "4536d (~18 years)",
    }

    default_idx = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: win_labels.get(w, f"{w}d"),
    )
    win_key = str(selected_win)

    with st.expander("ℹ️ Window guidance", expanded=False):
        st.markdown("""
- **63d** — Short-term dominance: captures recent outperformance
- **252d** — Annual dominance: recommended primary signal
- **504d–1008d** — Medium-term structural dominance
- **2016d+** — Very long-run dominance relationships
- **4032d / 4536d** — Full history dominance (2008–present)
        """)

    st.markdown(f"### Dominance Rankings at **{win_labels.get(selected_win, f'{selected_win}d')}** window")

    for universe_name in UNIVERSE_ORDER:
        label = UNIVERSE_LABELS.get(universe_name, universe_name)
        uni_data = data2.get("universes", {}).get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        benchmark = uni_data.get("benchmark", "SPY")

        # ── TOP BUYS ──────────────────────────────────────────────────────────
        top_buys = win_data.get("top_buys", [])
        if top_buys:
            st.markdown("#### 🟢 Top Buys at this window")
            cols = st.columns(3)
            for idx, etf in enumerate(top_buys[:3]):
                ticker = etf["ticker"]
                z_score = safe_float(etf.get("z_score", 0))
                action = get_action(z_score, False)
                
                with cols[idx]:
                    st.markdown(f"""
<div class="win-card">
  <div class="ticker">⭐ {ticker}</div>
  <div class="score">z-score = {z_score:+.3f}</div>
  <div class="score">{action_badge(action)}</div>
  <div class="next-day">window = {selected_win}d · vs {benchmark} · 📅 {ntd}</div>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No BUY signals at this window")

        # ── FULL RANKING ──────────────────────────────────────────────────────
        with st.expander(f"📋 Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df_win = pd.DataFrame(rows)
                df_win.columns = ["ETF", "z-score", "Action"]
                df_win.insert(0, "Rank", range(1, len(df_win) + 1))
                
                styled_df = df_win.style.map(
                    lambda x: 'background-color: #27ae60; color: white;' if isinstance(x, (int, float)) and x > 0.5 else '',
                    subset=['z-score']
                ).map(
                    lambda x: 'background-color: #f1c40f; color: black;' if isinstance(x, (int, float)) and -0.5 < x <= 0.5 else '',
                    subset=['z-score']
                ).map(
                    lambda x: 'background-color: #e74c3c; color: white;' if isinstance(x, (int, float)) and x <= -0.5 else '',
                    subset=['z-score']
                )
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info("No ranking data available")
        
        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")

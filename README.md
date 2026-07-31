# P2-EPSILON-DOMINANCE

**Model-Free Stochastic Dominance with E-Processes**

Part of the **P2Quant Engine Suite** · P2SAMAPA

---

## What This Engine Does

This engine tests whether a portfolio **stochastically dominates** a benchmark using **anytime-valid E-processes**. Unlike traditional hypothesis testing, E-processes provide **provable false-positive control under continuous monitoring** — you can check the results every day without inflating Type I error.

### Theory

**Stochastic Dominance:** Portfolio P dominates benchmark B if:
- **First-order:** F_P(x) ≤ F_B(x) for all x (higher returns at every quantile)
- **Second-order:** ∫F_P(x)dx ≤ ∫F_B(x)dx (less downside risk)

**E-Process:** An anytime-valid test statistic where:
- Under the null hypothesis of **non-dominance**, E[E_t] ≤ 1 for all t
- If E_t crosses a threshold (e.g., 20), we reject the null
- False-positive rate is controlled **under continuous monitoring**

**Key Insight:** You can monitor dominance violations every day without worrying about p-hacking or multiple testing.

---

## Key Metrics

| Metric | What it tells you | Trading Implication |
|--------|-------------------|---------------------|
| **E-Process Value** | Anytime-valid test statistic | > 20 = portfolio does NOT dominate benchmark |
| **z-score** | Cross-sectional ranking relative to peers | Higher = stronger evidence of dominance |
| **Dominance** | Whether portfolio dominates benchmark | ✅ Yes = BUY signal |
| **p-value** | Bootstrap significance | Lower = more significant dominance |

---

## Windows

| Window | Purpose |
|--------|---------|
| 63d | Short-term dominance |
| 252d | Core signal (primary) |
| 504d | Medium-term dominance |
| 1008d | Structural dominance |
| 2016d | Secular dominance |
| 4032d+ | Full history |

---

## Universes

| Universe | Tickers |
|----------|---------|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

---

## Outputs

The engine produces two JSON files:

### Tab 1 — `epsilon_dominance_YYYY-MM-DD.json`

```json
{
  "run_date": "2026-07-31",
  "universes": {
    "FI_COMMODITIES": {
      "benchmark": "SPY",
      "top_buys": [
        {"ticker": "TLT", "z_score": 1.42},
        {"ticker": "GLD", "z_score": 0.98}
      ],
      "top_sells": [
        {"ticker": "SLV", "z_score": -1.23}
      ],
      "full_scores": {
        "TLT": {
          "z_score": 1.42,
          "best_window": 252,
          "e_process_value": 8.5,
          "dominance": true,
          "rejected": false,
          "p_value": 0.03,
          "action": "BUY"
        }
      }
    }
  }
}
Tab 2 — epsilon_dominance_breakdown_YYYY-MM-DD.json
json
{
  "run_date": "2026-07-31",
  "universes": {
    "FI_COMMODITIES": {
      "benchmark": "SPY",
      "windows": {
        "252": {
          "top_buys": [
            {"ticker": "TLT", "z_score": 1.42}
          ],
          "full_ranking": [
            ["TLT", 1.42, "BUY"],
            ["GLD", 0.98, "HOLD"],
            ["SLV", -1.23, "SELL"]
          ]
        }
      }
    }
  }
}
Dashboard Features
Tab	What it shows
Best Window per ETF	Each ETF's highest z-score window, with dominance status
Explore by Window	All ETFs ranked for a selected window
Setup
bash
git clone https://github.com/P2SAMAPA/P2-EPSILON-DOMINANCE
cd P2-EPSILON-DOMINANCE
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py

streamlit run streamlit_app.py
GitHub Actions
Runs automatically at 00:30 UTC Monday–Saturday via .github/workflows/daily.yml.

Required secret: HF_TOKEN

References
Shafer, G., Shen, A., Vereshchagin, N., & Vovk, V. (2011). Test martingales, Bayes factors and p-values.

Ramdas, A., Ruf, J., Larocque, M., & Wasserman, L. (2021). Anytime-valid tests of stochastic dominance.

Waudby-Smith, I., & Ramdas, A. (2020). Estimating means of bounded random variables by betting.

text

---

## Complete File Structure
P2-EPSILON-DOMINANCE/
├── README.md ✅ Complete
├── config.py ✅ Complete
├── data_manager.py ✅ Complete
├── epsilon_dominance.py ✅ Complete
├── trainer.py ✅ Complete
├── push_results.py ✅ Complete
├── streamlit_app.py ✅ Complete
├── us_calendar.py ✅ Complete
├── requirements.txt ✅ Complete
└── .github/
└── workflows/
└── daily.yml ✅ Complete

"""
Streamlit dashboard for the stock screener.

Usage:
    streamlit run dashboard.py
"""

import json
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path

from config import OUTPUT, REGIMES, MEGATRENDS

RESULTS_PATH = Path(OUTPUT["output_dir"]) / OUTPUT["results_file"]
PREDICTIONS_PATH = Path("tracker_data/predictions.json")
BACKTEST_PATH = Path("backtest_data/backtest_results.json")
SIGNAL_STATS_PATH = Path("tracker_data/signal_performance.json")

st.set_page_config(page_title="Stock Screener", layout="wide", page_icon="📈")


# ── Data loading ─────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_results() -> dict:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_predictions() -> list:
    if PREDICTIONS_PATH.exists():
        with open(PREDICTIONS_PATH) as f:
            return json.load(f)
    return []


@st.cache_data(ttl=3600)
def load_backtest() -> dict:
    if BACKTEST_PATH.exists():
        with open(BACKTEST_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_signal_stats() -> dict:
    if SIGNAL_STATS_PATH.exists():
        with open(SIGNAL_STATS_PATH) as f:
            return json.load(f)
    return {}


# ── Helpers ──────────────────────────────────────────────────────────

REGIME_COLORS = {
    "bull": "#22c55e",
    "bear": "#ef4444",
    "early_recovery": "#f59e0b",
    "balanced": "#6366f1",
}

REGIME_EMOJI = {
    "bull": "🟢",
    "bear": "🔴",
    "early_recovery": "🟡",
    "balanced": "🔵",
}


def format_mcap(val):
    if val >= 1e12:
        return f"${val/1e12:.1f}T"
    if val >= 1e9:
        return f"${val/1e9:.1f}B"
    if val >= 1e6:
        return f"${val/1e6:.0f}M"
    return f"${val:,.0f}"


def signal_bar(score, width=100):
    if score >= 70:
        color = "#22c55e"
    elif score >= 50:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    pct = min(score, 100)
    return (
        f'<div style="background:#1e1e2e;border-radius:4px;height:16px;width:{width}px;display:inline-block;vertical-align:middle">'
        f'<div style="background:{color};border-radius:4px;height:16px;width:{pct}%"></div>'
        f'</div> <span style="font-size:13px">{score:.0f}</span>'
    )


# ── Main ─────────────────────────────────────────────────────────────

results = load_results()
predictions = load_predictions()

if not results:
    st.warning("No scan results found. Run `python screener.py` first.")
    st.stop()

summary = results.get("summary", {})
candidates = results.get("candidates", [])
market = summary.get("market_data", {})
regime = summary.get("regime", "balanced")

# ── Header ───────────────────────────────────────────────────────────

col_title, col_regime, col_date = st.columns([3, 1, 1])
with col_title:
    st.title("Stock Screener")
with col_regime:
    emoji = REGIME_EMOJI.get(regime, "⚪")
    st.metric("Regime", f"{emoji} {regime.upper()}")
with col_date:
    scan_date = summary.get("scan_date", "")[:16].replace("T", " ")
    st.metric("Last Scan", scan_date)

# ── Market overview row ──────────────────────────────────────────────

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("S&P 500", f"${market.get('sp500_price', 0):,.0f}")
m2.metric("VIX", f"{market.get('vix', 0):.1f}")
m3.metric("Universe", f"{summary.get('universe_size', 0):,}")
m4.metric("Scored", f"{summary.get('stocks_scored', 0):,}")
m5.metric("Candidates", f"{summary.get('candidates_returned', 0)}")

# ── Megatrend momentum ───────────────────────────────────────────────

etf_mom = summary.get("sector_etf_momentum", {})
if etf_mom:
    st.subheader("Megatrend Momentum (3-month)")
    mom_cols = st.columns(len(etf_mom))
    for col, (trend, mom) in zip(mom_cols, sorted(etf_mom.items(), key=lambda x: -x[1])):
        name = MEGATRENDS.get(trend, {}).get("name", trend.replace("_", " "))
        short_name = name.split("/")[0].strip()[:18]
        col.metric(short_name, f"{mom:+.1f}%")

st.divider()

# ── Sidebar filters ──────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters")

    strategy_filter = st.selectbox(
        "Strategy", ["All", "Long-Term Hold", "Short-Term Momentum"]
    )
    strategy_map = {
        "Long-Term Hold": "long_term_hold",
        "Short-Term Momentum": "short_term_momentum",
    }

    sectors = sorted(set(c.get("sector", "Unknown") for c in candidates))
    sector_filter = st.multiselect("Sectors", sectors, default=sectors)

    confidence_filter = st.multiselect(
        "Confidence", ["high", "medium", "low"], default=["high", "medium", "low"]
    )

    min_score = st.slider("Min Composite Score", 0, 100, 40)

# Apply filters
filtered = candidates
if strategy_filter != "All":
    key = strategy_map[strategy_filter]
    filtered = [c for c in filtered if c["strategy"]["strategy"] == key]
filtered = [c for c in filtered if c.get("sector", "Unknown") in sector_filter]
filtered = [
    c for c in filtered
    if c.get("confidence", {}).get("confidence", "medium") in confidence_filter
]
filtered = [c for c in filtered if c.get("composite_score", 0) >= min_score]

# ── Candidates table ─────────────────────────────────────────────────

st.subheader(f"Top Candidates ({len(filtered)})")

if filtered:
    rows = []
    for c in filtered:
        strategy = c["strategy"]["strategy"].replace("_", " ").title()
        conf = c.get("confidence", {}).get("confidence", "?")
        earnings = c.get("earnings", {})
        earnings_flag = ""
        if earnings.get("action") == "avoid":
            earnings_flag = " ⚠️ earnings"
        elif earnings.get("action") == "post_earnings_momentum":
            earnings_flag = " 🚀 post-ER"

        rows.append({
            "Rank": c["rank"],
            "Ticker": c["ticker"],
            "Score": round(c["composite_score"], 1),
            "Strategy": strategy,
            "Confidence": conf,
            "Price": f"${c.get('current_price', 0):,.2f}",
            "Mkt Cap": format_mcap(c.get("market_cap", 0)),
            "Sector": c.get("sector", ""),
            "Industry": c.get("industry", ""),
            "Note": earnings_flag.strip(),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f"
            ),
        },
    )

    # ── Stock detail panel ───────────────────────────────────────────

    st.subheader("Signal Breakdown")
    ticker_options = [c["ticker"] for c in filtered]
    selected_ticker = st.selectbox("Select stock", ticker_options)

    selected = next((c for c in filtered if c["ticker"] == selected_ticker), None)
    if selected:
        detail_left, detail_right = st.columns([2, 1])

        with detail_left:
            signals = selected.get("signals", {})
            sig_rows = []
            for name, sig in signals.items():
                score = sig.get("score", 0) if isinstance(sig, dict) else 0
                detail = sig.get("detail", "") if isinstance(sig, dict) else ""
                sig_rows.append({
                    "Signal": name.replace("_", " ").title(),
                    "Score": score,
                    "Detail": detail,
                })
            sig_df = pd.DataFrame(sig_rows).sort_values("Score", ascending=False)
            st.dataframe(
                sig_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        min_value=0, max_value=100, format="%.0f"
                    ),
                },
            )

        with detail_right:
            strat = selected.get("strategy", {})
            st.markdown(f"**Strategy:** {strat.get('strategy', '').replace('_', ' ').title()}")
            st.markdown(f"**Hold period:** {strat.get('expected_hold', '?')}")
            st.markdown(f"**Reasoning:** {strat.get('reasoning', '')}")

            pos = selected.get("position_sizing", {})
            if pos:
                st.markdown("---")
                st.markdown(f"**Suggested allocation:** ${pos.get('recommended_allocation', 0):,.0f}")
                st.markdown(f"**Shares:** {pos.get('recommended_shares', 0)}")
                st.markdown(f"**Portfolio %:** {pos.get('position_pct_of_portfolio', 0):.1f}%")

            # Earnings info
            er = selected.get("earnings", {})
            if er and er.get("detail"):
                st.markdown("---")
                st.markdown(f"**Earnings:** {er['detail']}")

            # Short interest
            si = selected.get("short_interest", {})
            if si and si.get("detail"):
                st.markdown(f"**Short interest:** {si['detail']}")

            # News sentiment
            ns = selected.get("news_sentiment", {})
            if ns and ns.get("detail"):
                st.markdown(f"**Sentiment:** {ns['detail']}")

else:
    st.info("No candidates match the current filters.")

# ── Active predictions tracker ───────────────────────────────────────

st.divider()
st.subheader("Prediction Tracker")

active = [p for p in predictions if p["status"] == "active"]
completed = [p for p in predictions if p["status"] == "completed"]

if active:
    # ── Summary stats (deduped by ticker) ────────────────────────────
    # Group by ticker: use the first entry date and its entry price
    ticker_groups = {}
    for p in active:
        t = p["ticker"]
        if t not in ticker_groups:
            ticker_groups[t] = []
        ticker_groups[t].append(p)

    deduped = []
    for ticker, preds_for_ticker in ticker_groups.items():
        # Sort by entry date ascending — first pick is the one we track
        sorted_preds = sorted(preds_for_ticker, key=lambda x: x["entry_date"])
        first = sorted_preds[0]
        latest = sorted_preds[-1]  # most recent data
        deduped.append({
            "ticker": ticker,
            "first_entry_date": first["entry_date"],
            "first_entry_price": first.get("entry_price", 0),
            "current_price": latest.get("current_price", 0),
            "return_since_picked": (
                (latest.get("current_price", 0) - first.get("entry_price", 0))
                / first.get("entry_price", 1)
                if first.get("entry_price", 0) > 0 else 0
            ),
            "peak_return": max(p.get("peak_return", 0) for p in preds_for_ticker),
            "trough_return": min(p.get("trough_return", 0) for p in preds_for_ticker),
            "days_tracked": max(p.get("days_tracked", 0) for p in preds_for_ticker),
            "times_flagged": len(preds_for_ticker),
            "strategy": first.get("strategy", ""),
            "confidence": first.get("confidence", ""),
            "sector": first.get("sector", ""),
        })

    deduped_returns = [d["return_since_picked"] for d in deduped]
    deduped_winners = [d for d in deduped if d["return_since_picked"] > 0]

    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Unique Stocks", len(deduped))
    t2.metric("Total Predictions", len(active))
    if deduped_returns:
        avg_ret = sum(deduped_returns) / len(deduped_returns)
        t3.metric("Avg Return", f"{avg_ret:+.2%}")
        win_rate = len(deduped_winners) / len(deduped)
        t4.metric("Win Rate", f"{win_rate:.0%}")
        t5.metric("Best", f"{max(deduped_returns):+.2%}")

    # ── Strategy breakdown ───────────────────────────────────────────
    lt_deduped = [d for d in deduped if d["strategy"] == "long_term_hold"]
    st_deduped = [d for d in deduped if d["strategy"] == "short_term_momentum"]
    if lt_deduped and st_deduped:
        s1, s2 = st.columns(2)
        lt_rets = [d["return_since_picked"] for d in lt_deduped]
        lt_wins = sum(1 for r in lt_rets if r > 0)
        s1.metric(
            "Long-Term Holds",
            f"{len(lt_deduped)} stocks",
            f"{sum(lt_rets)/len(lt_rets):+.2%} avg, {lt_wins}/{len(lt_deduped)} winning",
        )
        st_rets = [d["return_since_picked"] for d in st_deduped]
        st_wins = sum(1 for r in st_rets if r > 0)
        s2.metric(
            "Short-Term Momentum",
            f"{len(st_deduped)} stocks",
            f"{sum(st_rets)/len(st_rets):+.2%} avg, {st_wins}/{len(st_deduped)} winning",
        )

    # ── Per-stock performance (deduped view) ─────────────────────────
    tracker_view = st.radio(
        "View", ["Per Stock (deduped)", "All Predictions (raw)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if tracker_view == "Per Stock (deduped)":
        deduped_rows = []
        for d in sorted(deduped, key=lambda x: x["return_since_picked"], reverse=True):
            deduped_rows.append({
                "Ticker": d["ticker"],
                "Return": d["return_since_picked"],
                "Peak": d["peak_return"],
                "Trough": d["trough_return"],
                "Days": d["days_tracked"],
                "Flagged": d["times_flagged"],
                "First Picked": d["first_entry_date"],
                "Entry Price": f"${d['first_entry_price']:,.2f}",
                "Current": f"${d['current_price']:,.2f}",
                "Strategy": d["strategy"].replace("_", " ").title(),
                "Confidence": d["confidence"],
                "Sector": d["sector"],
            })
        deduped_df = pd.DataFrame(deduped_rows)
        st.dataframe(
            deduped_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Return": st.column_config.NumberColumn("Return", format="%.2%%"),
                "Peak": st.column_config.NumberColumn("Peak", format="%.2%%"),
                "Trough": st.column_config.NumberColumn("Trough", format="%.2%%"),
                "Flagged": st.column_config.NumberColumn("Flagged", help="Times surfaced by screener"),
            },
        )
    else:
        # Original raw view — all predictions
        pred_rows = []
        for p in sorted(active, key=lambda x: x.get("current_return", 0), reverse=True):
            pred_rows.append({
                "Ticker": p["ticker"],
                "Entry Date": p["entry_date"],
                "Entry Price": f"${p.get('entry_price', 0):,.2f}",
                "Current": f"${p.get('current_price', 0):,.2f}",
                "Return": p.get("current_return", 0),
                "Peak": p.get("peak_return", 0),
                "Trough": p.get("trough_return", 0),
                "Days": p.get("days_tracked", 0),
                "Strategy": p.get("strategy", "").replace("_", " ").title(),
                "Confidence": p.get("confidence", ""),
            })

        pred_df = pd.DataFrame(pred_rows)
        st.dataframe(
            pred_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Return": st.column_config.NumberColumn(format="%.2%%"),
                "Peak": st.column_config.NumberColumn(format="%.2%%"),
                "Trough": st.column_config.NumberColumn(format="%.2%%"),
            },
        )
else:
    st.info("No active predictions yet. Run a scan to start tracking.")

# ── Sector distribution ──────────────────────────────────────────────

if candidates:
    st.divider()
    col_sector, col_strategy = st.columns(2)

    with col_sector:
        st.subheader("Sector Distribution")
        sector_counts = {}
        for c in candidates:
            s = c.get("sector", "Unknown")
            sector_counts[s] = sector_counts.get(s, 0) + 1
        sector_df = pd.DataFrame(
            sorted(sector_counts.items(), key=lambda x: -x[1]),
            columns=["Sector", "Count"],
        )
        st.bar_chart(sector_df.set_index("Sector"))

    with col_strategy:
        st.subheader("Strategy Split")
        strat_counts = {"Long-Term Hold": 0, "Short-Term Momentum": 0}
        for c in candidates:
            s = c.get("strategy", {}).get("strategy", "")
            if s == "long_term_hold":
                strat_counts["Long-Term Hold"] += 1
            else:
                strat_counts["Short-Term Momentum"] += 1
        strat_df = pd.DataFrame(
            list(strat_counts.items()), columns=["Strategy", "Count"]
        )
        st.bar_chart(strat_df.set_index("Strategy"))

# ── Paper trading portfolio ──────────────────────────────────────────

PAPER_PORTFOLIO_PATH = Path("paper_trading/portfolio.json")
PAPER_HISTORY_PATH = Path("paper_trading/trade_history.json")

if PAPER_PORTFOLIO_PATH.exists():
    st.divider()
    st.subheader("Paper Trading Portfolio")

    with open(PAPER_PORTFOLIO_PATH) as f:
        paper = json.load(f)

    initial = paper.get("initial_capital", 10000)
    cash = paper.get("cash", 0)
    positions = paper.get("positions", {})
    pos_value = sum(
        p.get("current_value", p.get("cost_basis", 0))
        for p in positions.values()
    )
    total_value = cash + pos_value
    total_return = (total_value - initial) / initial if initial > 0 else 0
    total_trades = paper.get("total_trades", 0)
    realized_pnl = paper.get("total_realized_pnl", 0)

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Portfolio Value", f"${total_value:,.2f}", f"{total_return:+.2%}")
    p2.metric("Cash", f"${cash:,.2f}")
    p3.metric("Positions", len(positions))
    p4.metric("Realized P&L", f"${realized_pnl:+,.2f}")
    if total_trades > 0:
        wr = paper.get("winning_trades", 0) / total_trades
        p5.metric("Closed Trades", f"{total_trades} ({wr:.0%} win)")
    else:
        p5.metric("Closed Trades", "0")

    if positions:
        paper_rows = []
        for ticker, pos in sorted(
            positions.items(),
            key=lambda x: x[1].get("unrealized_pnl_pct", 0),
            reverse=True,
        ):
            paper_rows.append({
                "Ticker": ticker,
                "P&L %": pos.get("unrealized_pnl_pct", 0),
                "P&L $": pos.get("unrealized_pnl", 0),
                "Shares": pos.get("shares", 0),
                "Entry": f"${pos.get('entry_price', 0):,.2f}",
                "Current": f"${pos.get('current_price', 0):,.2f}",
                "Value": f"${pos.get('current_value', 0):,.2f}",
                "Entry Date": pos.get("entry_date", ""),
                "Strategy": pos.get("strategy", "").replace("_", " ").title(),
            })
        paper_df = pd.DataFrame(paper_rows)
        st.dataframe(
            paper_df,
            width="stretch",
            hide_index=True,
            column_config={
                "P&L %": st.column_config.NumberColumn("P&L %", format="%.2%%"),
                "P&L $": st.column_config.NumberColumn("P&L $", format="$%.2f"),
            },
        )

    # Trade history
    if PAPER_HISTORY_PATH.exists():
        with open(PAPER_HISTORY_PATH) as f:
            trade_hist = json.load(f)
        if trade_hist:
            with st.expander(f"Trade History ({len(trade_hist)} trades)"):
                hist_rows = []
                for tr in reversed(trade_hist):
                    hist_rows.append({
                        "Date": tr.get("date", ""),
                        "Type": tr.get("type", ""),
                        "Ticker": tr.get("ticker", ""),
                        "Shares": tr.get("shares", 0),
                        "Price": f"${tr.get('price', 0):,.2f}",
                        "Total": f"${tr.get('total', 0):,.2f}",
                        "P&L": f"${tr.get('pnl', 0):+,.2f}" if "pnl" in tr else "",
                        "Reason": tr.get("reason", ""),
                    })
                st.dataframe(pd.DataFrame(hist_rows), width="stretch", hide_index=True)

# ── Backtest summary (if available) ──────────────────────────────────

backtest = load_backtest()
if backtest:
    st.divider()
    st.subheader("Backtest Results")

    meta = backtest.get("metadata", {})
    analysis = backtest.get("analysis", {})
    overall = analysis.get("overall", {})

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Observations", f"{meta.get('total_observations', 0):,}")
    b2.metric("Baseline 30d Return", f"{overall.get('avg_return_30d', 0):+.2%}")
    b3.metric("Baseline Win Rate", f"{overall.get('win_rate_30d', 0):.0%}")
    b4.metric("Period", f"{meta.get('years', 0)} years")

    indiv = analysis.get("individual_signals", {})
    if indiv:
        bt_rows = []
        for key, d in sorted(indiv.items(), key=lambda x: x[1].get("edge", 0), reverse=True):
            edge = d.get("edge", 0)
            bt_rows.append({
                "Signal": key.replace("_", " ").title(),
                "Verdict": "Positive edge" if edge > 0.01 else ("Neutral" if edge > -0.01 else "No edge"),
                "Edge": edge,
                "High Score Avg": d.get("high_avg_return", 0),
                "Low Score Avg": d.get("low_avg_return", 0),
                "High Win %": d.get("high_win_rate", 0),
                "Samples": d.get("high_count", 0),
            })
        bt_df = pd.DataFrame(bt_rows)
        st.dataframe(
            bt_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Edge": st.column_config.NumberColumn(format="%.2%%"),
                "High Score Avg": st.column_config.NumberColumn(format="%.2%%"),
                "Low Score Avg": st.column_config.NumberColumn(format="%.2%%"),
                "High Win %": st.column_config.NumberColumn(format="%.0%%"),
            },
        )

    # Signal combos
    combos = analysis.get("signal_combos", {})
    if combos:
        with st.expander("Best Signal Combinations"):
            combo_rows = []
            for key, d in sorted(combos.items(), key=lambda x: x[1].get("edge", 0), reverse=True)[:10]:
                combo_rows.append({
                    "Combination": key.replace("_", " ").title(),
                    "Edge": d.get("edge", 0),
                    "Avg Return": d.get("avg_return", 0),
                    "Win %": d.get("win_rate", 0),
                    "Samples": d.get("count", 0),
                })
            if combo_rows:
                st.dataframe(
                    pd.DataFrame(combo_rows),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Edge": st.column_config.NumberColumn(format="%.2%%"),
                        "Avg Return": st.column_config.NumberColumn(format="%.2%%"),
                        "Win %": st.column_config.NumberColumn(format="%.0%%"),
                    },
                )

# ── Footer ───────────────────────────────────────────────────────────

st.divider()
st.caption("Decision-support tool, not financial advice. Run `python screener.py` for a fresh scan.")

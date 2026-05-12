"""
Paper trading simulator.

Simulates actual trading with virtual money, applying the full
soft landing exit rules (tiered stop loss, profit taking, trailing stop).

This is different from the tracker (which just logs recommendations).
The paper trader:
  - Maintains a virtual portfolio with a cash balance
  - Executes "buys" from the screener's top candidates
  - Checks exit rules daily and "sells" when triggered
  - Tracks realized P&L, win rate, and portfolio value over time

Usage:
    python paper_trader.py init --capital 10000    # Initialize with $10K
    python paper_trader.py run                     # Process today's scan + check exits
    python paper_trader.py status                  # Show portfolio + P&L
    python paper_trader.py history                 # Trade history
    python paper_trader.py reset                   # Start over
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PAPER_DIR = Path("paper_trading")
PAPER_DIR.mkdir(exist_ok=True)

PORTFOLIO_FILE = PAPER_DIR / "portfolio.json"
TRADE_HISTORY_FILE = PAPER_DIR / "trade_history.json"

# Import exit rules from config
from config import EXIT_RULES, POSITION_SIZING


def _load_portfolio() -> dict:
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return None


def _save_portfolio(portfolio: dict):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2, default=str)


def _load_history() -> list[dict]:
    if TRADE_HISTORY_FILE.exists():
        with open(TRADE_HISTORY_FILE) as f:
            return json.load(f)
    return []


def _save_history(history: list[dict]):
    with open(TRADE_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════
# Portfolio initialization
# ═══════════════════════════════════════════════════════════════════════

def init_portfolio(capital: float = 10_000) -> dict:
    """Initialize a new paper trading portfolio."""
    portfolio = {
        "initial_capital": capital,
        "cash": capital,
        "positions": {},  # ticker -> position details
        "total_value": capital,
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_realized_pnl": 0,
    }
    _save_portfolio(portfolio)
    logger.info(f"Paper portfolio initialized with ${capital:,.2f}")
    return portfolio


# ═══════════════════════════════════════════════════════════════════════
# Buy logic
# ═══════════════════════════════════════════════════════════════════════

def execute_buys(portfolio: dict, candidates: list[dict], max_new_positions: int = 3) -> dict:
    """
    Buy top candidates that we don't already hold.

    Rules:
    - Only buy if we have cash available
    - Max position size from config (20% of portfolio)
    - Max risk per trade from config (2%)
    - Skip stocks with high earnings risk
    - Only take top N new positions per day to avoid overtrading
    """
    history = _load_history()
    total_value = _compute_total_value(portfolio)
    max_position = total_value * POSITION_SIZING["max_single_position_pct"]
    current_positions = len(portfolio["positions"])

    # Determine how many positions we should hold
    for (lo, hi), (mn, mx) in POSITION_SIZING["tiers"].items():
        if lo <= total_value < hi:
            target_positions = mx
            break
    else:
        target_positions = 10

    slots_available = target_positions - current_positions
    if slots_available <= 0:
        logger.info("Paper trader: portfolio full, no new buys")
        return portfolio

    buys_today = 0

    for candidate in candidates:
        if buys_today >= max_new_positions:
            break
        if buys_today >= slots_available:
            break

        ticker = candidate["ticker"]

        # Skip if already holding
        if ticker in portfolio["positions"]:
            continue

        # Skip if earnings risk is high
        earnings = candidate.get("earnings", {})
        if earnings.get("action") == "avoid":
            logger.info(f"Paper trader: skipping {ticker} — earnings imminent")
            continue

        # Calculate position size
        price = candidate.get("current_price", 0)
        if price <= 0:
            continue

        # Position size = min(max_position, available_cash, recommended_allocation)
        rec_alloc = candidate.get("position_sizing", {}).get("recommended_allocation", max_position)
        allocation = min(max_position, portfolio["cash"], rec_alloc)

        if allocation < price:
            continue  # Can't afford even 1 share

        shares = int(allocation / price)
        cost = shares * price

        if cost > portfolio["cash"]:
            continue

        # Execute buy
        portfolio["cash"] -= cost
        portfolio["positions"][ticker] = {
            "ticker": ticker,
            "shares": shares,
            "entry_price": price,
            "entry_date": datetime.now().isoformat()[:10],
            "cost_basis": cost,
            "current_price": price,
            "current_value": cost,
            "unrealized_pnl": 0,
            "unrealized_pnl_pct": 0,
            "peak_price": price,
            "composite_score": candidate.get("composite_score", 0),
            "strategy": candidate.get("strategy", {}).get("strategy", "unknown"),
            "stop_loss_tier": 0,  # Which tier of stop loss has been triggered
            "original_shares": shares,
            "shares_sold": 0,
            "realized_pnl": 0,
        }

        trade = {
            "type": "BUY",
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "total": cost,
            "date": datetime.now().isoformat()[:10],
            "reason": f"Screener rank #{candidate.get('rank', '?')}, score {candidate.get('composite_score', 0):.1f}",
        }
        history.append(trade)
        buys_today += 1

        logger.info(f"Paper BUY: {shares} {ticker} @ ${price:.2f} = ${cost:,.2f}")

    portfolio["last_updated"] = datetime.now().isoformat()
    _save_portfolio(portfolio)
    _save_history(history)
    return portfolio


# ═══════════════════════════════════════════════════════════════════════
# Exit logic (soft landing)
# ═══════════════════════════════════════════════════════════════════════

def check_exits(portfolio: dict) -> dict:
    """
    Check all positions against exit rules:
    - Tiered stop loss (-7%, -12%, -18%)
    - Profit taking (+20%)
    - Trailing stop (12% from peak)

    Executes partial sells according to the soft landing rules.
    """
    from data_pipeline import get_price_history

    history = _load_history()
    positions_to_remove = []

    for ticker, pos in portfolio["positions"].items():
        try:
            hist = get_price_history(ticker, period="5d")
            if hist is None or hist.empty:
                continue

            current_price = float(hist["Close"].iloc[-1])
            entry_price = pos["entry_price"]
            shares_remaining = pos["shares"]

            if shares_remaining <= 0:
                positions_to_remove.append(ticker)
                continue

            # Update position
            pos["current_price"] = current_price
            pos["current_value"] = shares_remaining * current_price
            pos["unrealized_pnl"] = (current_price - entry_price) * shares_remaining
            pos["unrealized_pnl_pct"] = (current_price - entry_price) / entry_price

            # Track peak for trailing stop
            if current_price > pos.get("peak_price", entry_price):
                pos["peak_price"] = current_price

            pnl_pct = (current_price - entry_price) / entry_price

            # ── Stop loss tiers ───────────────────────────────────────
            for tier_idx, tier in enumerate(EXIT_RULES["stop_loss_tiers"]):
                if tier_idx <= pos.get("stop_loss_tier", -1):
                    continue  # Already triggered this tier

                if pnl_pct <= tier["pct_loss"]:
                    sell_shares = int(pos["original_shares"] * tier["sell_fraction"])
                    sell_shares = min(sell_shares, shares_remaining)

                    if sell_shares > 0:
                        proceeds = sell_shares * current_price
                        cost_per_share = pos["cost_basis"] / pos["original_shares"]
                        realized = (current_price - cost_per_share) * sell_shares

                        portfolio["cash"] += proceeds
                        pos["shares"] -= sell_shares
                        pos["shares_sold"] += sell_shares
                        pos["realized_pnl"] += realized
                        pos["stop_loss_tier"] = tier_idx

                        trade = {
                            "type": "SELL (stop loss)",
                            "ticker": ticker,
                            "shares": sell_shares,
                            "price": current_price,
                            "total": proceeds,
                            "date": datetime.now().isoformat()[:10],
                            "reason": f"Stop loss tier {tier_idx + 1}: {pnl_pct:+.1%}",
                            "pnl": round(realized, 2),
                        }
                        history.append(trade)
                        logger.info(
                            f"Paper SELL (stop): {sell_shares} {ticker} @ ${current_price:.2f} "
                            f"(tier {tier_idx + 1}, {pnl_pct:+.1%})"
                        )

            # ── Profit taking ─────────────────────────────────────────
            for tier in EXIT_RULES.get("profit_taking", []):
                if pnl_pct >= tier["pct_gain"]:
                    sell_shares = int(pos["original_shares"] * tier["sell_fraction"])
                    sell_shares = min(sell_shares, pos["shares"])

                    # Only trigger once
                    if pos.get("profit_taken", False):
                        continue

                    if sell_shares > 0:
                        proceeds = sell_shares * current_price
                        cost_per_share = pos["cost_basis"] / pos["original_shares"]
                        realized = (current_price - cost_per_share) * sell_shares

                        portfolio["cash"] += proceeds
                        pos["shares"] -= sell_shares
                        pos["shares_sold"] += sell_shares
                        pos["realized_pnl"] += realized
                        pos["profit_taken"] = True

                        trade = {
                            "type": "SELL (profit)",
                            "ticker": ticker,
                            "shares": sell_shares,
                            "price": current_price,
                            "total": proceeds,
                            "date": datetime.now().isoformat()[:10],
                            "reason": f"Profit taking at {pnl_pct:+.1%}",
                            "pnl": round(realized, 2),
                        }
                        history.append(trade)
                        logger.info(
                            f"Paper SELL (profit): {sell_shares} {ticker} @ ${current_price:.2f} "
                            f"({pnl_pct:+.1%})"
                        )

            # ── Trailing stop ─────────────────────────────────────────
            peak = pos.get("peak_price", entry_price)
            if peak > entry_price:
                drawdown = (current_price - peak) / peak
                if drawdown <= -EXIT_RULES["trailing_stop_pct"]:
                    # Sell all remaining shares
                    sell_shares = pos["shares"]
                    if sell_shares > 0:
                        proceeds = sell_shares * current_price
                        cost_per_share = pos["cost_basis"] / pos["original_shares"]
                        realized = (current_price - cost_per_share) * sell_shares

                        portfolio["cash"] += proceeds
                        pos["shares"] -= sell_shares
                        pos["shares_sold"] += sell_shares
                        pos["realized_pnl"] += realized

                        trade = {
                            "type": "SELL (trailing stop)",
                            "ticker": ticker,
                            "shares": sell_shares,
                            "price": current_price,
                            "total": proceeds,
                            "date": datetime.now().isoformat()[:10],
                            "reason": f"Trailing stop: {drawdown:+.1%} from peak ${peak:.2f}",
                            "pnl": round(realized, 2),
                        }
                        history.append(trade)
                        logger.info(
                            f"Paper SELL (trailing): {sell_shares} {ticker} @ ${current_price:.2f} "
                            f"({drawdown:+.1%} from peak)"
                        )

            # Clean up fully exited positions
            if pos["shares"] <= 0:
                positions_to_remove.append(ticker)
                portfolio["total_trades"] += 1
                if pos["realized_pnl"] > 0:
                    portfolio["winning_trades"] += 1
                else:
                    portfolio["losing_trades"] += 1
                portfolio["total_realized_pnl"] += pos["realized_pnl"]

        except Exception as e:
            logger.debug(f"Paper trader: error checking {ticker}: {e}")

    # Remove closed positions
    for ticker in positions_to_remove:
        if ticker in portfolio["positions"]:
            del portfolio["positions"][ticker]

    # Update total value
    portfolio["total_value"] = _compute_total_value(portfolio)
    portfolio["last_updated"] = datetime.now().isoformat()
    _save_portfolio(portfolio)
    _save_history(history)

    return portfolio


def _compute_total_value(portfolio: dict) -> float:
    cash = portfolio.get("cash", 0)
    positions_value = sum(
        p.get("current_value", p.get("cost_basis", 0))
        for p in portfolio.get("positions", {}).values()
    )
    return cash + positions_value


# ═══════════════════════════════════════════════════════════════════════
# Run daily (called by scheduler or manually)
# ═══════════════════════════════════════════════════════════════════════

def run_daily() -> dict:
    """
    Daily paper trading routine:
    1. Check exits on existing positions
    2. Buy new candidates from latest scan
    """
    portfolio = _load_portfolio()
    if portfolio is None:
        logger.info("No paper portfolio. Run: python paper_trader.py init --capital 10000")
        return None

    # Check exits first
    logger.info("Paper trader: checking exits...")
    portfolio = check_exits(portfolio)

    # Load latest scan results
    results_file = Path("output/screener_results.json")
    if results_file.exists():
        with open(results_file) as f:
            scan = json.load(f)
        candidates = scan.get("candidates", [])

        # Execute buys
        if candidates:
            logger.info("Paper trader: evaluating new buys...")
            portfolio = execute_buys(portfolio, candidates)

    return portfolio


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════

def print_status():
    portfolio = _load_portfolio()
    if not portfolio:
        print("No paper portfolio. Run: python paper_trader.py init --capital 10000")
        return

    total = _compute_total_value(portfolio)
    initial = portfolio["initial_capital"]
    total_return = (total - initial) / initial

    print(f"\n{'='*60}")
    print(f"PAPER TRADING PORTFOLIO")
    print(f"{'='*60}")
    print(f"Started: {portfolio['created'][:10]}")
    print(f"Initial capital: ${initial:,.2f}")
    print(f"Current value:   ${total:,.2f} ({total_return:+.2%})")
    print(f"Cash:            ${portfolio['cash']:,.2f}")
    print(f"Positions value: ${total - portfolio['cash']:,.2f}")

    total_trades = portfolio.get("total_trades", 0)
    if total_trades > 0:
        win_rate = portfolio["winning_trades"] / total_trades
        print(f"\nClosed trades: {total_trades}")
        print(f"Win rate: {win_rate:.1%} ({portfolio['winning_trades']}W / {portfolio['losing_trades']}L)")
        print(f"Realized P&L: ${portfolio['total_realized_pnl']:,.2f}")

    positions = portfolio.get("positions", {})
    if positions:
        print(f"\n{'─'*60}")
        print(f"OPEN POSITIONS ({len(positions)})")
        print(f"{'─'*60}")
        print(f"{'Ticker':<8} {'Shares':>7} {'Entry':>8} {'Current':>8} {'P&L':>10} {'P&L%':>8}")
        print(f"{'─'*60}")

        for ticker, pos in sorted(positions.items(), key=lambda x: x[1].get("unrealized_pnl_pct", 0), reverse=True):
            pnl = pos.get("unrealized_pnl", 0)
            pnl_pct = pos.get("unrealized_pnl_pct", 0)
            shares = pos.get("shares", 0)
            print(
                f"{ticker:<8} {shares:>7} "
                f"${pos.get('entry_price', 0):>7.2f} "
                f"${pos.get('current_price', 0):>7.2f} "
                f"${pnl:>+9.2f} "
                f"{pnl_pct:>+7.1%}"
            )

        # Exit rule status
        partially_exited = [t for t, p in positions.items() if p.get("shares_sold", 0) > 0]
        if partially_exited:
            print(f"\nPartially exited (stop/profit taken): {', '.join(partially_exited)}")

    print()


def print_history():
    history = _load_history()
    if not history:
        print("No trade history yet.")
        return

    print(f"\n{'='*60}")
    print(f"TRADE HISTORY ({len(history)} trades)")
    print(f"{'='*60}")
    print(f"{'Date':<12} {'Type':<22} {'Ticker':<8} {'Shares':>7} {'Price':>8} {'P&L':>10}")
    print(f"{'─'*70}")

    for trade in history:
        pnl_str = f"${trade.get('pnl', 0):>+9.2f}" if "pnl" in trade else ""
        print(
            f"{trade['date']:<12} {trade['type']:<22} {trade['ticker']:<8} "
            f"{trade['shares']:>7} ${trade['price']:>7.2f} {pnl_str}"
        )
    print()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Paper Trading Simulator")
    parser.add_argument("command", choices=["init", "run", "status", "history", "reset"])
    parser.add_argument("--capital", type=float, default=10_000)

    args = parser.parse_args()

    if args.command == "init":
        init_portfolio(args.capital)
        print_status()
    elif args.command == "run":
        result = run_daily()
        if result:
            print_status()
    elif args.command == "status":
        print_status()
    elif args.command == "history":
        print_history()
    elif args.command == "reset":
        if PORTFOLIO_FILE.exists():
            PORTFOLIO_FILE.unlink()
        if TRADE_HISTORY_FILE.exists():
            TRADE_HISTORY_FILE.unlink()
        print("Paper trading reset. Run 'init' to start fresh.")


if __name__ == "__main__":
    main()

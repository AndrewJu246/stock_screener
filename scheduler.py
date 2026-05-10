"""
Auto-scheduler: runs the screener on a schedule and generates
a daily digest report.

Setup:
  # Option 1: Cron job (Mac/Linux) — runs at 6 PM EST every weekday
  # Run: crontab -e
  # Add: 0 18 * * 1-5 cd /path/to/stock-screener && python scheduler.py

  # Option 2: Windows Task Scheduler
  # Create a task that runs: python /path/to/stock-screener/scheduler.py

  # Option 3: Manual
  # python scheduler.py              — run scan + digest
  # python scheduler.py --digest     — just generate digest from latest results
  # python scheduler.py --email you@gmail.com  — scan + email the digest

Usage:
  python scheduler.py                          # Run scan, print digest
  python scheduler.py --digest                 # Digest only (no new scan)
  python scheduler.py --email you@email.com    # Scan + email digest
  python scheduler.py --capital 15000          # Override capital
"""

import argparse
import json
import logging
import smtplib
import os
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

from config import OUTPUT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_digest(scan_results: dict = None) -> str:
    """
    Generate a daily digest report from scan results.
    Returns a formatted text string.
    """
    if scan_results is None:
        results_path = Path(OUTPUT["output_dir"]) / OUTPUT["results_file"]
        if not results_path.exists():
            return "No scan results found. Run the screener first."
        with open(results_path) as f:
            scan_results = json.load(f)

    summary = scan_results.get("summary", {})
    candidates = scan_results.get("candidates", [])
    market = summary.get("market_data", {})

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"DAILY STOCK SCREENER DIGEST")
    lines.append(f"{datetime.now().strftime('%A, %B %d, %Y')}")
    lines.append(f"{'='*60}")

    # Market overview
    lines.append(f"\nMARKET REGIME: {summary.get('regime', '?').upper()}")
    lines.append(f"  {summary.get('regime_description', '')}")
    lines.append(f"  S&P 500: ${market.get('sp500_price', 0):,.0f}")
    lines.append(f"  VIX: {market.get('vix', 0):.1f}")
    above_50 = "✓" if market.get("sp500_above_50dma") else "✗"
    above_200 = "✓" if market.get("sp500_above_200dma") else "✗"
    lines.append(f"  Above 50 DMA: {above_50}  |  Above 200 DMA: {above_200}")

    # Megatrend momentum
    etf_mom = summary.get("sector_etf_momentum", {})
    if etf_mom:
        lines.append(f"\nMEGATREND MOMENTUM (3-month):")
        for trend, mom in sorted(etf_mom.items(), key=lambda x: -x[1]):
            arrow = "▲" if mom > 0 else "▼"
            lines.append(f"  {arrow} {trend.replace('_', ' '):<20} {mom:+.1f}%")

    # Scan stats
    lines.append(f"\nSCAN SUMMARY:")
    lines.append(f"  Universe: {summary.get('universe_size', 0):,} stocks")
    lines.append(f"  Scored: {summary.get('stocks_scored', 0):,}")
    lines.append(f"  Rejected by quality gate: {summary.get('gate_rejected', 0):,}")
    lines.append(f"  Candidates surfaced: {len(candidates)}")
    sb = summary.get("strategy_breakdown", {})
    lines.append(f"  Long-term holds: {sb.get('long_term_hold', 0)} | Short-term plays: {sb.get('short_term_momentum', 0)}")

    # Top candidates
    if candidates:
        lines.append(f"\n{'─'*60}")
        lines.append("TOP CANDIDATES:")
        lines.append(f"{'─'*60}")

        # Long-term holds
        lt = [c for c in candidates if c["strategy"]["strategy"] == "long_term_hold"][:5]
        if lt:
            lines.append(f"\n  LONG-TERM HOLD (3-12 months):")
            for c in lt:
                conf = c.get("confidence", {}).get("confidence", "?")
                signals = c.get("signals", {})
                top_signals = sorted(
                    signals.items(),
                    key=lambda x: x[1].get("score", 0) if isinstance(x[1], dict) else 0,
                    reverse=True,
                )[:3]
                top_str = ", ".join(
                    f"{name.replace('_', ' ')} {sig['score']:.0f}"
                    for name, sig in top_signals
                    if isinstance(sig, dict)
                )
                lines.append(
                    f"  #{c['rank']:<3} {c['ticker']:<7} "
                    f"Score: {c['composite_score']:.1f}  "
                    f"Conf: {conf:<6}  "
                    f"${c.get('current_price', 0):,.2f}"
                )
                lines.append(f"       {c.get('industry', '')} | Top signals: {top_str}")
                lines.append(f"       {c['strategy'].get('reasoning', '')}")
                # Show warnings
                for sig_name, sig_data in signals.items():
                    if isinstance(sig_data, dict) and "⚠" in sig_data.get("detail", ""):
                        lines.append(f"       ⚠ {sig_data['detail']}")
                lines.append("")

        # Short-term momentum
        st = [c for c in candidates if c["strategy"]["strategy"] == "short_term_momentum"][:5]
        if st:
            lines.append(f"  SHORT-TERM MOMENTUM (2-8 weeks):")
            for c in st:
                conf = c.get("confidence", {}).get("confidence", "?")
                signals = c.get("signals", {})
                top_signals = sorted(
                    signals.items(),
                    key=lambda x: x[1].get("score", 0) if isinstance(x[1], dict) else 0,
                    reverse=True,
                )[:3]
                top_str = ", ".join(
                    f"{name.replace('_', ' ')} {sig['score']:.0f}"
                    for name, sig in top_signals
                    if isinstance(sig, dict)
                )
                lines.append(
                    f"  #{c['rank']:<3} {c['ticker']:<7} "
                    f"Score: {c['composite_score']:.1f}  "
                    f"Conf: {conf:<6}  "
                    f"${c.get('current_price', 0):,.2f}"
                )
                lines.append(f"       {c.get('industry', '')} | Top signals: {top_str}")
                lines.append(f"       {c['strategy'].get('reasoning', '')}")
                lines.append("")

    # Tracker status
    tracker_file = Path("tracker_data/predictions.json")
    if tracker_file.exists():
        with open(tracker_file) as f:
            predictions = json.load(f)
        active = [p for p in predictions if p["status"] == "active"]
        if active:
            lines.append(f"{'─'*60}")
            lines.append(f"TRACKING {len(active)} ACTIVE PREDICTIONS:")
            lines.append(f"{'─'*60}")
            lines.append(f"{'Ticker':<8} {'Days':<6} {'Return':>8} {'Peak':>8} {'Strategy':<18}")
            for p in sorted(active, key=lambda x: x.get("current_return", 0), reverse=True)[:10]:
                lines.append(
                    f"  {p['ticker']:<8} {p.get('days_tracked', 0):<6} "
                    f"{p.get('current_return', 0):>+7.2%} "
                    f"{p.get('peak_return', 0):>+7.2%} "
                    f"{p.get('strategy', ''):<18}"
                )

    lines.append(f"\n{'='*60}")
    lines.append("This is a decision-support tool, not financial advice.")
    lines.append(f"{'='*60}")

    return "\n".join(lines)


def send_email(digest: str, to_email: str):
    """
    Send the digest via email.

    Requires environment variables:
      SMTP_HOST     (default: smtp.gmail.com)
      SMTP_PORT     (default: 587)
      SMTP_USER     (your email)
      SMTP_PASSWORD  (app password for Gmail)
    """
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASSWORD")

    if not smtp_user or not smtp_pass:
        logger.error(
            "Email not configured. Set SMTP_USER and SMTP_PASSWORD environment variables.\n"
            "For Gmail: use an App Password (Settings → Security → App Passwords)"
        )
        return False

    msg = MIMEMultipart()
    msg["Subject"] = f"Stock Screener Digest — {datetime.now().strftime('%b %d, %Y')}"
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg.attach(MIMEText(digest, "plain"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logger.info(f"Digest emailed to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Stock Screener Scheduler")
    parser.add_argument(
        "--digest", action="store_true",
        help="Generate digest only (don't run a new scan)",
    )
    parser.add_argument(
        "--email", type=str, default=None,
        help="Email address to send digest to",
    )
    parser.add_argument(
        "--capital", type=float, default=10_000,
        help="Portfolio capital (default: $10,000)",
    )

    args = parser.parse_args()

    if not args.digest:
        # Run the full scan
        logger.info("Starting scheduled scan...")
        from screener import run_scan
        results = run_scan(capital=args.capital)
        digest = generate_digest(results)
    else:
        digest = generate_digest()

    # Print to console
    print(digest)

    # Email if requested
    if args.email:
        send_email(digest, args.email)

    # Save digest to file
    digest_path = Path(OUTPUT["output_dir"]) / f"digest_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(digest_path, "w") as f:
        f.write(digest)
    logger.info(f"Digest saved to {digest_path}")


if __name__ == "__main__":
    main()

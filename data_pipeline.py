"""
Data pipeline: fetches and caches stock data from free sources.

Sources:
  - yfinance: price, volume, financials, analyst estimates
  - SEC EDGAR: insider transactions (Form 4), institutional holdings (13F)
  - Wikipedia: S&P index constituents
"""

import json
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config import UNIVERSE_MODE, SIGNAL_PARAMS, MEGATRENDS

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Universe loading
# ═══════════════════════════════════════════════════════════════════════

def get_universe() -> list[str]:
    """Get the list of tickers to scan based on UNIVERSE_MODE."""
    cache_file = CACHE_DIR / "universe.json"
    cache_age = _cache_age_hours(cache_file)

    # Refresh universe weekly, but never use an empty cached list
    if cache_age is not None and cache_age < 168:
        with open(cache_file) as f:
            cached = json.load(f)
        if len(cached) > 50:  # Don't use a broken/empty cache
            return cached
        else:
            logger.warning(f"Cached universe only has {len(cached)} tickers — refreshing")

    if UNIVERSE_MODE == "sp1500":
        tickers = _fetch_sp1500()
    else:
        tickers = _fetch_full_universe()

    # Only cache if we got a reasonable number of tickers
    if len(tickers) > 50:
        with open(cache_file, "w") as f:
            json.dump(tickers, f)

    logger.info(f"Universe loaded: {len(tickers)} tickers ({UNIVERSE_MODE})")
    return tickers


def _fetch_sp1500() -> list[str]:
    """Backward compat — now routes to NASDAQ universe."""
    return _fetch_nasdaq_universe()


def _fetch_nasdaq_universe() -> list[str]:
    """
    Fetch all US-traded stocks from NASDAQ's official screener API.
    Aggressively filters out non-common-stock securities.
    """
    import urllib.request
    import ssl
    import certifi
    import re

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "application/json",
    }

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    tickers = set()

    try:
        url = (
            "https://api.nasdaq.com/api/screener/stocks"
            "?tableType=traded&limit=10000&offset=0"
        )
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        rows = raw.get("data", {}).get("table", {}).get("rows", [])
        logger.info(f"NASDAQ API returned {len(rows)} securities")

        for row in rows:
            symbol = row.get("symbol", "").strip()
            name = row.get("name", "").lower()
            market_cap = row.get("marketCap", "")

            if not symbol:
                continue

            # --- Aggressive junk filtering ---

            # Must be only letters + optional hyphen, max 5 chars
            if not all(c.isalpha() or c == "-" for c in symbol):
                continue
            if len(symbol) > 5:
                continue

            # Warrants end in W (BDCIW, BLZRW, etc)
            if len(symbol) >= 4 and symbol.endswith("W"):
                continue

            # Units end in U (BIXIU, BPACU, etc)
            if len(symbol) >= 4 and symbol.endswith("U"):
                continue

            # Rights end in R after 3+ base chars (BHAVR, BSAAR, etc)
            if len(symbol) >= 4 and symbol.endswith("R") and symbol[-2].isalpha():
                # Allow real tickers like ABBR by checking if base exists
                # Simple heuristic: if 4+ chars and ends in R, likely a right
                if len(symbol) >= 5:
                    continue

            # Preferred shares: tickers ending in P after 3+ chars (BPYPO, BHFAP, etc)
            # Be careful not to filter BP, GAP, etc
            if len(symbol) >= 5 and symbol[-1] in ("P", "O", "N") and symbol[-2].isalpha():
                continue

            # Filter by name keywords (closed-end funds, notes, debentures)
            skip_names = ["warrant", "unit", "right", "debenture", "note ",
                          "preferred", "acquisition corp", "blank check"]
            if any(kw in name for kw in skip_names):
                continue

            # Market cap filter
            if market_cap:
                try:
                    mcap_num = float(str(market_cap).replace(",", ""))
                    if mcap_num < 100_000_000:
                        continue
                except (ValueError, TypeError):
                    pass

            tickers.add(symbol)

        logger.info(f"After filtering: {len(tickers)} tradeable stocks")

    except Exception as e:
        logger.warning(f"NASDAQ API fetch failed: {e}")

    if len(tickers) < 50:
        logger.warning("NASDAQ API failed — using hardcoded fallback universe")
        tickers.update(FALLBACK_TICKERS)

    return sorted(tickers)


def _fetch_full_universe() -> list[str]:
    """Fetch all US-traded stocks from NASDAQ."""
    return _fetch_nasdaq_universe()


# Hardcoded fallback: ~300 high-liquidity tickers across sectors
# Used only if NASDAQ API is unreachable
FALLBACK_TICKERS = [
    # Tech / Semiconductors
    "AAPL","MSFT","NVDA","AMD","AVGO","INTC","QCOM","TXN","MU","MRVL","ARM","SMCI","ANET",
    "ADI","NXPI","ON","LRCX","KLAC","AMAT","ASML",
    # Software / Cloud
    "CRWD","PLTR","NOW","PANW","SNOW","DDOG","NET","ZS","FTNT","HUBS","TEAM","MDB","BILL",
    "TTD","DKNG","APP","TOST","DUOL","SHOP","SQ","COIN",
    # Big tech / Internet
    "GOOGL","AMZN","META","NFLX","TSLA","UBER","ABNB","SPOT","SNAP","PINS","RBLX",
    # Healthcare / Biotech
    "LLY","UNH","JNJ","ABBV","MRK","PFE","TMO","ABT","AMGN","GILD","ISRG","DXCM",
    "VRTX","REGN","MRNA","BIIB","ILMN","ZTS","EW","BSX","MDT",
    # Financials
    "JPM","BAC","GS","MS","BLK","SCHW","C","WFC","AXP","V","MA","PYPL","FIS",
    "ICE","CME","MCO","SPGI","COF","DFS",
    # Industrials / Defense
    "GE","CAT","HON","RTX","LMT","NOC","GD","BA","DE","EMR","ETN","ITW",
    "AXON","RKLB","GEV","CARR","TT","PH","ROK",
    # Consumer
    "COST","WMT","TGT","HD","LOW","NKE","SBUX","MCD","YUM","CMG","DPZ",
    "LULU","ELF","DECK","BURL","ROST","TJX","ORLY","AZO",
    # Energy / Utilities
    "XOM","CVX","COP","SLB","EOG","PXD","OXY","VST","CEG","NEE","DUK","SO","AEP",
    # Clean energy
    "ENPH","SEDG","FSLR","RUN","PLUG","BE",
    # Communication
    "DIS","CMCSA","T","VZ","TMUS","CHTR",
    # Materials
    "LIN","APD","SHW","ECL","NEM","FCX","STLD","NUE",
    # REITs
    "PLD","AMT","CCI","EQIX","SPG","O","WELL","DLR",
    # Consumer Staples
    "PG","KO","PEP","PM","MO","CL","EL","STZ","MNST",
    # Other notable
    "BRK-B","MELI","MSTR","IONQ","SOFI","HOOD","RIVN","LCID",
    "CRSP","BEAM","NTLA","EDIT",
    "CRM","ORCL","IBM","ADBE","INTU",
]


# ═══════════════════════════════════════════════════════════════════════
# Price & volume data
# ═══════════════════════════════════════════════════════════════════════

def get_price_history(
    ticker: str,
    period: str = "1y",
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Get OHLCV price history for a ticker.
    Caches daily — won't re-fetch within the same day.
    """
    cache_file = CACHE_DIR / f"price_{ticker}.parquet"

    if use_cache and cache_file.exists():
        age = _cache_age_hours(cache_file)
        if age is not None and age < 18:  # Refresh after market close
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty or len(hist) < 20:
            return None

        hist.to_parquet(cache_file)
        return hist

    except Exception as e:
        logger.debug(f"Failed to get price data for {ticker}: {e}")
        return None


def get_price_history_batch(
    tickers: list[str],
    period: str = "1y",
    batch_size: int = 20,  # Reduced from 50 to prevent DNS overload
) -> dict[str, pd.DataFrame]:
    """
    Batch-download price data for efficiency.
    yfinance supports multi-ticker downloads.
    """
    results = {}
    uncached = []

    # Check cache first
    for t in tickers:
        cache_file = CACHE_DIR / f"price_{t}.parquet"
        age = _cache_age_hours(cache_file)
        if age is not None and age < 18:
            try:
                results[t] = pd.read_parquet(cache_file)
                continue
            except Exception:
                pass
        uncached.append(t)

    logger.info(f"  {len(results)} from cache, {len(uncached)} to download")

    # Batch download uncached tickers
    for i in range(0, len(uncached), batch_size):
        batch = uncached[i:i + batch_size]
        try:
            data = yf.download(
                batch,
                period=period,
                group_by="ticker",
                threads=False,  # Disable threading to prevent DNS overload
                progress=False,
            )
            for t in batch:
                try:
                    if len(batch) == 1:
                        df = data.copy()
                    else:
                        df = data[t].copy()

                    df = df.dropna(how="all")
                    if len(df) >= 20:
                        cache_file = CACHE_DIR / f"price_{t}.parquet"
                        df.to_parquet(cache_file)
                        results[t] = df
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Batch download failed for {batch[:3]}...: {e}")

        # Rate limiting — 2s between batches to prevent DNS overload
        if i + batch_size < len(uncached):
            time.sleep(2)

    logger.info(
        f"Price data: {len(results)}/{len(tickers)} tickers loaded "
        f"({len(results) - (len(tickers) - len(uncached))} from cache)"
    )
    return results


# ═══════════════════════════════════════════════════════════════════════
# Fundamental data (quarterly financials)
# ═══════════════════════════════════════════════════════════════════════

def get_financials(ticker: str) -> Optional[dict]:
    """
    Get key financial data: revenue, earnings, cash flow, margins.
    Returns a dict with quarterly time series.
    """
    cache_file = CACHE_DIR / f"financials_{ticker}.json"
    age = _cache_age_hours(cache_file)

    if age is not None and age < 72:  # Cache for 3 days
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception:
            pass

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        quarterly_financials = stock.quarterly_financials
        quarterly_cashflow = stock.quarterly_cashflow
        quarterly_balance = stock.quarterly_balance_sheet

        result = {
            "ticker": ticker,
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", 0),
            "description": info.get("longBusinessSummary", ""),
            "analyst_target_mean": info.get("targetMeanPrice"),
            "analyst_recommendation": info.get("recommendationKey"),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice")),
            "revenue_quarterly": _extract_series(quarterly_financials, "Total Revenue"),
            "net_income_quarterly": _extract_series(quarterly_financials, "Net Income"),
            "gross_profit_quarterly": _extract_series(quarterly_financials, "Gross Profit"),
            "operating_cashflow_quarterly": _extract_series(quarterly_cashflow, "Operating Cash Flow"),
            "free_cashflow_quarterly": _extract_series(quarterly_cashflow, "Free Cash Flow"),
            "total_debt": _extract_latest(quarterly_balance, "Total Debt"),
            "total_assets": _extract_latest(quarterly_balance, "Total Assets"),
            "rd_expense": _extract_series(quarterly_financials, "Research Development"),
        }

        with open(cache_file, "w") as f:
            json.dump(result, f, default=str)

        return result

    except Exception as e:
        logger.debug(f"Failed to get financials for {ticker}: {e}")
        return None


def _extract_series(df: Optional[pd.DataFrame], row_name: str) -> Optional[list]:
    """Extract a row from quarterly financials as a list of {date, value} dicts."""
    if df is None or df.empty:
        return None
    try:
        if row_name in df.index:
            row = df.loc[row_name].dropna()
            return [
                {"date": str(d.date()), "value": float(v)}
                for d, v in row.items()
            ]
    except Exception:
        pass
    return None


def _extract_latest(df: Optional[pd.DataFrame], row_name: str) -> Optional[float]:
    """Extract the most recent value from a balance sheet row."""
    if df is None or df.empty:
        return None
    try:
        if row_name in df.index:
            row = df.loc[row_name].dropna()
            if not row.empty:
                return float(row.iloc[0])
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════
# Insider transactions (SEC EDGAR Form 4)
# ═══════════════════════════════════════════════════════════════════════

def get_insider_transactions(ticker: str) -> Optional[list[dict]]:
    """
    Get recent insider transactions from yfinance (sources from SEC filings).
    Returns list of {date, insider, type, shares, value} dicts.
    """
    cache_file = CACHE_DIR / f"insider_{ticker}.json"
    age = _cache_age_hours(cache_file)

    if age is not None and age < 24:
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception:
            pass

    try:
        stock = yf.Ticker(ticker)
        insider_df = stock.insider_transactions

        if insider_df is None or insider_df.empty:
            return []

        transactions = []
        for _, row in insider_df.iterrows():
            txn = {
                "date": str(row.get("Start Date", "")),
                "insider": str(row.get("Insider Trading", row.get("Insider", ""))),
                "type": str(row.get("Transaction", row.get("Text", ""))),
                "shares": int(row.get("Shares", 0)) if pd.notna(row.get("Shares")) else 0,
                "value": float(row.get("Value", 0)) if pd.notna(row.get("Value")) else 0,
            }
            transactions.append(txn)

        with open(cache_file, "w") as f:
            json.dump(transactions, f, default=str)

        return transactions

    except Exception as e:
        logger.debug(f"Failed to get insider data for {ticker}: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════
# Institutional holdings (from yfinance, sourced from 13F)
# ═══════════════════════════════════════════════════════════════════════

def get_institutional_holders(ticker: str) -> Optional[list[dict]]:
    """Get top institutional holders."""
    cache_file = CACHE_DIR / f"institutions_{ticker}.json"
    age = _cache_age_hours(cache_file)

    if age is not None and age < 168:  # Weekly refresh
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception:
            pass

    try:
        stock = yf.Ticker(ticker)
        holders = stock.institutional_holders

        if holders is None or holders.empty:
            return []

        result = []
        for _, row in holders.iterrows():
            result.append({
                "holder": str(row.get("Holder", "")),
                "shares": int(row.get("Shares", 0)) if pd.notna(row.get("Shares")) else 0,
                "date_reported": str(row.get("Date Reported", "")),
                "pct_held": float(row.get("% Out", 0)) if pd.notna(row.get("% Out")) else 0,
                "value": float(row.get("Value", 0)) if pd.notna(row.get("Value")) else 0,
            })

        with open(cache_file, "w") as f:
            json.dump(result, f, default=str)

        return result

    except Exception as e:
        logger.debug(f"Failed to get institutional data for {ticker}: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════
# Market regime data (S&P 500, VIX)
# ═══════════════════════════════════════════════════════════════════════

def get_market_regime_data() -> dict:
    """
    Fetch data needed to detect the current market regime:
    S&P 500 vs its moving averages, VIX level.
    """
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="1y")

        if hist.empty:
            return {"regime": "balanced", "reason": "No market data available"}

        close = hist["Close"]
        current = close.iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]

        # VIX
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="5d")
        vix_level = vix_hist["Close"].iloc[-1] if not vix_hist.empty else 20

        return {
            "sp500_price": float(current),
            "sp500_ma50": float(ma50),
            "sp500_ma200": float(ma200),
            "sp500_above_50dma": bool(current > ma50),
            "sp500_above_200dma": bool(current > ma200),
            "vix": float(vix_level),
            "date": str(datetime.now().date()),
        }

    except Exception as e:
        logger.warning(f"Failed to get market regime data: {e}")
        return {"regime": "balanced", "reason": f"Error: {e}"}


# ═══════════════════════════════════════════════════════════════════════
# Sector ETF data (for megatrend tracking)
# ═══════════════════════════════════════════════════════════════════════

def get_sector_etf_momentum() -> dict[str, float]:
    """
    Get 3-month momentum for megatrend ETF proxies.
    Returns {megatrend_key: average_momentum_pct}.
    """
    results = {}

    for trend_key, trend_info in MEGATRENDS.items():
        momentums = []
        for etf in trend_info["etf_proxies"]:
            try:
                hist = get_price_history(etf, period="6mo")
                if hist is not None and len(hist) >= 63:
                    close = hist["Close"]
                    mom_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100
                    momentums.append(float(mom_3m))
            except Exception:
                continue

        results[trend_key] = float(np.mean(momentums)) if momentums else 0.0

    return results


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _cache_age_hours(path: Path) -> Optional[float]:
    """Return the age of a cache file in hours, or None if it doesn't exist."""
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() / 3600


def clear_cache(older_than_hours: int = 0):
    """Clear cache files. If older_than_hours=0, clear everything."""
    count = 0
    for f in CACHE_DIR.iterdir():
        if f.is_file():
            if older_than_hours == 0:
                f.unlink()
                count += 1
            else:
                age = _cache_age_hours(f)
                if age and age > older_than_hours:
                    f.unlink()
                    count += 1
    logger.info(f"Cleared {count} cache files")

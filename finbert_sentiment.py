"""
FinBERT news sentiment analysis via Hugging Face Inference API.

Scores recent financial news headlines using ProsusAI/finbert
through the HF API — no local PyTorch or model download needed.

Setup:
    1. Create free account at huggingface.co
    2. Get API token from huggingface.co/settings/tokens
    3. Set environment variable: export HF_API_TOKEN=hf_xxxxx
       Or create a file: echo "hf_xxxxx" > .hf_token

Free tier: ~30,000 characters/month (plenty for daily scans).
"""

import json
import logging
import os
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
TOKEN_FILE = Path(".hf_token")


def _get_hf_token() -> Optional[str]:
    """Get HuggingFace API token from env var or file."""
    token = os.environ.get("HF_API_TOKEN")
    if token:
        return token.strip()

    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()

    return None


def _query_finbert(texts: list[str]) -> Optional[list]:
    """Send texts to HuggingFace Inference API for sentiment analysis."""
    token = _get_hf_token()

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = json.dumps({"inputs": texts}).encode("utf-8")

    try:
        req = urllib.request.Request(HF_API_URL, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 503:
            logger.debug("FinBERT model is loading on HF — will retry next scan")
        elif e.code == 401:
            logger.warning("HF API token invalid — set HF_API_TOKEN or create .hf_token file")
        else:
            logger.debug(f"HF API error {e.code}: {e.reason}")
        return None
    except Exception as e:
        logger.debug(f"HF API request failed: {e}")
        return None


def get_news_headlines(ticker: str, days_back: int = 7) -> list[str]:
    """Get recent news headlines for a ticker from yfinance."""
    cache_file = CACHE_DIR / f"news_{ticker}.json"

    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = (datetime.now() - mtime).total_seconds() / 3600
        if age < 12:
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return []

        headlines = []
        for article in news:
            title = article.get("title", "")
            if title:
                headlines.append(title)

        with open(cache_file, "w") as f:
            json.dump(headlines[:20], f)

        return headlines[:20]

    except Exception as e:
        logger.debug(f"Failed to get news for {ticker}: {e}")
        return []


def analyze_sentiment(ticker: str, headlines: list[str] = None) -> dict:
    """
    Analyze sentiment of recent news headlines using FinBERT via HF API.

    Returns: {
        "sentiment": "positive" | "negative" | "neutral" | "unavailable",
        "score": float (-1 to 1),
        "score_modifier": float (-10 to +10),
        "num_headlines": int,
        "breakdown": {"positive": n, "negative": n, "neutral": n},
        "detail": str,
    }
    """
    if headlines is None:
        headlines = get_news_headlines(ticker)

    if not headlines:
        return {
            "sentiment": "neutral",
            "score": 0,
            "score_modifier": 0,
            "num_headlines": 0,
            "breakdown": {},
            "detail": "No recent news found",
        }

    # Check sentiment cache
    cache_file = CACHE_DIR / f"sentiment_{ticker}.json"
    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = (datetime.now() - mtime).total_seconds() / 3600
        if age < 12:
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass

    # Query HF API
    truncated = [h[:512] for h in headlines[:15]]
    results = _query_finbert(truncated)

    if results is None:
        return {
            "sentiment": "unavailable",
            "score": 0,
            "score_modifier": 0,
            "num_headlines": len(headlines),
            "breakdown": {},
            "detail": "HF API unavailable — check token or try later",
        }

    try:
        breakdown = {"positive": 0, "negative": 0, "neutral": 0}
        weighted_score = 0
        num = 0

        for result in results:
            if not isinstance(result, list):
                continue

            # Each result is a list of {label, score} dicts
            best = max(result, key=lambda x: x.get("score", 0))
            label = best.get("label", "neutral").lower()
            confidence = best.get("score", 0)

            if label in breakdown:
                breakdown[label] += 1

            if label == "positive":
                weighted_score += confidence
            elif label == "negative":
                weighted_score -= confidence

            num += 1

        if num == 0:
            return {
                "sentiment": "neutral",
                "score": 0,
                "score_modifier": 0,
                "num_headlines": len(headlines),
                "breakdown": breakdown,
                "detail": "Could not parse sentiment results",
            }

        avg_score = weighted_score / num

        if avg_score > 0.15:
            sentiment = "positive"
        elif avg_score < -0.15:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        score_modifier = round(max(-10, min(10, avg_score * 10)), 1)

        result = {
            "sentiment": sentiment,
            "score": round(avg_score, 3),
            "score_modifier": score_modifier,
            "num_headlines": num,
            "breakdown": breakdown,
            "detail": (
                f"News sentiment: {sentiment} ({avg_score:+.2f}) — "
                f"{breakdown['positive']}↑ {breakdown['negative']}↓ {breakdown['neutral']}→ "
                f"from {num} headlines"
            ),
        }

        # Cache result
        with open(cache_file, "w") as f:
            json.dump(result, f)

        return result

    except Exception as e:
        logger.warning(f"FinBERT parsing failed for {ticker}: {e}")
        return {
            "sentiment": "unavailable",
            "score": 0,
            "score_modifier": 0,
            "num_headlines": len(headlines),
            "breakdown": {},
            "detail": f"Analysis failed: {e}",
        }


def is_available() -> bool:
    """Check if FinBERT API is configured."""
    return _get_hf_token() is not None

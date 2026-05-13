"""
FinBERT news sentiment analysis.

Scores recent financial news headlines for each stock using
ProsusAI/finbert, a BERT model fine-tuned on financial text.

Sentiment categories: positive, negative, neutral
The aggregate sentiment becomes a signal modifier.

Requirements:
    pip install transformers torch

If transformers/torch aren't installed, this module gracefully
skips and returns neutral scores.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Check if FinBERT dependencies are available
_FINBERT_AVAILABLE = False
_sentiment_pipeline = None

try:
    from transformers import pipeline as hf_pipeline
    _FINBERT_AVAILABLE = True
except ImportError:
    logger.info(
        "FinBERT not available — install with: pip install transformers torch\n"
        "Sentiment analysis will be skipped."
    )


def _get_pipeline():
    """Lazy-load the FinBERT pipeline (downloads model on first use)."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None and _FINBERT_AVAILABLE:
        try:
            logger.info("Loading FinBERT model (first time may take a minute)...")
            _sentiment_pipeline = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
            )
            logger.info("FinBERT loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}")
            return None
    return _sentiment_pipeline


def get_news_headlines(ticker: str, days_back: int = 7) -> list[str]:
    """
    Get recent news headlines for a ticker from yfinance.
    Returns list of headline strings.
    """
    cache_file = CACHE_DIR / f"news_{ticker}.json"

    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = (datetime.now() - mtime).total_seconds() / 3600
        if age < 12:  # Cache for 12 hours
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
        cutoff = datetime.now() - timedelta(days=days_back)

        for article in news:
            title = article.get("title", "")
            pub_time = article.get("providerPublishTime", 0)

            if pub_time:
                pub_date = datetime.fromtimestamp(pub_time)
                if pub_date < cutoff:
                    continue

            if title:
                headlines.append(title)

        # Cache
        with open(cache_file, "w") as f:
            json.dump(headlines, f)

        return headlines

    except Exception as e:
        logger.debug(f"Failed to get news for {ticker}: {e}")
        return []


def analyze_sentiment(ticker: str, headlines: list[str] = None) -> dict:
    """
    Analyze sentiment of recent news headlines using FinBERT.

    Returns: {
        "sentiment": "positive" | "negative" | "neutral" | "unavailable",
        "score": float (-1 to 1, negative=bearish, positive=bullish),
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

    if not _FINBERT_AVAILABLE:
        return {
            "sentiment": "unavailable",
            "score": 0,
            "score_modifier": 0,
            "num_headlines": len(headlines),
            "breakdown": {},
            "detail": "FinBERT not installed — run: pip install transformers torch",
        }

    pipe = _get_pipeline()
    if pipe is None:
        return {
            "sentiment": "unavailable",
            "score": 0,
            "score_modifier": 0,
            "num_headlines": len(headlines),
            "breakdown": {},
            "detail": "FinBERT failed to load",
        }

    try:
        # Score each headline
        # Truncate long headlines to avoid tokenizer issues
        truncated = [h[:512] for h in headlines[:20]]  # Max 20 headlines
        results = pipe(truncated)

        breakdown = {"positive": 0, "negative": 0, "neutral": 0}
        weighted_score = 0

        for result in results:
            label = result["label"].lower()
            confidence = result["score"]

            if label in breakdown:
                breakdown[label] += 1

            if label == "positive":
                weighted_score += confidence
            elif label == "negative":
                weighted_score -= confidence
            # neutral contributes 0

        # Normalize to -1 to +1
        num = len(results)
        avg_score = weighted_score / num if num > 0 else 0

        # Determine overall sentiment
        if avg_score > 0.15:
            sentiment = "positive"
        elif avg_score < -0.15:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Score modifier: scale to -10 to +10
        score_modifier = round(avg_score * 10, 1)
        score_modifier = max(-10, min(10, score_modifier))

        detail = (
            f"News sentiment: {sentiment} ({avg_score:+.2f}) — "
            f"{breakdown['positive']}↑ {breakdown['negative']}↓ {breakdown['neutral']}→ "
            f"from {num} headlines"
        )

        return {
            "sentiment": sentiment,
            "score": round(avg_score, 3),
            "score_modifier": score_modifier,
            "num_headlines": num,
            "breakdown": breakdown,
            "detail": detail,
        }

    except Exception as e:
        logger.warning(f"FinBERT analysis failed for {ticker}: {e}")
        return {
            "sentiment": "unavailable",
            "score": 0,
            "score_modifier": 0,
            "num_headlines": len(headlines),
            "breakdown": {},
            "detail": f"Analysis failed: {e}",
        }


def is_available() -> bool:
    """Check if FinBERT is ready to use."""
    return _FINBERT_AVAILABLE

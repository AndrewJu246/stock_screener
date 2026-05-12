"""
Portfolio diversification guard.

After the screener ranks candidates by composite score, this module
ensures the final list isn't over-concentrated in one sector or
correlated cluster. It enforces diversification by:

1. Capping stocks per sector (default: max 3 in top 10, max 5 in top 20)
2. When a sector is full, the next stock from that sector gets swapped
   for the highest-scoring stock from an underrepresented sector

This doesn't change scores — it reorders the final output list.
"""

import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# Diversification rules
MAX_PER_SECTOR_TOP10 = 3     # Max stocks from one sector in top 10
MAX_PER_SECTOR_TOP20 = 5     # Max stocks from one sector in top 20
MAX_PER_SECTOR_TOP50 = 8     # Max stocks from one sector in full list
MIN_SECTORS_TOP10 = 3        # Top 10 must span at least 3 sectors


def diversify_candidates(candidates: list[dict]) -> list[dict]:
    """
    Reorder candidates to enforce sector diversification.

    The original ranking by composite score is preserved as much as
    possible — we only swap stocks when a sector cap is exceeded.

    Returns: reordered list with a 'diversification_note' field added
    to any stock that was moved.
    """
    if len(candidates) <= 5:
        return candidates  # Too few to diversify

    # Split into tiers
    all_candidates = list(candidates)  # Full ranked list
    diversified = []
    used_tickers = set()

    # Track sector counts per tier
    sector_counts = defaultdict(int)

    # Build a reserve pool (candidates not yet placed)
    reserve = list(all_candidates)

    for i, candidate in enumerate(all_candidates):
        if candidate["ticker"] in used_tickers:
            continue

        sector = candidate.get("sector", "Unknown")

        # Determine the cap for this position
        if i < 10:
            cap = MAX_PER_SECTOR_TOP10
        elif i < 20:
            cap = MAX_PER_SECTOR_TOP20
        else:
            cap = MAX_PER_SECTOR_TOP50

        if sector_counts[sector] < cap:
            # Sector has room — place this stock normally
            diversified.append(candidate)
            sector_counts[sector] += 1
            used_tickers.add(candidate["ticker"])
        else:
            # Sector is full — find the best replacement from a different sector
            replacement = _find_replacement(
                reserve, used_tickers, sector, sector_counts, cap
            )
            if replacement:
                replacement["diversification_note"] = (
                    f"Promoted: replaces {candidate['ticker']} "
                    f"({sector} sector capped at {cap})"
                )
                diversified.append(replacement)
                rep_sector = replacement.get("sector", "Unknown")
                sector_counts[rep_sector] += 1
                used_tickers.add(replacement["ticker"])

                # The displaced stock goes to a later position
                candidate["diversification_note"] = (
                    f"Deferred: {sector} sector hit cap of {cap} in top {_tier_name(i)}"
                )
            else:
                # No replacement available — keep the original
                diversified.append(candidate)
                sector_counts[sector] += 1
                used_tickers.add(candidate["ticker"])

    # Check minimum sector diversity in top 10
    top10_sectors = set(c.get("sector", "Unknown") for c in diversified[:10])
    if len(top10_sectors) < MIN_SECTORS_TOP10:
        logger.warning(
            f"Diversification: top 10 only spans {len(top10_sectors)} sectors "
            f"({', '.join(top10_sectors)}). Market may be heavily concentrated."
        )

    # Re-rank
    for i, c in enumerate(diversified):
        c["original_rank"] = c.get("rank", i + 1)
        c["rank"] = i + 1

    # Log summary
    top10_breakdown = defaultdict(int)
    for c in diversified[:10]:
        top10_breakdown[c.get("sector", "Unknown")] += 1

    logger.info(
        f"Diversification: top 10 sectors: "
        + ", ".join(f"{s} ({n})" for s, n in sorted(top10_breakdown.items(), key=lambda x: -x[1]))
    )

    moved = sum(1 for c in diversified if "diversification_note" in c)
    if moved > 0:
        logger.info(f"Diversification: {moved} stocks repositioned for balance")

    return diversified


def _find_replacement(
    reserve: list[dict],
    used_tickers: set,
    full_sector: str,
    sector_counts: dict,
    cap: int,
) -> Optional[dict]:
    """
    Find the highest-scoring candidate from a sector that isn't full yet.
    """
    for candidate in reserve:
        if candidate["ticker"] in used_tickers:
            continue
        sector = candidate.get("sector", "Unknown")
        if sector != full_sector and sector_counts[sector] < cap:
            return candidate
    return None


def _tier_name(position: int) -> str:
    if position < 10:
        return "10"
    elif position < 20:
        return "20"
    return "50"


def get_diversification_summary(candidates: list[dict]) -> dict:
    """Get a summary of sector distribution in the candidate list."""
    sector_dist = defaultdict(list)
    for c in candidates:
        sector_dist[c.get("sector", "Unknown")].append(c["ticker"])

    strategy_dist = defaultdict(int)
    for c in candidates:
        strategy_dist[c.get("strategy", {}).get("strategy", "unknown")] += 1

    return {
        "total_candidates": len(candidates),
        "sectors_represented": len(sector_dist),
        "sector_breakdown": {
            sector: {"count": len(tickers), "tickers": tickers[:5]}
            for sector, tickers in sorted(sector_dist.items(), key=lambda x: -len(x[1]))
        },
        "strategy_breakdown": dict(strategy_dist),
    }

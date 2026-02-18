"""Timezone handling and market calendar (ET)."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# Market hours 9:30–16:00 ET
MARKET_OPEN = time(9, 30, tzinfo=ET)
MARKET_CLOSE = time(16, 0, tzinfo=ET)


def market_open_et(d: datetime | None = None) -> datetime:
    """Next/current market open (9:30 ET) on the given date or today ET."""
    base = (d or datetime.now(ET)).replace(tzinfo=ET)
    if isinstance(base, datetime) and base.tzinfo is None:
        base = base.replace(tzinfo=ET)
    open_t = datetime.combine(base.date(), MARKET_OPEN.time(), tzinfo=ET)
    if base.time() > MARKET_CLOSE.time():
        open_t += timedelta(days=1)
    return open_t


def market_close_et(d: datetime | None = None) -> datetime:
    """Market close (16:00 ET) on the given date or today ET."""
    base = (d or datetime.now(ET)).replace(tzinfo=ET)
    if isinstance(base, datetime) and base.tzinfo is None:
        base = base.replace(tzinfo=ET)
    return datetime.combine(base.date(), MARKET_CLOSE.time(), tzinfo=ET)


def bar_timestamps_15min_et(
    start: datetime, end: datetime
) -> list[datetime]:
    """Generate 15-minute bar close timestamps during market hours (9:30–16:00 ET)."""
    out: list[datetime] = []
    current = start.replace(tzinfo=ET) if start.tzinfo is None else start
    end_dt = end.replace(tzinfo=ET) if end.tzinfo is None else end
    while current <= end_dt:
        t = current.time()
        if MARKET_OPEN.time() <= t < MARKET_CLOSE.time():
            out.append(current)
        current += timedelta(minutes=15)
    return out

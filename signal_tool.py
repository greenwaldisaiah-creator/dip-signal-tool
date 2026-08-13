#!/usr/bin/env python3
"""
Dip Signal Tool
===============
Scans a watchlist for oversold pullbacks that historically resolve upward,
confirms them against news sentiment and insider ("smart money") flow,
and pushes alerts to Telegram.

Signal architecture
-------------------
  1. TECHNICAL TRIGGER (0-100)  -- the dip itself must fire first.
       RSI(14) oversold, close below lower Bollinger band, drawdown from
       recent high, capitulation volume, and long-term trend quality.
  2. CONFIRMATION FILTER        -- news + insider flow adjust, never trigger.
       Bearish news or heavy insider selling downgrades or vetoes a signal.
       Insider buying into weakness upgrades it.
  3. REGIME GATE                -- in a risk-off tape every threshold rises.

Usage
-----
  python signal_tool.py scan                 # run once, alert on new signals
  python signal_tool.py scan --dry-run       # print, don't send to Telegram
  python signal_tool.py backtest --years 3   # validate the scoring rules
  python signal_tool.py test-telegram        # verify bot credentials
  python signal_tool.py refresh-flow         # rebuild insider-flow cache

Credentials are read from environment variables (see .env.example).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
STATE_FILE = BASE_DIR / "state.json"

# --- Environment parsing -----------------------------------------------------
# GitHub Actions sets an env var to the EMPTY STRING when the repository
# variable behind it is undefined -- the key exists, so os.environ.get() returns
# "" rather than the default, and float("") raises at import time. That takes
# the whole tool down, not just the command that needed the value. These helpers
# treat empty/whitespace as unset, tolerate values typed with $ or commas, and
# fall back with a warning instead of crashing.

def env_str(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


def _env_num(name: str, default, cast):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    cleaned = raw.strip().replace("$", "").replace(",", "").replace("%", "")
    try:
        return cast(cleaned)
    except ValueError:
        print(f"! {name}={raw!r} is not a number — using {default}", file=sys.stderr)
        return default


def env_float(name: str, default: float) -> float:
    return _env_num(name, default, float)


def env_int(name: str, default: int) -> int:
    return _env_num(name, default, lambda x: int(float(x)))


ALPHA_KEY = env_str("ALPHAVANTAGE_API_KEY")
TELEGRAM_TOKEN = env_str("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = env_str("TELEGRAM_CHAT_ID")

# SEC requires a descriptive User-Agent with contact info on every request.
SEC_USER_AGENT = env_str("SEC_USER_AGENT", "DipSignalTool contact@example.com")

# --- Price data providers ----------------------------------------------------
# Twelve Data is tried first when a key is present: its free tier allows 800
# requests/day (vs Alpha Vantage's 25) and returns deep daily history in one
# call, which is what makes a true 200-day MA and a real backtest possible.
# Alpha Vantage remains the fallback, and is still the sole source of news
# sentiment. If every provider fails, cached history is used.
TD_KEY = env_str("TWELVEDATA_API_KEY")
TD_DAILY_LIMIT = env_int("TD_DAILY_LIMIT", 800)
TD_RATE_SLEEP = env_float("TD_RATE_SLEEP", 8.0)   # free tier: 8 requests/minute
_td_requests_used = 0
_td_last_request_at = 0.0

INDICES = ["SPY", "QQQ", "IWM"]
MEGACAPS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO"]
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "XLC"]

# The watchlist sizes itself to the data budget rather than making you tune it.
# Alpha Vantage alone (25 req/day) supports 11 symbols; adding a Twelve Data key
# lifts the ceiling enough to scan the sector ETFs too.
WATCHLIST = INDICES + MEGACAPS + (SECTORS if TD_KEY else [])
BENCHMARK = "SPY"

# Intraday scans the whole watchlist when Twelve Data is available, and only a
# handful otherwise. 15-minute bars only produce new data every 15 minutes, so
# scanning more often than that returns identical numbers -- 27 sweeps covers a
# full session, which is ~567 of Twelve Data's 800 free daily credits.
INTRADAY_WATCHLIST = WATCHLIST if TD_KEY else ["SPY", "QQQ", "NVDA"]
INTRADAY_INTERVAL = "15min"

# Intraday fires far more often than the end-of-day scan, so it holds a higher
# bar. Without this, 27 sweeps a day across 21 symbols would bury the genuinely
# strong setups under WATCH-tier noise on your phone.
INTRADAY_MIN_SCORE = env_float("INTRADAY_MIN_SCORE", 55.0)


# US market hours in UTC during EDT. Scans outside these bounds would burn
# credits on stale bars, so they exit early.
MARKET_OPEN_UTC = (13, 30)
MARKET_CLOSE_UTC = (20, 0)

# NYSE holidays. Scheduled runs on these days would waste credits and could
# re-alert yesterday's bars as if they were new.
MARKET_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}

# --- Alpha Vantage request budget -------------------------------------------
# Free tier allows 25 requests/day. A daily scan costs 21 price calls + 1
# batched news call = 22, which fits. Anything more must come out of the same
# pot, so requests are counted and hard-capped rather than left to fail
# mid-scan with a confusing rate-limit message.
AV_DAILY_LIMIT = env_int("AV_DAILY_LIMIT", 25)
AV_RATE_SLEEP = 1.2  # Alpha Vantage asks for <=1 request/second on free keys
_av_requests_used = 0
_av_last_request_at = 0.0


def av_pace() -> None:
    """Sleep so consecutive requests stay at least AV_RATE_SLEEP apart.

    This runs BEFORE each request, not after a successful one. Pacing only on
    success is worse than useless: the moment a call fails, every remaining call
    fires back to back and the whole scan gets rate-limited.
    """
    global _av_last_request_at
    wait = AV_RATE_SLEEP - (time.monotonic() - _av_last_request_at)
    if wait > 0:
        time.sleep(wait)
    _av_last_request_at = time.monotonic()


def market_status(now: datetime | None = None) -> tuple[bool, str]:
    """Is the US market open right now? Returns (open, reason).

    Each scheduled run is a separate process, and GitHub's scheduler can fire
    late, so this is checked at runtime rather than trusted to cron alone.
    """
    now = now or datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False, "weekend"
    if now.date().isoformat() in MARKET_HOLIDAYS:
        return False, "market holiday"
    minutes = now.hour * 60 + now.minute
    open_m = MARKET_OPEN_UTC[0] * 60 + MARKET_OPEN_UTC[1]
    close_m = MARKET_CLOSE_UTC[0] * 60 + MARKET_CLOSE_UTC[1]
    if minutes < open_m:
        return False, "before the open"
    if minutes > close_m:
        return False, "after the close"
    return True, "open"


# --- Cross-run usage tracking -----------------------------------------------
# Each scheduled run is a fresh process, so an in-memory counter resets 27 times
# a day and can never see the real daily total. Usage is persisted per UTC date
# in the same cache the workflow already carries between runs.

def _usage_path() -> Path:
    return cache_path(f"usage_{datetime.now(timezone.utc).date().isoformat()}.json")


def _load_usage() -> dict:
    p = _usage_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {"av": 0, "td": 0}


def _save_usage(u: dict) -> None:
    try:
        _usage_path().write_text(json.dumps(u))
    except OSError:
        pass  # usage tracking is best-effort; never fail a scan over it


def load_persisted_usage() -> None:
    """Seed the in-process counters from today's persisted totals."""
    global _av_requests_used, _td_requests_used
    u = _load_usage()
    _av_requests_used = int(u.get("av", 0))
    _td_requests_used = int(u.get("td", 0))


def persist_usage() -> None:
    _save_usage({"av": _av_requests_used, "td": _td_requests_used})


def av_budget_check(n: int = 1) -> None:
    """Raise before spending a request we don't have."""
    global _av_requests_used
    if _av_requests_used + n > AV_DAILY_LIMIT:
        raise DataError(
            f"Alpha Vantage request budget exhausted "
            f"({_av_requests_used}/{AV_DAILY_LIMIT}). Cached data is still used; "
            f"raise AV_DAILY_LIMIT if you hold a premium key."
        )
    _av_requests_used += n


def av_budget_report() -> str:
    parts = [f"Alpha Vantage {_av_requests_used}/{AV_DAILY_LIMIT}"]
    if TD_KEY:
        parts.append(f"Twelve Data {_td_requests_used}/{TD_DAILY_LIMIT}")
    return "Requests used today — " + ", ".join(parts)


# --- Position sizing ---------------------------------------------------------
# Set ACCOUNT_SIZE to enable share-count suggestions in alerts.
ACCOUNT_SIZE = env_float("ACCOUNT_SIZE", 0.0)
RISK_PCT = env_float("RISK_PCT", 1.0)             # % of account risked per trade
MAX_POSITION_PCT = env_float("MAX_POSITION_PCT", 20.0)  # cap per name

# Only these outlets count toward the news signal. Aggregators, promotional
# newsletters and SEO content farms are excluded -- they add noise, not signal.
REPUTABLE_SOURCES = {
    "bloomberg.com", "reuters.com", "wsj.com", "ft.com", "cnbc.com",
    "marketwatch.com", "barrons.com", "theglobeandmail.com", "economist.com",
    "fortune.com", "businessinsider.com", "axios.com", "apnews.com",
    "seekingalpha.com", "quiverquant.com", "benzinga.com",
}

# Technical scoring weights -- these sum to 100.
W_RSI, W_BOLLINGER, W_DRAWDOWN, W_VOLUME, W_TREND = 30, 20, 20, 15, 15

TIER_HIGH, TIER_STRONG, TIER_WATCH = 70, 55, 40
RISK_OFF_PENALTY = 10  # thresholds rise by this much when SPY < SMA200

# Floor for ALL alerts, end-of-day included. Set to 70 to alert only on
# HIGH CONVICTION -- worth doing when a backtest shows the lower tiers
# failing to beat the baseline, which is the common outcome.
ALERT_MIN_SCORE = env_float("ALERT_MIN_SCORE", float(TIER_WATCH))

# Paper-trade every qualifying setup, including ones below the alert threshold.
# Costs nothing, sends no extra notifications, and builds the evidence needed to
# confirm the muted tiers really are worse -- roughly 3x faster than alerts alone.
TRACK_ALL_TIERS = env_str("TRACK_ALL_TIERS", "1").lower() not in ("0", "false", "no")

RESEND_AFTER_DAYS = 5  # don't re-alert the same ticker/tier inside this window


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------

class DataError(Exception):
    """Raised when a data source is unavailable. Callers degrade gracefully."""


def http_get_json(url: str, headers: dict | None = None, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - any failure is a data failure
        raise DataError(f"{type(exc).__name__}: {exc}") from exc


def cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return CACHE_DIR / safe


def cached_json(name: str, max_age_hours: float, loader) -> dict:
    """Return cached JSON if fresh, else call loader() and cache the result.

    On loader failure a stale cache is returned rather than crashing the scan --
    a day-old price series still produces a usable signal.
    """
    path = cache_path(name)
    if path.exists():
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if age_h < max_age_hours:
            return json.loads(path.read_text())
    try:
        data = loader()
        path.write_text(json.dumps(data))
        return data
    except DataError:
        if path.exists():
            print(f"  ! {name}: fetch failed, using stale cache", file=sys.stderr)
            return json.loads(path.read_text())
        raise


# --------------------------------------------------------------------------
# Price data
# --------------------------------------------------------------------------

@dataclass
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def _bars_to_json(bars: list[Bar], source: str) -> dict:
    return {"source": source, "bars": [asdict(b) for b in bars]}


def _bars_from_json(d: dict) -> list[Bar]:
    return [Bar(**x) for x in d.get("bars", [])]


def _merge_bars(old: list[Bar], new: list[Bar]) -> list[Bar]:
    """Union two bar series by date, newer data winning on overlap.

    This is what lets a provider that only returns a short window still build
    real history over time.
    """
    by_date = {b.date: b for b in old}
    by_date.update({b.date: b for b in new})
    return sorted(by_date.values(), key=lambda b: b.date)


# --- Twelve Data -------------------------------------------------------------
# Free tier: 800 requests/day and 8/minute, with deep daily history in a single
# call. That is 32x Alpha Vantage's daily allowance, and it is what makes a real
# 200-day moving average (and a real backtest) possible without paying.

def td_pace() -> None:
    global _td_last_request_at
    wait = TD_RATE_SLEEP - (time.monotonic() - _td_last_request_at)
    if wait > 0:
        time.sleep(wait)
    _td_last_request_at = time.monotonic()


def td_budget_check(n: int = 1) -> None:
    global _td_requests_used
    if _td_requests_used + n > TD_DAILY_LIMIT:
        raise DataError(f"Twelve Data budget exhausted ({_td_requests_used}/{TD_DAILY_LIMIT})")
    _td_requests_used += n


def td_time_series(symbol: str, interval: str = "1day", outputsize: int = 500) -> list[Bar]:
    """Fetch a normalised bar series from Twelve Data."""
    if not TD_KEY:
        raise DataError("TWELVEDATA_API_KEY is not set")
    td_budget_check()
    td_pace()
    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol={urllib.parse.quote(symbol)}&interval={interval}"
        f"&outputsize={outputsize}&apikey={TD_KEY}"
    )
    data = http_get_json(url)
    if str(data.get("status", "")).lower() == "error" or "values" not in data:
        raise DataError(f"{symbol}: {data.get('message') or 'unexpected response'}")
    bars = []
    for v in data["values"]:
        try:
            bars.append(Bar(
                date=v["datetime"],
                open=float(v["open"]), high=float(v["high"]),
                low=float(v["low"]), close=float(v["close"]),
                # Index ETFs sometimes report no volume; treat that as 0 rather
                # than dropping the bar entirely.
                volume=float(v.get("volume") or 0),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    if not bars:
        raise DataError(f"{symbol}: empty series")
    bars.sort(key=lambda b: b.date)
    return bars


def _parse_series(raw: dict, key: str) -> list[Bar]:
    series = raw[key]
    bars = [
        Bar(
            date=d,
            open=float(v["1. open"]),
            high=float(v["2. high"]),
            low=float(v["3. low"]),
            close=float(v["4. close"]),
            volume=float(v["5. volume"]),
        )
        for d, v in series.items()
    ]
    bars.sort(key=lambda b: b.date)
    return bars


def fetch_prices(symbol: str, full: bool = False, max_age_hours: float = 20) -> list[Bar]:
    """Daily OHLCV, oldest first.

    IMPORTANT: `outputsize=full` is a PREMIUM feature on TIME_SERIES_DAILY. A
    free key gets 'compact' -- the most recent 100 bars -- and asking for full
    fails outright. So this defaults to compact, and the long-run trend comes
    from the weekly series instead (see fetch_weekly).

    To claw back history anyway, each response is MERGED into whatever this
    symbol already had cached. Every run adds at most a few new bars but never
    discards old ones, so the stored series grows past 100 bars on its own. Once
    it passes 200 the true 200-day average becomes available for free.
    """
    path = cache_path(f"px_{symbol}.json")

    stored: list[Bar] = []
    if path.exists():
        try:
            cached = json.loads(path.read_text())
            stored = _bars_from_json(cached)
            age_h = (time.time() - path.stat().st_mtime) / 3600
            if age_h < max_age_hours and stored:
                return stored
        except (json.JSONDecodeError, TypeError, KeyError):
            stored = []

    def from_twelvedata() -> list[Bar]:
        return td_time_series(symbol, "1day", 5000 if full else 500)

    def from_alphavantage() -> list[Bar]:
        if not ALPHA_KEY:
            raise DataError("ALPHAVANTAGE_API_KEY is not set")
        av_budget_check()
        av_pace()
        # NOTE: outputsize=full is a PREMIUM feature here. Free keys get the
        # last 100 bars only, which is why the weekly series exists.
        size = "full" if full else "compact"
        url = (
            "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
            f"&symbol={urllib.parse.quote(symbol)}&outputsize={size}&apikey={ALPHA_KEY}"
        )
        data = http_get_json(url)
        if "Time Series (Daily)" not in data:
            note = (data.get("Note") or data.get("Information")
                    or data.get("Error Message"))
            raise DataError(f"{symbol}: {note or 'unexpected response'}")
        return _parse_series(data, "Time Series (Daily)")

    errors = []
    for name, provider in (("twelvedata", from_twelvedata),
                           ("alphavantage", from_alphavantage)):
        try:
            fresh_bars = provider()
        except DataError as exc:
            errors.append(f"{name}: {exc}")
            continue
        merged = _merge_bars(stored, fresh_bars)
        path.write_text(json.dumps(_bars_to_json(merged, name)))
        return merged

    # Every provider failed. Stale history still produces a usable signal, and
    # is far better than dropping the symbol from the scan entirely.
    if stored:
        print(f"  ! {symbol}: all providers failed, using cached history "
              f"({'; '.join(errors)})", file=sys.stderr)
        return stored
    raise DataError("; ".join(errors) or "no price provider available")


def fetch_weekly(symbol: str, max_age_hours: float = 168) -> list[Bar]:
    """Weekly OHLCV covering 20+ years -- and free, unlike daily full history.

    This is what makes a real long-term trend filter possible on a free key.
    A 40-week moving average is the standard equivalent of the 200-day: same
    horizon, one fifth the data. Cached a week, because a weekly bar only
    changes once a week.
    """
    path = cache_path(f"weekly_{symbol}.json")
    if path.exists():
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if age_h < max_age_hours:
            try:
                bars = _bars_from_json(json.loads(path.read_text()))
                if bars:
                    return bars
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

    def from_twelvedata() -> list[Bar]:
        return td_time_series(symbol, "1week", 1000)

    def from_alphavantage() -> list[Bar]:
        if not ALPHA_KEY:
            raise DataError("ALPHAVANTAGE_API_KEY is not set")
        av_budget_check()
        av_pace()
        url = (
            "https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY"
            f"&symbol={urllib.parse.quote(symbol)}&apikey={ALPHA_KEY}"
        )
        data = http_get_json(url)
        if "Weekly Time Series" not in data:
            note = (data.get("Note") or data.get("Information")
                    or data.get("Error Message"))
            raise DataError(f"{symbol} weekly: {note or 'unexpected response'}")
        return _parse_series(data, "Weekly Time Series")

    errors = []
    for name, provider in (("twelvedata", from_twelvedata),
                           ("alphavantage", from_alphavantage)):
        try:
            bars = provider()
        except DataError as exc:
            errors.append(f"{name}: {exc}")
            continue
        path.write_text(json.dumps(_bars_to_json(bars, name)))
        return bars

    if path.exists():
        try:
            return _bars_from_json(json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
    raise DataError("; ".join(errors) or "no weekly provider available")


def fetch_intraday(symbol: str, interval: str = INTRADAY_INTERVAL,
                   max_age_hours: float = 0.4) -> list[Bar]:
    """Intraday OHLCV bars, oldest first.

    Cached only ~25 minutes: the whole point of intraday is freshness. Note
    this is a *separate* request from the daily series and comes out of the
    same free-tier budget -- see the scan-cadence note in the README.
    """
    path = cache_path(f"intraday_{symbol}_{interval}.json")
    if path.exists():
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if age_h < max_age_hours:
            try:
                bars = _bars_from_json(json.loads(path.read_text()))
                if bars:
                    return bars
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

    def from_twelvedata() -> list[Bar]:
        return td_time_series(symbol, interval, 500)

    def from_alphavantage() -> list[Bar]:
        if not ALPHA_KEY:
            raise DataError("ALPHAVANTAGE_API_KEY is not set")
        av_budget_check()
        av_pace()
        url = (
            "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
            f"&symbol={urllib.parse.quote(symbol)}&interval={interval}"
            f"&outputsize=full&apikey={ALPHA_KEY}"
        )
        data = http_get_json(url)
        if f"Time Series ({interval})" not in data:
            note = (data.get("Note") or data.get("Information")
                    or data.get("Error Message"))
            raise DataError(f"{symbol} intraday: {note or 'unexpected response'}")
        return _parse_series(data, f"Time Series ({interval})")

    errors = []
    for name, provider in (("twelvedata", from_twelvedata),
                           ("alphavantage", from_alphavantage)):
        try:
            bars = provider()
        except DataError as exc:
            errors.append(f"{name}: {exc}")
            continue
        path.write_text(json.dumps(_bars_to_json(bars, name)))
        return bars
    raise DataError("; ".join(errors) or "no intraday provider available")


def blend_intraday_with_daily(intra: Technicals, daily: Technicals) -> Technicals:
    """Intraday oscillators + daily trend context.

    A 200-period average of 15-minute bars spans about eight sessions, which is
    noise, not trend. So the oversold/stretched/volume factors come from the
    intraday series while the moving averages -- and therefore the trend-quality
    score and the falling-knife veto -- keep using daily bars. Drawdown is taken
    from the daily 20-day high so "how far off the recent high" still means what
    it does in the end-of-day scan.
    """
    return Technicals(
        close=intra.close,
        rsi14=intra.rsi14,
        sma20=intra.sma20,
        sma50=daily.sma50, sma200=daily.sma200,     # trend context: daily
        bb_lower=intra.bb_lower, bb_upper=intra.bb_upper,
        bb_width_pct=intra.bb_width_pct,
        pct_from_bb_lower=intra.pct_from_bb_lower,
        high20=daily.high20, high50=daily.high50,
        dd_from_high20=((daily.high20 - intra.close) / daily.high20 * 100)
                       if daily.high20 else None,
        dd_from_high50=((daily.high50 - intra.close) / daily.high50 * 100)
                       if daily.high50 else None,
        vol_ratio=intra.vol_ratio,
        atr_pct=daily.atr_pct,                       # size on daily volatility
        # Trend reference and the 52-week high are long-run context by
        # definition -- they must come from the daily/weekly legs, or the
        # falling-knife veto silently stops working in intraday mode.
        trend_ma=daily.trend_ma,
        trend_basis=daily.trend_basis,
        high52w=daily.high52w,
    )


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

def sma(values: list[float], n: int) -> float | None:
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def stdev(values: list[float], n: int) -> float | None:
    if len(values) < n:
        return None
    window = values[-n:]
    mean = sum(window) / n
    var = sum((x - mean) ** 2 for x in window) / n
    return math.sqrt(var)


def rsi(closes: list[float], period: int = 14) -> float | None:
    """Wilder's RSI.

    Seeded with a simple average of the first `period` changes, then smoothed
    forward across the remaining history -- the standard construction. Returns
    None until there is enough history.
    """
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for prev, cur in zip(closes[:-1], closes[1:]):
        change = cur - prev
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(bars: list[Bar], period: int = 14) -> float | None:
    if len(bars) < period + 1:
        return None
    trs = []
    for prev, cur in zip(bars[-(period + 1):-1], bars[-period:]):
        trs.append(max(
            cur.high - cur.low,
            abs(cur.high - prev.close),
            abs(cur.low - prev.close),
        ))
    return sum(trs) / len(trs)


@dataclass
class Technicals:
    close: float
    rsi14: float | None
    sma20: float | None
    sma50: float | None
    sma200: float | None
    bb_lower: float | None
    bb_upper: float | None
    bb_width_pct: float | None     # band width as a percent of price
    pct_from_bb_lower: float | None
    high20: float | None
    high50: float | None
    dd_from_high20: float | None   # positive number = percent below the high
    dd_from_high50: float | None
    vol_ratio: float | None        # today's volume / 20-day average
    atr_pct: float | None
    # Long-run trend reference. Prefers the 200-day MA when enough daily history
    # has accumulated, else the 40-week MA from the free weekly series.
    trend_ma: float | None = None
    trend_basis: str = "none"      # "200d" | "40w" | "none"
    high52w: float | None = None
    ann_vol_pct: float | None = None   # annualised realised volatility


def compute_technicals(bars: list[Bar]) -> Technicals:
    closes = [b.close for b in bars]
    vols = [b.volume for b in bars]
    close = closes[-1]

    s20, s50, s200 = sma(closes, 20), sma(closes, 50), sma(closes, 200)
    sd20 = stdev(closes, 20)
    bb_lower = s20 - 2 * sd20 if s20 is not None and sd20 is not None else None
    bb_upper = s20 + 2 * sd20 if s20 is not None and sd20 is not None else None

    high20 = max(b.high for b in bars[-20:]) if len(bars) >= 20 else None
    high50 = max(b.high for b in bars[-50:]) if len(bars) >= 50 else None

    avg_vol20 = sma(vols, 20)

    return Technicals(
        close=close,
        rsi14=rsi(closes, 14),
        sma20=s20, sma50=s50, sma200=s200,
        bb_lower=bb_lower, bb_upper=bb_upper,
        bb_width_pct=(((bb_upper - bb_lower) / close * 100)
                      if bb_lower is not None and bb_upper is not None and close else None),
        pct_from_bb_lower=((close - bb_lower) / bb_lower * 100) if bb_lower else None,
        high20=high20, high50=high50,
        dd_from_high20=((high20 - close) / high20 * 100) if high20 else None,
        dd_from_high50=((high50 - close) / high50 * 100) if high50 else None,
        vol_ratio=(vols[-1] / avg_vol20) if avg_vol20 else None,
        atr_pct=((atr(bars) or 0) / close * 100) if close else None,
        trend_ma=s200,
        trend_basis="200d" if s200 is not None else "none",
        ann_vol_pct=_ann_vol(closes),
    )


def _ann_vol(closes: list[float], period: int = 20) -> float | None:
    """Annualised realised volatility from daily returns (252 trading days)."""
    if len(closes) < period + 1:
        return None
    rets = [(b - a) / a for a, b in zip(closes[-(period + 1):-1], closes[-period:]) if a]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252) * 100


def attach_weekly_trend(t: Technicals, weekly: list[Bar]) -> Technicals:
    """Supply the long-run trend from weekly bars when daily history is short.

    A free Alpha Vantage key returns only 100 daily bars, so sma200 is usually
    None on a fresh install. The 40-week moving average spans the same ~200
    trading days and comes from an endpoint that is free at full history.
    A true 200-day MA still wins when it is available.
    """
    updates: dict = {}
    if t.trend_basis != "200d":
        w40 = sma([b.close for b in weekly], 40)
        if w40 is not None:
            updates["trend_ma"], updates["trend_basis"] = w40, "40w"
    if len(weekly) >= 52:
        updates["high52w"] = max(b.high for b in weekly[-52:])
    # Return a copy rather than mutating in place: callers hold references to
    # the original, and silently rewriting it under them hides bugs.
    return replace(t, **updates) if updates else t


# --------------------------------------------------------------------------
# Technical scoring -- this is the trigger
# --------------------------------------------------------------------------

def score_technicals(t: Technicals) -> tuple[float, list[str]]:
    score, reasons = 0.0, []

    # 1. Oversold momentum.
    if t.rsi14 is not None:
        if t.rsi14 <= 25:
            score += W_RSI; reasons.append(f"RSI {t.rsi14:.0f} deeply oversold")
        elif t.rsi14 <= 30:
            score += W_RSI * 0.8; reasons.append(f"RSI {t.rsi14:.0f} oversold")
        elif t.rsi14 <= 35:
            score += W_RSI * 0.53; reasons.append(f"RSI {t.rsi14:.0f} approaching oversold")
        elif t.rsi14 <= 40:
            score += W_RSI * 0.27; reasons.append(f"RSI {t.rsi14:.0f} weak")

    # 2. Stretched below the volatility envelope.
    #    Guard on band width: when 20-day volatility collapses the bands sit
    #    almost on top of the price, so "touching the lower band" carries no
    #    information and would otherwise hand out free points.
    MIN_BB_WIDTH_PCT = 1.0
    if t.pct_from_bb_lower is not None and (t.bb_width_pct or 0) >= MIN_BB_WIDTH_PCT:
        if t.pct_from_bb_lower < 0:
            score += W_BOLLINGER
            reasons.append(f"{abs(t.pct_from_bb_lower):.1f}% below lower Bollinger band")
        elif t.pct_from_bb_lower < 1.0:
            score += W_BOLLINGER * 0.5
            reasons.append("hugging lower Bollinger band")

    # 3. Depth of the pullback.
    if t.dd_from_high20 is not None:
        dd = t.dd_from_high20
        if dd >= 10:
            score += W_DRAWDOWN; reasons.append(f"-{dd:.1f}% from 20d high")
        elif dd >= 7:
            score += W_DRAWDOWN * 0.75; reasons.append(f"-{dd:.1f}% from 20d high")
        elif dd >= 5:
            score += W_DRAWDOWN * 0.5; reasons.append(f"-{dd:.1f}% from 20d high")
        elif dd >= 3:
            score += W_DRAWDOWN * 0.25; reasons.append(f"-{dd:.1f}% from 20d high")

    # 4. Capitulation volume -- sellers exhausting themselves.
    if t.vol_ratio is not None:
        if t.vol_ratio >= 2.0:
            score += W_VOLUME; reasons.append(f"volume {t.vol_ratio:.1f}x average (capitulation)")
        elif t.vol_ratio >= 1.5:
            score += W_VOLUME * 0.67; reasons.append(f"volume {t.vol_ratio:.1f}x average")
        elif t.vol_ratio >= 1.2:
            score += W_VOLUME * 0.33; reasons.append(f"volume {t.vol_ratio:.1f}x average")

    # 5. Trend quality. Buying dips only pays inside an uptrend -- this is the
    #    single most important guard against catching a falling knife.
    if t.trend_ma is not None:
        label = "200d MA" if t.trend_basis == "200d" else "40w MA"
        rel = (t.close - t.trend_ma) / t.trend_ma * 100
        if rel > 0:
            score += W_TREND; reasons.append(f"above {label} (uptrend intact)")
        elif rel > -3:
            score += W_TREND * 0.47; reasons.append(f"just below {label}")
        else:
            reasons.append(f"{abs(rel):.1f}% below {label} (downtrend)")
    else:
        # No long-run reference at all: withhold the points rather than assume
        # a healthy trend, so an unverified setup can never reach a top tier.
        reasons.append("no long-term trend data (trend points withheld)")

    return round(score, 1), reasons


# --------------------------------------------------------------------------
# News sentiment -- confirmation layer
# --------------------------------------------------------------------------

@dataclass
class NewsSignal:
    score: float = 0.0          # -1 (bearish) .. +1 (bullish)
    article_count: int = 0
    headlines: list[str] = field(default_factory=list)
    available: bool = False


def fetch_news(symbols: list[str]) -> dict[str, NewsSignal]:
    """One batched Alpha Vantage call covering the whole watchlist.

    Sentiment is relevance-weighted and recency-weighted, and only articles
    from REPUTABLE_SOURCES are counted.
    """
    out = {s: NewsSignal() for s in symbols}

    def loader() -> dict:
        if not ALPHA_KEY:
            raise DataError("ALPHAVANTAGE_API_KEY is not set")
        av_budget_check()
        tickers = ",".join(symbols)
        url = (
            "https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
            f"&tickers={urllib.parse.quote(tickers)}&limit=1000&sort=LATEST&apikey={ALPHA_KEY}"
        )
        data = http_get_json(url)
        if "feed" not in data:
            note = data.get("Note") or data.get("Information") or data.get("Error Message")
            raise DataError(f"news: {note or 'unexpected response'}")
        time.sleep(AV_RATE_SLEEP)
        return data

    try:
        data = cached_json("news_feed.json", 8, loader)
    except DataError as exc:
        print(f"  ! news unavailable: {exc}", file=sys.stderr)
        return out

    now = datetime.now(timezone.utc)
    acc: dict[str, list[tuple[float, float, str]]] = {s: [] for s in symbols}

    for item in data.get("feed", []):
        domain = (item.get("source_domain") or "").lower()
        if domain not in REPUTABLE_SOURCES:
            continue
        try:
            published = datetime.strptime(item["time_published"], "%Y%m%dT%H%M%S")
            published = published.replace(tzinfo=timezone.utc)
        except Exception:  # noqa: BLE001 - malformed timestamp, skip the article
            continue
        age_days = (now - published).total_seconds() / 86400
        if age_days > 7:
            continue
        recency_w = math.exp(-age_days / 3.0)  # ~3-day half-life

        for ts in item.get("ticker_sentiment", []):
            sym = ts.get("ticker")
            if sym not in acc:
                continue
            try:
                relevance = float(ts.get("relevance_score", 0))
                sentiment = float(ts.get("ticker_sentiment_score", 0))
            except (TypeError, ValueError):
                continue
            if relevance < 0.1:  # passing mention, not really about this name
                continue
            acc[sym].append((relevance * recency_w, sentiment, item.get("title", "")))

    for sym, rows in acc.items():
        if not rows:
            continue
        total_w = sum(w for w, _, _ in rows)
        if total_w <= 0:
            continue
        weighted = sum(w * s for w, s, _ in rows) / total_w
        top = sorted(rows, key=lambda r: -r[0])[:3]
        out[sym] = NewsSignal(
            score=round(weighted, 3),
            article_count=len(rows),
            headlines=[t for _, _, t in top if t],
            available=True,
        )
    return out


# --------------------------------------------------------------------------
# Insider / institutional flow -- the equity analogue of "whale wallets"
# --------------------------------------------------------------------------

@dataclass
class FlowSignal:
    net_usd: float = 0.0        # insider buys minus sells, last 90 days
    buy_count: int = 0
    sell_count: int = 0
    available: bool = False

    @property
    def direction(self) -> str:
        if not self.available or (self.buy_count + self.sell_count) == 0:
            return "none"
        if self.net_usd > 1_000_000:
            return "buying"
        if self.net_usd < -1_000_000:
            return "selling"
        return "neutral"


def _sec_get(url: str) -> dict:
    return http_get_json(url, headers={
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    })


def cik_for(symbol: str) -> str | None:
    def loader() -> dict:
        return _sec_get("https://www.sec.gov/files/company_tickers.json")

    try:
        table = cached_json("sec_tickers.json", 24 * 30, loader)
    except DataError:
        return None
    for row in table.values():
        if row.get("ticker", "").upper() == symbol.upper():
            return str(row["cik_str"]).zfill(10)
    return None


def fetch_flow(symbol: str, lookback_days: int = 90, max_filings: int = 40) -> FlowSignal:
    """Net open-market insider buying from SEC Form 4 filings.

    Transaction code P = open-market purchase, S = open-market sale. Grants,
    option exercises and tax withholding are deliberately ignored -- only cash
    an insider chose to put in or take out carries information.

    ETFs have no insiders; this returns an unavailable signal for them.
    """
    sig = FlowSignal()
    cik = cik_for(symbol)
    if not cik:
        return sig

    def loader() -> dict:
        return _sec_get(f"https://data.sec.gov/submissions/CIK{cik}.json")

    try:
        subs = cached_json(f"sec_subs_{symbol}.json", 24, loader)
    except DataError as exc:
        print(f"  ! {symbol} flow unavailable: {exc}", file=sys.stderr)
        return sig

    recent = subs.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accns = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    docs = recent.get("primaryDocument", [])

    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date()
    targets = []
    for form, accn, fdate, doc in zip(forms, accns, dates, docs):
        if form != "4":
            continue
        try:
            if datetime.strptime(fdate, "%Y-%m-%d").date() < cutoff:
                continue
        except ValueError:
            continue
        targets.append((accn.replace("-", ""), doc))
        if len(targets) >= max_filings:
            break

    if not targets:
        sig.available = True  # we looked, there was simply nothing
        return sig

    import re
    import xml.etree.ElementTree as ET

    net, buys, sells = 0.0, 0, 0
    parsed_any = False

    for accn_nodash, doc in targets:
        # primaryDocument is usually the rendered .xml form itself.
        xml_doc = doc if doc.lower().endswith(".xml") else re.sub(r"\.html?$", ".xml", doc)
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_nodash}/{xml_doc}"
        path = cache_path(f"form4_{accn_nodash}.xml")
        try:
            if path.exists():
                xml_text = path.read_text()
            else:
                req = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    xml_text = resp.read().decode("utf-8", errors="replace")
                path.write_text(xml_text)
                time.sleep(0.12)  # SEC asks for <10 req/sec
            root = ET.fromstring(xml_text)
        except Exception:  # noqa: BLE001 - a single unreadable filing is not fatal
            continue

        parsed_any = True
        for txn in root.iter("nonDerivativeTransaction"):
            def val(tag: str) -> str | None:
                node = txn.find(f".//{tag}/value")
                return node.text if node is not None and node.text else None

            code = val("transactionCode")
            if code not in ("P", "S"):
                continue
            try:
                shares = float(val("transactionShares") or 0)
                price = float(val("transactionPricePerShare") or 0)
            except ValueError:
                continue
            amount = shares * price
            if amount <= 0:
                continue
            if code == "P":
                net += amount; buys += 1
            else:
                net -= amount; sells += 1

    sig.net_usd, sig.buy_count, sig.sell_count = net, buys, sells
    sig.available = parsed_any
    return sig


# --------------------------------------------------------------------------
# Market regime
# --------------------------------------------------------------------------

@dataclass
class Regime:
    risk_on: bool = True
    label: str = "unknown"
    spy_vs_200: float = 0.0
    breadth: float = 0.0  # fraction of watchlist above its own 50d MA


def compute_regime(tech_by_symbol: dict[str, Technicals]) -> Regime:
    bench = tech_by_symbol.get(BENCHMARK)
    if not bench or bench.trend_ma is None:
        return Regime()
    spy_vs_200 = (bench.close - bench.trend_ma) / bench.trend_ma * 100

    above = [
        1 for t in tech_by_symbol.values()
        if t.sma50 is not None and t.close > t.sma50
    ]
    breadth = len(above) / max(len(tech_by_symbol), 1)

    risk_on = spy_vs_200 > 0
    if spy_vs_200 > 3 and breadth >= 0.5:
        label = "risk-on"
    elif spy_vs_200 > 0:
        label = "neutral"
    else:
        label = "risk-off"
    return Regime(risk_on=risk_on, label=label, spy_vs_200=spy_vs_200, breadth=breadth)


# --------------------------------------------------------------------------
# Combining everything
# --------------------------------------------------------------------------

@dataclass
class TradePlan:
    """Concrete levels for a signal, derived from the instrument's volatility.

    Nothing here is a forecast. The target is a mean-reversion level (the
    20-day average, which is the middle Bollinger band the price has fallen
    away from), the stop is a volatility multiple, and the holding estimate is
    simply distance-to-target divided by the average daily range. They describe
    a plan, not a prediction -- whether the plan actually pays is what the
    paper-trading tracker measures.
    """
    entry: float
    stop: float
    target: float
    risk_per_share: float
    reward_per_share: float
    rr: float                  # reward-to-risk ratio
    target_pct: float          # % gain if the target is reached
    stop_pct: float            # % loss if the stop is hit
    hold_days_est: int         # ATR-based estimate of bars to target
    atr_pct: float | None
    ann_vol_pct: float | None


# Default trade geometry. These are the knobs `optimize` sweeps.
STOP_ATR_MULT = env_float("STOP_ATR_MULT", 1.5)
TARGET_ATR_MULT = env_float("TARGET_ATR_MULT", 0.0)   # 0 = use the 20-day mean


def build_trade_plan(t: Technicals, stop_mult: float | None = None,
                     target_mult: float | None = None) -> TradePlan | None:
    """Entry / stop / target for a dip signal, or None if inputs are unusable.

    `stop_mult` is in ATR units. `target_mult` in ATR units too; pass 0 (the
    default) to target reversion to the 20-day mean instead of a fixed multiple.
    Both are swept by the `optimize` command -- widening the target raises R:R
    but lowers the hit rate, and only measurement can say where expectancy peaks.
    """
    stop_mult = STOP_ATR_MULT if stop_mult is None else stop_mult
    target_mult = TARGET_ATR_MULT if target_mult is None else target_mult

    if not t.close or not t.atr_pct or t.atr_pct <= 0:
        return None
    atr = t.close * t.atr_pct / 100

    # Entry: don't chase. If price is still above the lower band, wait for it;
    # if it has already broken below, the close is the reference.
    entry = min(t.close, t.bb_lower) if t.bb_lower else t.close
    if entry <= 0:
        return None

    stop = entry - stop_mult * atr

    if target_mult and target_mult > 0:
        target = entry + target_mult * atr
    else:
        # Reversion to the 20-day mean, floored at 1 ATR so a shallow dip
        # cannot produce a target too close to be worth the risk.
        target = max(t.sma20 or entry + 2 * atr, entry + atr)

    risk = entry - stop
    reward = target - entry
    if risk <= 0 or reward <= 0:
        return None

    # Holding estimate: how many average daily ranges lie between entry and
    # target. Mean reversion rarely travels a full ATR of *net* progress per
    # day, so this uses half an ATR/day and floors at 2 days.
    hold = max(2, math.ceil(reward / (0.5 * atr)))

    # Round directionally, never favourably: entry and stop DOWN, target UP.
    # Plain round() can push the entry above the close, turning a "don't chase"
    # limit into a chase, and can shave a target into being easier to hit.
    def floor2(x: float) -> float:
        return math.floor(x * 100) / 100

    def ceil2(x: float) -> float:
        return math.ceil(x * 100) / 100

    entry, stop, target = floor2(entry), floor2(stop), ceil2(target)
    risk, reward = entry - stop, target - entry
    if risk <= 0 or reward <= 0:
        return None

    return TradePlan(
        entry=entry, stop=stop, target=target,
        risk_per_share=round(risk, 2), reward_per_share=round(reward, 2),
        rr=round(reward / risk, 2),
        target_pct=round(reward / entry * 100, 2),
        stop_pct=round(risk / entry * 100, 2),
        hold_days_est=min(hold, 60),
        atr_pct=round(t.atr_pct, 2) if t.atr_pct else None,
        ann_vol_pct=round(t.ann_vol_pct, 1) if t.ann_vol_pct else None,
    )


@dataclass
class Signal:
    symbol: str
    tier: str
    final_score: float
    tech_score: float
    price: float
    reasons: list[str]
    notes: list[str]
    news: NewsSignal
    flow: FlowSignal
    tech: Technicals
    stop_hint: float | None = None
    shares: int | None = None
    risk_usd: float | None = None
    notional: float | None = None
    plan: TradePlan | None = None


def evaluate(symbol: str, tech: Technicals, news: NewsSignal,
             flow: FlowSignal, regime: Regime) -> Signal | None:
    tech_score, reasons = score_technicals(tech)
    score = tech_score
    notes: list[str] = []

    # --- Confirmation layer: news ---
    if news.available:
        if news.score <= -0.35:
            score -= 15
            notes.append(f"bearish news flow ({news.score:+.2f}, {news.article_count} articles) — downgraded")
        elif news.score <= -0.15:
            score -= 7
            notes.append(f"slightly negative news ({news.score:+.2f})")
        elif news.score >= 0.25:
            score += 8
            notes.append(f"positive news flow ({news.score:+.2f}) despite the dip")
        elif news.score >= 0.15:
            score += 4
            notes.append(f"mildly positive news ({news.score:+.2f})")
        else:
            notes.append(f"news neutral ({news.score:+.2f})")

    # --- Confirmation layer: insider flow ---
    if flow.available and flow.direction != "none":
        if flow.direction == "buying":
            score += 10
            notes.append(f"insiders net buying ${flow.net_usd/1e6:.1f}M in 90d ({flow.buy_count} buys)")
        elif flow.direction == "selling":
            score -= 8
            notes.append(f"insiders net selling ${abs(flow.net_usd)/1e6:.1f}M in 90d ({flow.sell_count} sales)")
        else:
            notes.append("insider flow balanced")

    # --- Falling-knife veto ---
    # A broken long-term trend plus a deep drawdown is the classic value trap.
    # This fires on price alone: news is only allowed to RESCUE the setup, never
    # to enable the veto. Gating the veto on news being present would disable it
    # exactly when the news feed is down, which is the worst possible time.
    # Drawdown reference: prefer the 52-week high from weekly data when we have
    # it, since 100 daily bars only reach back ~5 months and would understate
    # how far a name has actually fallen.
    deep_dd = max(
        tech.dd_from_high50 or 0,
        ((tech.high52w - tech.close) / tech.high52w * 100) if tech.high52w else 0,
    )
    broken_trend = (
        tech.trend_ma is not None
        and tech.close < tech.trend_ma
        and deep_dd > 20
    )
    if broken_trend:
        rescued = news.available and news.score >= 0.15
        if not rescued:
            why = "no positive news to offset" if news.available else "news unavailable"
            label = "200d MA" if tech.trend_basis == "200d" else "40w MA"
            notes.append(f"VETO: {deep_dd:.0f}% off the high and below "
                         f"{label} — {why} (falling knife)")
            return None
        notes.append("broken trend, but news flow is positive — allowed through")

    # --- Regime gate ---
    hi, strong, watch = TIER_HIGH, TIER_STRONG, TIER_WATCH
    if not regime.risk_on:
        hi += RISK_OFF_PENALTY; strong += RISK_OFF_PENALTY; watch += RISK_OFF_PENALTY
        notes.append(f"risk-off tape ({regime.label}) — thresholds raised")

    score = max(0.0, min(100.0, score))
    if score >= hi:
        tier = "HIGH CONVICTION"
    elif score >= strong:
        tier = "STRONG"
    elif score >= watch:
        tier = "WATCH"
    else:
        return None

    # Suggested stop: 1.5 ATR below the close, a volatility-aware level rather
    # than an arbitrary percentage.
    plan = build_trade_plan(tech)
    stop = plan.stop if plan else None
    entry = plan.entry if plan else tech.close
    shares, risk_usd, notional = position_size(entry, stop)

    return Signal(
        symbol=symbol, tier=tier, final_score=round(score, 1), tech_score=tech_score,
        price=tech.close, reasons=reasons, notes=notes, news=news, flow=flow,
        tech=tech, stop_hint=stop, shares=shares, risk_usd=risk_usd, notional=notional,
        plan=plan,
    )


def position_size(price: float, stop: float | None
                  ) -> tuple[int | None, float | None, float | None]:
    """Risk-based sizing: risk a fixed % of the account per trade.

    Shares are set so that being stopped out costs RISK_PCT of the account,
    which means a volatile name automatically gets a smaller position than a
    quiet one for the same dollar risk. The result is then capped at
    MAX_POSITION_PCT of the account so a very tight stop cannot imply an
    absurdly large position.

    Returns (shares, risk_usd, notional), or (None, None, None) when
    ACCOUNT_SIZE is unset or the inputs are unusable.
    """
    if ACCOUNT_SIZE <= 0 or not stop or price <= 0 or stop >= price:
        return None, None, None

    risk_per_share = price - stop
    risk_budget = ACCOUNT_SIZE * (RISK_PCT / 100.0)
    shares = int(risk_budget // risk_per_share)

    max_notional = ACCOUNT_SIZE * (MAX_POSITION_PCT / 100.0)
    if shares * price > max_notional:
        shares = int(max_notional // price)

    if shares < 1:
        return None, None, None
    return shares, round(shares * risk_per_share, 2), round(shares * price, 2)


# --------------------------------------------------------------------------
# Telegram
# --------------------------------------------------------------------------

def telegram_send(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("! Telegram credentials not set; printing instead:\n", file=sys.stderr)
        print(text)
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true",
    }).encode()
    try:
        req = urllib.request.Request(url, data=payload)
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode())
        if not body.get("ok"):
            print(f"! Telegram error: {body}", file=sys.stderr)
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"! Telegram send failed: {exc}", file=sys.stderr)
        return False


TIER_ICON = {"HIGH CONVICTION": "🟢", "STRONG": "🟡", "WATCH": "⚪"}


def format_message(signals: list[Signal], regime: Regime, intraday: bool = False,
                   stats: dict | None = None) -> str:
    stamp = datetime.now(timezone.utc).astimezone()
    today = stamp.strftime("%b %d, %Y %H:%M" if intraday else "%b %d, %Y")
    lines = [
        f"<b>Dip Scan{' · INTRADAY' if intraday else ''} — {today}</b>",
        f"<i>Market: {regime.label} · SPY {regime.spy_vs_200:+.1f}% vs 200d · "
        f"breadth {regime.breadth*100:.0f}%</i>",
        "",
    ]
    for s in signals:
        icon = TIER_ICON.get(s.tier, "•")
        lines.append(f"{icon} <b>{s.symbol}</b> — {s.tier} ({s.final_score:.0f}/100)")
        lines.append(f"   ${s.price:,.2f}")
        for r in s.reasons[:4]:
            lines.append(f"   • {r}")
        for n in s.notes[:3]:
            lines.append(f"   › {n}")
        if s.plan:
            pl = s.plan
            lines.append(f"   🎯 Buy ≤ <b>${pl.entry:,.2f}</b>  ·  "
                         f"Stop <b>${pl.stop:,.2f}</b> (−{pl.stop_pct:.1f}%)")
            lines.append(f"   💰 Target <b>${pl.target:,.2f}</b> (+{pl.target_pct:.1f}%)  ·  "
                         f"R:R <b>{pl.rr:.2f}</b>")
            lines.append(f"   ⏱ Est. hold ~{pl.hold_days_est}d  ·  "
                         f"ATR {pl.atr_pct:.1f}%/day  ·  vol {pl.ann_vol_pct:.0f}%/yr"
                         if pl.ann_vol_pct else
                         f"   ⏱ Est. hold ~{pl.hold_days_est}d  ·  ATR {pl.atr_pct:.1f}%/day")
        elif s.stop_hint:
            lines.append(f"   ⚠ 1.5·ATR stop ≈ ${s.stop_hint:,.2f}")
        if s.shares:
            # Report the risk actually taken, not the configured target: the
            # position cap can bind and cut it well below RISK_PCT.
            actual_pct = (s.risk_usd / ACCOUNT_SIZE * 100) if ACCOUNT_SIZE else 0
            capped = " capped" if actual_pct < RISK_PCT - 0.05 else ""
            lines.append(f"   📐 {s.shares:,} sh ≈ ${s.notional:,.0f} "
                         f"· risk ${s.risk_usd:,.0f} ({actual_pct:.2f}%{capped})")
        if s.news.headlines:
            lines.append(f"   📰 {s.news.headlines[0][:110]}")
        lines.append("")
    if stats:
        lines.append(paper_summary_line(stats))
    lines.append("<i>Signal output, not investment advice. Position size accordingly.</i>")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Paper trading -- the honest scoreboard
# --------------------------------------------------------------------------
# Every alert opens a simulated $100 position. Positions are resolved against
# real subsequent bars, so this measures what the alerts would actually have
# done rather than what the scoring model hoped. This is the only part of the
# tool that can tell you whether the signal is worth anything.

PAPER_TRADE_USD = env_float("PAPER_TRADE_USD", 100.0)

# --- Trading costs -----------------------------------------------------------
# Modelled per exit TYPE, because they do not slip alike:
#   entry  -- a LIMIT buy. You either get your price or you get nothing, and
#             "nothing" is already modelled as an unfilled order. No slippage.
#   target -- a LIMIT sell resting above the market. No slippage.
#   stop   -- a STOP order becomes a MARKET order when triggered, and fills
#             worse than the trigger. This is where essentially all slippage
#             lives, and it lands entirely on losing trades.
#   time   -- a market exit. Slips like any market order.
# Applying a flat cost to every leg would overstate drag on winners and
# understate the asymmetry that actually matters.
SLIPPAGE_BPS = env_float("SLIPPAGE_BPS", 5.0)      # basis points on market-type exits
COMMISSION_USD = env_float("COMMISSION_USD", 0.0)  # per round trip; most brokers are $0
PAPER_MAX_HOLD_DAYS = env_int("PAPER_MAX_HOLD_DAYS", 30)
TRADES_FILE = BASE_DIR / "trades.json"


def load_trades() -> dict:
    if TRADES_FILE.exists():
        try:
            d = json.loads(TRADES_FILE.read_text())
            d.setdefault("open", [])
            d.setdefault("closed", [])
            return d
        except json.JSONDecodeError:
            pass
    return {"open": [], "closed": []}


def save_trades(t: dict) -> None:
    TRADES_FILE.write_text(json.dumps(t, indent=2))


def open_paper_trade(book: dict, sig: Signal, intraday: bool,
                     alerted: bool = True) -> bool:
    """Record a simulated entry. One open position per symbol at a time."""
    if not sig.plan:
        return False
    if any(t["symbol"] == sig.symbol for t in book["open"]):
        return False        # already holding this name; don't pyramid
    pl = sig.plan
    book["open"].append({
        "symbol": sig.symbol,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "entry": pl.entry, "stop": pl.stop, "target": pl.target,
        "shares": round(PAPER_TRADE_USD / pl.entry, 6),
        "cost": PAPER_TRADE_USD,
        "tier": sig.tier, "score": sig.final_score,
        "mode": "intraday" if intraday else "eod",
        "alerted": alerted,
        "rr": pl.rr, "target_pct": pl.target_pct, "stop_pct": pl.stop_pct,
        "hold_days_est": pl.hold_days_est,
    })
    return True


def resolve_paper_trades(book: dict, bars_by_symbol: dict[str, list[Bar]]) -> list[dict]:
    """Close any open position whose stop, target or time limit was reached.

    Resolution uses the daily high/low of bars *after* entry. When a single bar
    touches both the stop and the target, the stop is assumed to have hit first:
    without intraday sequencing there is no way to know the order, and assuming
    the good outcome would flatter every result.
    """
    still_open, newly_closed = [], []
    now = datetime.now(timezone.utc)

    for tr in book["open"]:
        bars = bars_by_symbol.get(tr["symbol"])
        if not bars:
            still_open.append(tr)
            continue
        try:
            opened = datetime.fromisoformat(tr["opened_at"])
        except (ValueError, TypeError):
            opened = now
        open_date = opened.date().isoformat()

        exit_price, exit_reason, exit_date = None, None, None
        for b in bars:
            if b.date[:10] <= open_date:
                continue
            if b.low <= tr["stop"]:
                exit_price, exit_reason, exit_date = tr["stop"], "stop", b.date
                break
            if b.high >= tr["target"]:
                exit_price, exit_reason, exit_date = tr["target"], "target", b.date
                break

        if exit_price is None:
            held = (now - opened).days
            if held >= PAPER_MAX_HOLD_DAYS:
                exit_price = bars[-1].close
                exit_reason, exit_date = "time", bars[-1].date
            else:
                still_open.append(tr)
                continue

        pnl, cost = apply_costs(tr["entry"], exit_price, exit_reason, tr["shares"])
        closed = dict(tr)
        closed.update({
            "exit": round(exit_price, 2), "exit_reason": exit_reason,
            "exit_date": exit_date, "closed_at": now.isoformat(),
            "pnl": round(pnl, 2),
            "cost": tr.get("cost", PAPER_TRADE_USD),
            "costs_usd": round(cost, 4),
            "pnl_pct": round(pnl / (tr["entry"] * tr["shares"]) * 100, 2),
        })
        newly_closed.append(closed)

    book["open"] = still_open
    book["closed"].extend(newly_closed)
    return newly_closed


def apply_costs(entry: float, exit_price: float, exit_reason: str,
                shares: float) -> tuple[float, float]:
    """Return (net_pnl, cost_usd) for one round trip.

    Slippage applies only to market-type exits (stop, time). A limit entry and
    a limit target fill at their price or not at all.
    """
    slip = 0.0
    if exit_reason in ("stop", "time"):
        slip = exit_price * SLIPPAGE_BPS / 10_000
    realised_exit = exit_price - slip
    gross = (realised_exit - entry) * shares
    return gross - COMMISSION_USD, slip * shares + COMMISSION_USD


def paper_stats(book: dict) -> dict:
    closed = book.get("closed", [])
    wins = [t for t in closed if t.get("pnl", 0) > 0]
    losses = [t for t in closed if t.get("pnl", 0) <= 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    total = sum(t.get("pnl", 0) for t in closed)
    invested = sum(t.get("cost", PAPER_TRADE_USD) for t in closed)
    return {
        "trades": len(closed),
        "open": len(book.get("open", [])),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(closed) * 100) if closed else 0.0,
        "total_pnl": round(total, 2),
        "return_pct": round(total / invested * 100, 2) if invested else 0.0,
        # Profit factor: gross wins / gross losses. Above 1.0 is profitable.
        # With no losses yet it is undefined, not infinite -- report None.
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "avg_win": round(gross_win / len(wins), 2) if wins else 0.0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
        "best": max((t["pnl"] for t in closed), default=0.0),
        "worst": min((t["pnl"] for t in closed), default=0.0),
    }


def paper_summary_line(st: dict) -> str:
    if not st["trades"]:
        return f"📊 Paper: no closed trades yet ({st['open']} open)"
    pf = f"{st['profit_factor']:.2f}" if st["profit_factor"] is not None else "—"
    sign = "+" if st["total_pnl"] >= 0 else ""
    return (f"📊 Paper: {st['wins']}W/{st['losses']}L "
            f"({st['win_rate']:.0f}%) · PF {pf} · "
            f"{sign}${st['total_pnl']:.2f} ({sign}{st['return_pct']:.1f}%) · "
            f"{st['open']} open")


# --------------------------------------------------------------------------
# Dedupe state
# --------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def state_key(symbol: str, intraday: bool = False) -> str:
    """Intraday and end-of-day alerts dedupe independently.

    They answer different questions -- "is this a dip today" vs "is this a dip
    right now" -- so an intraday alert should not suppress the evening one.
    """
    return f"intraday:{symbol}" if intraday else symbol


def is_new_signal(state: dict, sig: Signal, intraday: bool = False) -> bool:
    """Suppress repeats of the same ticker+tier inside the resend window.

    An upgrade (WATCH -> STRONG) always sends: that is new information.
    Intraday uses a much shorter window, since the whole point is timeliness.
    """
    window_days = 0.25 if intraday else RESEND_AFTER_DAYS
    rank = {"WATCH": 1, "STRONG": 2, "HIGH CONVICTION": 3}
    prev = state.get(state_key(sig.symbol, intraday))
    if not prev:
        return True
    try:
        last = datetime.fromisoformat(prev["date"])
    except (KeyError, ValueError, TypeError):
        return True
    if rank.get(sig.tier, 0) > rank.get(prev.get("tier", ""), 0):
        return True
    elapsed_days = (datetime.now(timezone.utc) - last).total_seconds() / 86400
    return elapsed_days >= window_days


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def apply_tuned_params() -> dict:
    """Load any validated parameters written by `optimize --apply`."""
    global STOP_ATR_MULT, TARGET_ATR_MULT
    p = load_params()
    if "stop_mult" in p:
        STOP_ATR_MULT = clamp_param("stop_mult", float(p["stop_mult"]))
    if "target_mult" in p:
        TARGET_ATR_MULT = clamp_param("target_mult", float(p["target_mult"]))
    return p


def tuned_min_score(params: dict, intraday: bool) -> float:
    """The score an alert must clear, after tuning. Intraday keeps its own floor."""
    base = clamp_param("min_score", params.get("min_score", ALERT_MIN_SCORE))
    return max(base, INTRADAY_MIN_SCORE) if intraday else base


def cmd_scan(args) -> int:
    tuned = apply_tuned_params()
    if tuned:
        print(f"Using tuned params: stop {STOP_ATR_MULT:.1f}x ATR, "
              f"target {'20d mean' if not TARGET_ATR_MULT else f'{TARGET_ATR_MULT:.1f}x ATR'}"
              f" (validated {tuned.get('updated_at','?')[:10]}, "
              f"OOS {tuned.get('oos_expectancy','?')}%)")
    intraday = getattr(args, "intraday", False)
    symbols = args.symbols or (INTRADAY_WATCHLIST if intraday else WATCHLIST)
    mode = f"intraday {INTRADAY_INTERVAL}" if intraday else "end-of-day"

    # Intraday runs every 15 minutes all session. Outside market hours the bars
    # are stale, so exit before spending any credits.
    if intraday and not getattr(args, "ignore_market_hours", False):
        is_open, why = market_status()
        if not is_open:
            print(f"Market is closed ({why}) — skipping intraday scan.")
            return 0

    load_persisted_usage()
    print(f"Scanning {len(symbols)} symbols ({mode})...")

    tech_by_symbol: dict[str, Technicals] = {}
    bars_by_symbol: dict[str, list[Bar]] = {}
    for sym in symbols:
        try:
            bars = fetch_prices(sym)
        except DataError as exc:
            print(f"  ! {sym}: {exc}", file=sys.stderr)
            continue
        if len(bars) < 30:
            print(f"  ! {sym}: only {len(bars)} bars, skipping", file=sys.stderr)
            continue
        t = compute_technicals(bars)

        # A free key returns 100 daily bars, so sma200 is usually missing. Pull
        # the weekly series (free, decades deep, cached a week) for the 40-week
        # average that stands in for it.
        if t.trend_basis != "200d":
            try:
                t = attach_weekly_trend(t, fetch_weekly(sym))
            except DataError as exc:
                print(f"  ! {sym} weekly trend: {exc}", file=sys.stderr)

        if intraday:
            # Intraday oscillators, daily trend context. If the intraday leg
            # fails we fall back to the daily reading rather than dropping the
            # symbol -- a stale-but-valid signal beats a silent gap.
            try:
                ibars = fetch_intraday(sym)
                if len(ibars) >= 30:
                    t = blend_intraday_with_daily(compute_technicals(ibars), t)
                else:
                    print(f"  ! {sym}: only {len(ibars)} intraday bars, using daily",
                          file=sys.stderr)
            except DataError as exc:
                print(f"  ! {sym} intraday: {exc} — using daily", file=sys.stderr)

        bars_by_symbol[sym] = bars
        tech_by_symbol[sym] = t
        basis = {"200d": "200d MA", "40w": "40w MA", "none": "NO TREND DATA"}[t.trend_basis]
        print(f"  ✓ {sym}: ${t.close:,.2f}  [{len(bars)} daily bars, trend={basis}]")

    if not tech_by_symbol:
        print("No price data retrieved. Check ALPHAVANTAGE_API_KEY.", file=sys.stderr)
        return 1

    # The regime gate needs the benchmark. If the caller narrowed the watchlist
    # and left SPY out, fetch it anyway rather than silently running ungated.
    regime_input = dict(tech_by_symbol)
    if BENCHMARK not in regime_input:
        try:
            bench_bars = fetch_prices(BENCHMARK)
            if len(bench_bars) >= 30:
                bt = compute_technicals(bench_bars)
                if bt.trend_basis != "200d":
                    bt = attach_weekly_trend(bt, fetch_weekly(BENCHMARK))
                regime_input[BENCHMARK] = bt
        except DataError as exc:
            print(f"  ! benchmark {BENCHMARK} unavailable: {exc}", file=sys.stderr)

    # Settle any open paper positions against today's bars before scoring.
    book = load_trades()
    closed_now = resolve_paper_trades(book, bars_by_symbol)
    for c in closed_now:
        print(f"  → closed {c['symbol']}: {c['exit_reason']} "
              f"@ ${c['exit']:,.2f}  {c['pnl']:+.2f} ({c['pnl_pct']:+.1f}%)")

    regime = compute_regime(regime_input)
    if regime.label == "unknown":
        print("  ! regime unknown — running without the risk-off gate", file=sys.stderr)
    print(f"\nMarket regime: {regime.label} "
          f"(SPY {regime.spy_vs_200:+.1f}% vs 200d, breadth {regime.breadth*100:.0f}%)")

    news_by_symbol = fetch_news(list(tech_by_symbol))

    min_score = tuned_min_score(tuned, intraday)
    if tuned.get("halted"):
        print(f"  ! CIRCUIT BREAKER ACTIVE — only score ≥{min_score:.0f} alerts fire")
    # Alerting and tracking are deliberately separate. You are notified only
    # above `min_score`, but every qualifying setup can still be paper-traded --
    # which accumulates evidence roughly 3x faster and keeps measuring whether
    # the muted tiers really do underperform, at no cost in phone noise.
    all_signals: list[Signal] = []
    for sym, tech in tech_by_symbol.items():
        # Only spend SEC requests on names that already look technically interesting.
        prelim, _ = score_technicals(tech)
        flow = fetch_flow(sym) if (prelim >= TIER_WATCH - 10 and not args.no_flow) else FlowSignal()
        sig = evaluate(sym, tech, news_by_symbol.get(sym, NewsSignal()), flow, regime)
        if sig:
            all_signals.append(sig)

    all_signals.sort(key=lambda s: -s.final_score)
    signals = [s for s in all_signals if s.final_score >= min_score]
    muted = [s for s in all_signals if s.final_score < min_score]
    if muted:
        print(f"  ({len(muted)} setup(s) below the {min_score:.0f} alert threshold"
              f"{' — tracked silently' if TRACK_ALL_TIERS else ' — ignored'})")

    if TRACK_ALL_TIERS and not signals:
        for sgl in muted:
            if open_paper_trade(book, sgl, intraday, alerted=False):
                print(f"  → tracked (no alert): {sgl.symbol} @ ${sgl.plan.entry:,.2f}")
    save_trades(book)
    persist_usage()
    if not signals:
        print("\nNo qualifying dips today.")
        if args.always_send:
            telegram_send(f"<b>Dip Scan</b>\nNo qualifying setups today.\n"
                          f"<i>Market: {regime.label}</i>")
        return 0

    print(f"\n{len(signals)} signal(s):")
    for s in signals:
        print(f"  {s.tier:16s} {s.symbol:6s} {s.final_score:5.1f}  ${s.price:,.2f}")

    state = load_state()
    fresh = [s for s in signals if is_new_signal(state, s, intraday)]
    if not fresh:
        print(f"\nAll signals already alerted recently; nothing new to send.")
        print(av_budget_report())
        return 0

    alerted_syms = {s.symbol for s in fresh}
    to_track = (fresh + muted) if TRACK_ALL_TIERS else fresh
    for sgl in to_track:
        if open_paper_trade(book, sgl, intraday,
                            alerted=sgl.symbol in alerted_syms):
            tag = "" if sgl.symbol in alerted_syms else "  [tracked only, not alerted]"
            print(f"  → paper trade opened: {sgl.symbol} "
                  f"${PAPER_TRADE_USD:.0f} @ ${sgl.plan.entry:,.2f}{tag}")
    save_trades(book)

    message = format_message(fresh, regime, intraday=intraday,
                             stats=paper_stats(book))
    if args.dry_run:
        print("\n--- DRY RUN, message not sent ---\n")
        print(message)
        print(f"\n{av_budget_report()}")
        return 0

    if telegram_send(message):
        print(f"\n✓ Alert sent for {len(fresh)} signal(s).")
        now = datetime.now(timezone.utc).isoformat()
        for s in fresh:
            state[state_key(s.symbol, intraday)] = {
                "date": now, "tier": s.tier, "score": s.final_score,
            }
        save_state(state)
    persist_usage()
    print(av_budget_report())
    return 0


def cmd_history(args) -> int:
    """Print the paper-trading log and performance summary."""
    book = load_trades()
    st = paper_stats(book)
    closed = sorted(book["closed"], key=lambda t: t.get("closed_at", ""), reverse=True)
    if args.limit:
        closed = closed[: args.limit]

    if book["open"]:
        print(f"\nOPEN POSITIONS ({len(book['open'])})")
        print(f"{'SYM':<7}{'OPENED':<12}{'ENTRY':>9}{'STOP':>9}{'TARGET':>9}"
              f"{'R:R':>6}  TIER")
        print("-" * 66)
        for t in sorted(book["open"], key=lambda x: x["opened_at"]):
            print(f"{t['symbol']:<7}{t['opened_at'][:10]:<12}"
                  f"{t['entry']:>9,.2f}{t['stop']:>9,.2f}{t['target']:>9,.2f}"
                  f"{t.get('rr', 0):>6.2f}  {t.get('tier', '')}")

    if closed:
        print(f"\nCLOSED TRADES ({len(closed)} shown of {st['trades']})")
        print(f"{'SYM':<7}{'OPENED':<12}{'CLOSED':<12}{'ENTRY':>9}{'EXIT':>9}"
              f"{'P/L':>9}{'%':>8}  WHY")
        print("-" * 82)
        for t in closed:
            print(f"{t['symbol']:<7}{t['opened_at'][:10]:<12}"
                  f"{str(t.get('closed_at', ''))[:10]:<12}"
                  f"{t['entry']:>9,.2f}{t.get('exit', 0):>9,.2f}"
                  f"{t.get('pnl', 0):>+9.2f}{t.get('pnl_pct', 0):>+8.1f}"
                  f"  {t.get('exit_reason', '')}")

    if not book["open"] and not closed:
        print("\nNo paper trades recorded yet. They open automatically on each alert.")
        if args.telegram:
            telegram_send(
                "<b>Paper Trading Report</b>\nNo trades recorded yet.\n\n"
                "Positions open automatically when a setup qualifies. With "
                f"ALERT_MIN_SCORE={ALERT_MIN_SCORE:.0f} expect roughly one every "
                "few trading days.")
        return 0

    def _bucket(rows: list[dict]) -> str:
        if not rows:
            return f"{'—':>6}{'':>9}{'':>9}{'':>9}"
        wins = [r for r in rows if r.get("pnl", 0) > 0]
        gw = sum(r["pnl"] for r in wins)
        gl = abs(sum(r["pnl"] for r in rows if r.get("pnl", 0) <= 0))
        avg = sum(r.get("pnl_pct", 0) for r in rows) / len(rows)
        pf_ = f"{gw/gl:.2f}" if gl > 0 else "n/a"
        return (f"{len(rows):>6}{len(wins)/len(rows)*100:>8.1f}%"
                f"{avg:>9.2f}%{pf_:>9}")

    allc = book["closed"]
    if allc:
        print(f"\n{'='*66}")
        print("BY TIER  (all tracked setups, whether or not they alerted)")
        print(f"{'TIER':<18}{'N':>6}{'WIN%':>9}{'AVG%':>9}{'PF':>9}")
        print("-" * 66)
        for tier in ("WATCH", "STRONG", "HIGH CONVICTION"):
            print(f"{tier:<18}" + _bucket([r for r in allc if r.get("tier") == tier]))
        alerted = [r for r in allc if r.get("alerted", True)]
        silent = [r for r in allc if not r.get("alerted", True)]
        if silent:
            print("-" * 66)
            print(f"{'alerted to you':<18}" + _bucket(alerted))
            print(f"{'tracked silently':<18}" + _bucket(silent))
            print("\n  If the silent rows keep underperforming, the alert threshold")
            print("  is doing its job. If they catch up, reconsider lowering it.")

    pf = f"{st['profit_factor']:.2f}" if st["profit_factor"] is not None else "n/a"
    print(f"\n{'='*66}")
    print(f"PAPER PERFORMANCE  (${PAPER_TRADE_USD:.0f} simulated per alert)")
    print(f"{'='*66}")
    print(f"  Closed trades   {st['trades']}          Open  {st['open']}")
    print(f"  Win / loss      {st['wins']}W / {st['losses']}L  ({st['win_rate']:.1f}% win rate)")
    print(f"  Total P/L       ${st['total_pnl']:+,.2f}  ({st['return_pct']:+.2f}% on capital deployed)")
    print(f"  Profit factor   {pf}   (gross wins / gross losses; >1.0 is profitable)")
    print(f"  Avg win/loss    ${st['avg_win']:+,.2f} / ${st['avg_loss']:+,.2f}")
    print(f"  Best / worst    ${st['best']:+,.2f} / ${st['worst']:+,.2f}")
    print(f"{'='*66}")
    if st["trades"] < 30:
        print("! Fewer than 30 closed trades — too small a sample to conclude")
        print("  anything. Treat these numbers as provisional.")

    if args.telegram:
        rows = "\n".join(
            f"{t['symbol']} {t.get('pnl',0):+.2f} ({t.get('pnl_pct',0):+.1f}%) {t.get('exit_reason','')}"
            for t in closed[:10])
        telegram_send(
            f"<b>Paper Trading Report</b>\n{paper_summary_line(st)}\n\n"
            f"<b>Recent closes</b>\n<pre>{rows or 'none yet'}</pre>"
        )
    return 0


def cmd_test_telegram(args) -> int:
    ok = telegram_send(
        "<b>Dip Signal Tool</b>\nConnection test successful. "
        "Alerts will arrive here after each scan."
    )
    print("✓ Sent." if ok else "✗ Failed — check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
    return 0 if ok else 1


def cmd_refresh_flow(args) -> int:
    for sym in (args.symbols or WATCHLIST):
        f = fetch_flow(sym)
        status = f.direction if f.available else "unavailable"
        print(f"  {sym:6s} {status:12s} net ${f.net_usd/1e6:+.2f}M "
              f"({f.buy_count} buys / {f.sell_count} sells)")
    return 0


def simulate_trade(bars: list[Bar], signal_idx: int, plan: TradePlan,
                   max_hold: int = 30, fill_window: int = 3) -> dict | None:
    """Run one trade plan forward through real bars.

    Uses exactly the rules the paper tracker uses, so backtest and live results
    are directly comparable:
      * The entry is a LIMIT below the signal bar's close. If no bar within
        `fill_window` trades down to it, there is no trade at all -- counting
        unfilled limits as wins is a classic way to fake a backtest.
      * Stop is checked before target on every bar, including the fill bar.
      * Anything still open after `max_hold` bars exits at the close.
    """
    fill_idx = None
    for j in range(signal_idx + 1, min(signal_idx + 1 + fill_window, len(bars))):
        if bars[j].low <= plan.entry:
            fill_idx = j
            break
    if fill_idx is None:
        return None                      # limit never filled

    def _result(reason: str, price: float, days: int) -> dict:
        # Same cost model the live paper tracker uses, on one notional share,
        # so backtest and live results stay directly comparable.
        net, cost = apply_costs(plan.entry, price, reason, 1.0)
        return {"exit_reason": reason, "exit": price,
                "pnl_pct": net / plan.entry * 100,
                "gross_pct": (price - plan.entry) / plan.entry * 100,
                "cost_pct": cost / plan.entry * 100,
                "days": days}

    for k in range(fill_idx, min(fill_idx + max_hold + 1, len(bars))):
        b = bars[k]
        if b.low <= plan.stop:           # stop always checked first
            return _result("stop", plan.stop, k - fill_idx)
        if b.high >= plan.target:
            return _result("target", plan.target, k - fill_idx)

    last = bars[min(fill_idx + max_hold, len(bars) - 1)]
    return _result("time", last.close, min(max_hold, len(bars) - 1 - fill_idx))


def _agg(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0}
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    gw = sum(t["pnl_pct"] for t in wins)
    gl = abs(sum(t["pnl_pct"] for t in losses))
    return {
        "n": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "avg": sum(t["pnl_pct"] for t in trades) / len(trades),
        "avg_win": (gw / len(wins)) if wins else 0.0,
        "avg_loss": (-gl / len(losses)) if losses else 0.0,
        "pf": (gw / gl) if gl > 0 else None,
        "avg_days": sum(t["days"] for t in trades) / len(trades),
        "targets": sum(1 for t in trades if t["exit_reason"] == "target"),
        "stops": sum(1 for t in trades if t["exit_reason"] == "stop"),
        "times": sum(1 for t in trades if t["exit_reason"] == "time"),
    }


def _fmt_row(label: str, a: dict) -> str:
    if not a["n"]:
        return f"  {label:<18} no trades"
    pf = f"{a['pf']:.2f}" if a["pf"] is not None else " n/a"
    return (f"  {label:<18} n={a['n']:<5} win {a['win_rate']:5.1f}%  "
            f"avg {a['avg']:+6.2f}%  PF {pf:>5}  hold {a['avg_days']:4.1f}d")


PARAMS_FILE = BASE_DIR / "params.json"

# Bounds for anything auto-tuning is allowed to change. A self-adjusting system
# without bounds will happily walk itself somewhere absurd on a run of noise.
PARAM_BOUNDS = {
    "stop_mult": (1.0, 3.0),
    "target_mult": (0.0, 6.0),
    "min_score": (40.0, 80.0),
}
MIN_TRADES_TO_TUNE = env_int("MIN_TRADES_TO_TUNE", 50)


def load_params() -> dict:
    if PARAMS_FILE.exists():
        try:
            return json.loads(PARAMS_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def save_params(p: dict) -> None:
    PARAMS_FILE.write_text(json.dumps(p, indent=2))


def clamp_param(name: str, value: float) -> float:
    lo, hi = PARAM_BOUNDS[name]
    return max(lo, min(hi, value))


def _collect_trades(bars: list[Bar], start_idx: int, end_idx: int,
                    stop_mult: float, target_mult: float, min_score: float,
                    max_hold: int) -> list[dict]:
    out = []
    for i in range(start_idx, end_idx):
        window = bars[max(0, i - 259): i + 1]
        t = compute_technicals(window)
        sc, _ = score_technicals(t)
        if sc < min_score:
            continue
        plan = build_trade_plan(t, stop_mult=stop_mult, target_mult=target_mult)
        if not plan:
            continue
        r = simulate_trade(bars, i, plan, max_hold)
        if r:
            out.append(r)
    return out


def cmd_autotune(args) -> int:
    """Adjust selectivity from realised paper results, within hard limits.

    This deliberately tunes only ONE thing -- how high the score has to be
    before an alert fires. Trade geometry is not touched here: changing stops
    and targets from a few dozen live trades is how a working strategy gets
    curve-fitted into a broken one. Geometry changes go through `optimize`,
    which validates on data it never saw.

    Guards, all of which matter more than the tuning itself:
      * Nothing happens below MIN_TRADES_TO_TUNE closed trades.
      * Steps are small and bounded; one bad run cannot move it far.
      * A circuit breaker halts trading outright if results go badly negative,
        because "adapt harder" is the wrong response to a losing edge.
    """
    book = load_trades()
    closed = book.get("closed", [])
    params = load_params()
    current = clamp_param("min_score", params.get("min_score", float(TIER_WATCH)))

    recent = sorted(closed, key=lambda t: t.get("closed_at", ""))[-args.window:]
    st = paper_stats({"open": book.get("open", []), "closed": recent})

    print(f"Reviewing the last {len(recent)} closed trades "
          f"(need {MIN_TRADES_TO_TUNE} to act)")
    print(f"  {paper_summary_line(st)}")
    print(f"  Current minimum score: {current:.0f}")

    if len(recent) < MIN_TRADES_TO_TUNE:
        print(f"\n  Too few trades to tune on. {MIN_TRADES_TO_TUNE - len(recent)} more "
              f"needed.\n  Tuning on a small sample fits noise, not signal.")
        return 0

    pf = st["profit_factor"]
    new, reason, halt = current, None, False

    if pf is None:
        reason = "no losing trades yet — nothing to correct"
    elif pf < 0.7:
        new, halt = clamp_param("min_score", 80.0), True
        reason = (f"CIRCUIT BREAKER: profit factor {pf:.2f} over {len(recent)} trades. "
                  f"The edge is not working. Selectivity forced to maximum.")
    elif pf < 1.0:
        new = clamp_param("min_score", current + 5)
        reason = f"profit factor {pf:.2f} is below 1.0 — becoming more selective"
    elif pf > 1.5 and st["win_rate"] > 45:
        new = clamp_param("min_score", current - 2.5)
        reason = f"profit factor {pf:.2f} is healthy — loosening slightly to take more setups"
    else:
        reason = f"profit factor {pf:.2f} is acceptable — leaving settings alone"

    print(f"\n  {reason}")
    if new != current:
        print(f"  Minimum score {current:.0f} → {new:.0f}")
    else:
        print(f"  No change.")

    if args.apply and new != current:
        history = params.get("tuning_log", [])
        history.append({"at": datetime.now(timezone.utc).isoformat(),
                        "from": current, "to": new, "reason": reason,
                        "pf": pf, "trades": len(recent)})
        params.update({"min_score": new, "tuning_log": history[-40:],
                       "halted": halt})
        save_params(params)
        print(f"  ✓ Applied to params.json")
        if halt:
            telegram_send(
                f"<b>⛔ Circuit breaker tripped</b>\nProfit factor "
                f"{pf:.2f} over {len(recent)} paper trades. Alerts restricted to "
                f"the strongest setups only.\n\nThe strategy is not working as "
                f"configured. Re-run <code>optimize</code> before trusting it again.")
    elif new != current:
        print(f"  (dry run — pass --apply to make the change)")
    return 0


def cmd_optimize(args) -> int:
    """Sweep trade geometry and report what actually maximises expectancy.

    Split in two: the sweep picks a winner on the OLDER 70% of history
    (in-sample), then that winner is measured on the newer 30% it never saw
    (out-of-sample). A setting that looks brilliant in-sample and collapses
    out-of-sample is curve-fitted, not skilful, and this prints both so the
    difference is impossible to miss.
    """
    symbols = args.symbols or WATCHLIST
    load_persisted_usage()
    max_hold = args.max_hold

    series: dict[str, tuple[list[Bar], int, int, int]] = {}
    for sym in symbols:
        try:
            bars = fetch_prices(sym, full=True, max_age_hours=168)
        except DataError as exc:
            print(f"  ! {sym}: {exc}", file=sys.stderr)
            continue
        if len(bars) < 220 + max_hold + 100:
            print(f"  ! {sym}: only {len(bars)} bars, skipping", file=sys.stderr)
            continue
        cutoff = datetime.now().date() - timedelta(days=int(365.25 * args.years))
        s0 = max(next((i for i, b in enumerate(bars) if b.date[:10] >= cutoff.isoformat()), 0), 220)
        last = len(bars) - max_hold - 1
        if last - s0 < 200:
            continue
        split = s0 + int((last - s0) * 0.7)
        series[sym] = (bars, s0, split, last)

    if not series:
        print("No symbol had enough history to optimise on.", file=sys.stderr)
        print("A free Alpha Vantage key returns 100 daily bars — add a Twelve Data key.",
              file=sys.stderr)
        return 1

    stop_grid = [1.0, 1.5, 2.0, 2.5, 3.0]
    target_grid = [0.0, 1.5, 2.0, 3.0, 4.0, 6.0]   # 0 = revert to the 20-day mean
    score_grid = [args.min_score] if args.min_score else [40.0, 55.0, 70.0]

    print(f"\nSweeping {len(stop_grid)*len(target_grid)*len(score_grid)} combinations "
          f"across {len(series)} symbols...\n")
    print(f"{'STOP':>5}{'TARGET':>8}{'SCORE':>7}{'N':>6}{'WIN%':>7}"
          f"{'AVG%':>8}{'PF':>7}{'HOLD':>6}   {'EXPECTANCY':>10}")
    print("-" * 72)

    results = []
    for sm in stop_grid:
        for tm in target_grid:
            for ms in score_grid:
                ins = []
                for bars, s0, split, last in series.values():
                    ins += _collect_trades(bars, s0, split, sm, tm, ms, max_hold)
                a = _agg(ins)
                if a["n"] < 20:
                    continue
                results.append({"stop_mult": sm, "target_mult": tm, "min_score": ms,
                                "is": a})
                pf = f"{a['pf']:.2f}" if a["pf"] is not None else "  n/a"
                tlabel = "mean" if tm == 0 else f"{tm:.1f}x"
                print(f"{sm:>5.1f}{tlabel:>8}{ms:>7.0f}{a['n']:>6}"
                      f"{a['win_rate']:>7.1f}{a['avg']:>8.2f}{pf:>7}"
                      f"{a['avg_days']:>6.1f}   {a['avg']:>9.2f}%")

    if not results:
        print("\nNo combination produced enough trades to judge.")
        return 1

    results.sort(key=lambda r: -r["is"]["avg"])
    best = results[0]

    # Validate the winner on data the sweep never touched.
    oos = []
    for bars, s0, split, last in series.values():
        oos += _collect_trades(bars, split, last, best["stop_mult"],
                               best["target_mult"], best["min_score"], max_hold)
    oa = _agg(oos)

    # And the current live settings, for comparison on the same OOS window.
    cur = load_params()
    cur_sm = cur.get("stop_mult", STOP_ATR_MULT)
    cur_tm = cur.get("target_mult", TARGET_ATR_MULT)
    cur_ms = cur.get("min_score", TIER_WATCH)
    cur_oos = []
    for bars, s0, split, last in series.values():
        cur_oos += _collect_trades(bars, split, last, cur_sm, cur_tm, cur_ms, max_hold)
    ca = _agg(cur_oos)

    tlabel = "20d mean" if best["target_mult"] == 0 else f"{best['target_mult']:.1f}x ATR"
    print(f"\n{'='*72}")
    print("BEST IN-SAMPLE")
    print(f"  stop {best['stop_mult']:.1f}x ATR · target {tlabel} · min score {best['min_score']:.0f}")
    print(f"  in-sample      {_fmt_row('', best['is']).strip()}")
    print(f"  OUT-OF-SAMPLE  {_fmt_row('', oa).strip()}")
    print(f"\nCURRENT SETTINGS on the same out-of-sample window")
    print(f"  stop {cur_sm:.1f}x ATR · target "
          f"{'20d mean' if not cur_tm else f'{cur_tm:.1f}x ATR'} · min score {cur_ms:.0f}")
    print(f"  OUT-OF-SAMPLE  {_fmt_row('', ca).strip()}")
    print(f"{'='*72}")

    verdict, adopt = [], False
    if not oa["n"] or oa["n"] < 20:
        verdict.append("Out-of-sample sample too small to trust. Not adopting.")
    elif oa["avg"] <= 0:
        verdict.append("The best in-sample setting LOSES money out-of-sample.")
        verdict.append("That is curve-fitting. Not adopting.")
    elif best["is"]["avg"] > 0 and oa["avg"] < best["is"]["avg"] * 0.5:
        verdict.append(f"Out-of-sample expectancy ({oa['avg']:+.2f}%) is less than half "
                       f"in-sample ({best['is']['avg']:+.2f}%).")
        verdict.append("That degradation is the signature of overfitting. Not adopting.")
    elif ca["n"] and oa["avg"] <= ca["avg"]:
        verdict.append(f"No better than current settings out-of-sample "
                       f"({oa['avg']:+.2f}% vs {ca['avg']:+.2f}%). Keeping what you have.")
    else:
        adopt = True
        verdict.append(f"Out-of-sample improvement: {oa['avg']:+.2f}% vs "
                       f"{ca['avg']:+.2f}% per trade.")
    for line in verdict:
        print("  " + line)

    if adopt and args.apply:
        cur.update({"stop_mult": clamp_param("stop_mult", best["stop_mult"]),
                    "target_mult": clamp_param("target_mult", best["target_mult"]),
                    "min_score": clamp_param("min_score", best["min_score"]),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "oos_expectancy": round(oa["avg"], 3),
                    "oos_trades": oa["n"]})
        save_params(cur)
        print(f"\n  ✓ Applied to params.json — future scans use these.")
    elif args.apply:
        print(f"\n  Nothing applied. Current settings retained.")
    else:
        print(f"\n  (dry run — pass --apply to adopt a validated improvement)")

    persist_usage()
    print(av_budget_report())
    return 0


def cmd_backtest(args) -> int:
    """Replay the strategy over history using the real trade plan.

    Only the technical layer is tested. News sentiment and Form 4 history are
    not available point-in-time from these endpoints, so including them would
    leak lookahead bias and make the results look better than they are.
    """
    symbols = args.symbols or WATCHLIST
    load_persisted_usage()
    max_hold = args.max_hold

    by_tier: dict[str, list[dict]] = {"WATCH": [], "STRONG": [], "HIGH CONVICTION": []}
    baseline: list[dict] = []
    insufficient, signals_seen, unfilled = [], 0, 0

    for sym in symbols:
        try:
            bars = fetch_prices(sym, full=True, max_age_hours=168)
        except DataError as exc:
            print(f"  ! {sym}: full history unavailable ({exc})", file=sys.stderr)
            try:
                bars = fetch_prices(sym)
            except DataError as exc2:
                print(f"  ! {sym}: {exc2}", file=sys.stderr)
                continue

        need = 220 + max_hold
        if len(bars) < need:
            print(f"  ! {sym}: only {len(bars)} bars — need ~{need}. Skipping.",
                  file=sys.stderr)
            insufficient.append(sym)
            continue

        cutoff = datetime.now().date() - timedelta(days=int(365.25 * args.years))
        start_idx = next((i for i, b in enumerate(bars)
                          if b.date[:10] >= cutoff.isoformat()), 0)
        start_idx = max(start_idx, 220)          # 200d MA warm-up
        sym_trades = 0

        for i in range(start_idx, len(bars) - max_hold - 1):
            # A 260-bar window is all any indicator here needs, and keeps this
            # loop linear instead of quadratic over decades of history.
            window = bars[max(0, i - 259): i + 1]
            t = compute_technicals(window)
            sc, _ = score_technicals(t)

            # Baseline: identical trade mechanics, no signal required. This is
            # the number every tier has to beat to be worth anything.
            if i % 5 == 0:
                bp = build_trade_plan(t)
                if bp:
                    r = simulate_trade(bars, i, bp, max_hold)
                    if r:
                        baseline.append(r)

            tier = ("HIGH CONVICTION" if sc >= TIER_HIGH
                    else "STRONG" if sc >= TIER_STRONG
                    else "WATCH" if sc >= TIER_WATCH else None)
            if not tier:
                continue
            signals_seen += 1
            plan = build_trade_plan(t)
            if not plan:
                continue
            res = simulate_trade(bars, i, plan, max_hold)
            if res is None:
                unfilled += 1
                continue
            res["symbol"] = sym
            by_tier[tier].append(res)
            sym_trades += 1

        print(f"  {sym}: {sym_trades} trades over {args.years}y")

    print(f"\n{'='*72}")
    print(f"BACKTEST — technical layer only, {args.years}y, "
          f"{max_hold}d max hold, ${PAPER_TRADE_USD:.0f}/trade")
    print(f"{'='*72}")
    b = _agg(baseline)
    print(_fmt_row("BASELINE", b))
    print("  (same entry/stop/target mechanics, but every 5th bar, no signal required)")
    print()
    for tier in ("WATCH", "STRONG", "HIGH CONVICTION"):
        print(_fmt_row(tier, _agg(by_tier[tier])))

    all_sig = by_tier["WATCH"] + by_tier["STRONG"] + by_tier["HIGH CONVICTION"]
    a = _agg(all_sig)
    print()
    print(_fmt_row("ALL SIGNALS", a))
    if signals_seen:
        print(f"\n  Limit orders never filled: {unfilled} of {signals_seen} signals "
              f"({unfilled / signals_seen * 100:.0f}%)")
    if a["n"]:
        print(f"  Exits — target {a['targets']}, stop {a['stops']}, time {a['times']}")
        print(f"  Actual average hold: {a['avg_days']:.1f} days")

    print(f"\n{'='*72}")
    verdict = []
    if not b["n"] or not a["n"]:
        verdict.append("Not enough data to judge. Get more history before trusting this.")
    else:
        edge = a["avg"] - b["avg"]
        verdict.append(f"Edge over baseline: {edge:+.2f}% per trade.")
        if edge <= 0:
            verdict.append("NEGATIVE EDGE — the signal is not beating random entry.")
            verdict.append("Do not trade these thresholds. They need retuning.")
        elif a["n"] < 30:
            verdict.append(f"Only {a['n']} trades — too small to conclude anything.")
        elif a["pf"] is not None and a["pf"] < 1.2:
            verdict.append("Positive but thin. Slippage and fees could erase it.")
        else:
            verdict.append("Positive edge on this sample. Still a backtest, not a promise:")
            verdict.append("it assumes limit fills, and ignores slippage and commission.")
    for line in verdict:
        print("  " + line)
    if insufficient:
        print(f"\n  ! Skipped for lack of history: {', '.join(insufficient)}")
        print("    A free Alpha Vantage key returns 100 daily bars. Add a Twelve")
        print("    Data key (free) for the deep history a backtest needs.")
    print(f"{'='*72}")

    persist_usage()
    print(av_budget_report())

    if args.telegram:
        rows = "\n".join(_fmt_row(t, _agg(by_tier[t])).strip()
                          for t in ("WATCH", "STRONG", "HIGH CONVICTION"))
        telegram_send(
            f"<b>Backtest — {args.years}y</b>\n<pre>{_fmt_row('BASELINE', b).strip()}\n"
            f"{rows}\n{_fmt_row('ALL', a).strip()}</pre>\n"
            + "\n".join(verdict))
    return 0



def main() -> int:
    p = argparse.ArgumentParser(description="Dip signal scanner with Telegram alerts")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="run a scan and alert on new signals")
    s.add_argument("--symbols", nargs="*", help="override the watchlist")
    s.add_argument("--dry-run", action="store_true", help="print instead of sending")
    s.add_argument("--no-flow", action="store_true", help="skip SEC insider lookups")
    s.add_argument("--always-send", action="store_true", help="send even when nothing fires")
    s.add_argument("--intraday", action="store_true",
                   help=f"use {INTRADAY_INTERVAL} bars with daily trend context")
    s.add_argument("--ignore-market-hours", action="store_true",
                   help="run an intraday scan even when the market is closed")
    s.set_defaults(func=cmd_scan)

    b = sub.add_parser("backtest", help="validate the scoring rules on history")
    b.add_argument("--years", type=float, default=3.0)
    b.add_argument("--symbols", nargs="*")
    b.add_argument("--max-hold", type=int, default=PAPER_MAX_HOLD_DAYS,
                   help="bars to hold before exiting at the close")
    b.add_argument("--telegram", action="store_true", help="send a summary")
    b.set_defaults(func=cmd_backtest)

    h = sub.add_parser("history", help="show the paper-trading log and performance")
    h.add_argument("--limit", type=int, default=40, help="closed trades to show")
    h.add_argument("--telegram", action="store_true", help="also send a summary")
    h.set_defaults(func=cmd_history)

    at = sub.add_parser("autotune", help="adjust selectivity from realised results")
    at.add_argument("--window", type=int, default=100,
                    help="most recent closed trades to judge on")
    at.add_argument("--apply", action="store_true", help="write the change")
    at.set_defaults(func=cmd_autotune)

    o = sub.add_parser("optimize", help="sweep trade geometry with out-of-sample validation")
    o.add_argument("--years", type=float, default=5.0)
    o.add_argument("--symbols", nargs="*")
    o.add_argument("--max-hold", type=int, default=PAPER_MAX_HOLD_DAYS)
    o.add_argument("--min-score", type=float, default=None,
                   help="fix the score threshold instead of sweeping it")
    o.add_argument("--apply", action="store_true",
                   help="adopt the winner IF it validates out-of-sample")
    o.set_defaults(func=cmd_optimize)

    t = sub.add_parser("test-telegram", help="verify bot credentials")
    t.set_defaults(func=cmd_test_telegram)

    f = sub.add_parser("refresh-flow", help="rebuild the insider-flow cache")
    f.add_argument("--symbols", nargs="*")
    f.set_defaults(func=cmd_refresh_flow)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

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
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
STATE_FILE = BASE_DIR / "state.json"

ALPHA_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# SEC requires a descriptive User-Agent with contact info on every request.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "DipSignalTool contact@example.com")

INDICES = ["SPY", "QQQ", "IWM"]
MEGACAPS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO"]
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "XLC"]
WATCHLIST = INDICES + MEGACAPS + SECTORS          # 21 symbols
BENCHMARK = "SPY"

# Intraday is request-hungry, so it runs against a deliberately short list.
# Free-tier arithmetic: the daily scan spends 22 of 25 requests, leaving 3.
# Each intraday symbol costs 1 (its daily bars come from the cache the EOD scan
# already filled). Widen this only with a premium key -- see README.
INTRADAY_WATCHLIST = ["SPY", "QQQ", "NVDA"]
INTRADAY_INTERVAL = "15min"

# --- Alpha Vantage request budget -------------------------------------------
# Free tier allows 25 requests/day. A daily scan costs 21 price calls + 1
# batched news call = 22, which fits. Anything more must come out of the same
# pot, so requests are counted and hard-capped rather than left to fail
# mid-scan with a confusing rate-limit message.
AV_DAILY_LIMIT = int(os.environ.get("AV_DAILY_LIMIT", "25"))
AV_RATE_SLEEP = 1.0  # polite pacing between calls
_av_requests_used = 0


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
    return f"{_av_requests_used}/{AV_DAILY_LIMIT} Alpha Vantage requests used"


# --- Position sizing ---------------------------------------------------------
# Set ACCOUNT_SIZE to enable share-count suggestions in alerts.
ACCOUNT_SIZE = float(os.environ.get("ACCOUNT_SIZE", "0") or 0)
RISK_PCT = float(os.environ.get("RISK_PCT", "1.0") or 1.0)      # % of account risked per trade
MAX_POSITION_PCT = float(os.environ.get("MAX_POSITION_PCT", "20") or 20)  # cap per name

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


def fetch_prices(symbol: str, full: bool = True, max_age_hours: float = 20) -> list[Bar]:
    """Daily OHLCV, oldest first.

    Defaults to the full series: the 200-day moving average is the model's
    primary falling-knife guard, and a 100-bar "compact" response cannot
    produce one. Costs the same single API request either way.
    """
    size = "full" if full else "compact"

    def loader() -> dict:
        if not ALPHA_KEY:
            raise DataError("ALPHAVANTAGE_API_KEY is not set")
        av_budget_check()
        url = (
            "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
            f"&symbol={urllib.parse.quote(symbol)}&outputsize={size}&apikey={ALPHA_KEY}"
        )
        data = http_get_json(url)
        if "Time Series (Daily)" not in data:
            note = data.get("Note") or data.get("Information") or data.get("Error Message")
            raise DataError(f"{symbol}: {note or 'unexpected response'}")
        time.sleep(AV_RATE_SLEEP)
        return data

    raw = cached_json(f"px_{symbol}_{size}.json", max_age_hours, loader)
    series = raw["Time Series (Daily)"]
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


def fetch_intraday(symbol: str, interval: str = INTRADAY_INTERVAL,
                   max_age_hours: float = 0.4) -> list[Bar]:
    """Intraday OHLCV bars, oldest first.

    Cached only ~25 minutes: the whole point of intraday is freshness. Note
    this is a *separate* request from the daily series and comes out of the
    same free-tier budget -- see the scan-cadence note in the README.
    """
    def loader() -> dict:
        if not ALPHA_KEY:
            raise DataError("ALPHAVANTAGE_API_KEY is not set")
        av_budget_check()
        url = (
            "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
            f"&symbol={urllib.parse.quote(symbol)}&interval={interval}"
            f"&outputsize=full&apikey={ALPHA_KEY}"
        )
        data = http_get_json(url)
        if f"Time Series ({interval})" not in data:
            note = data.get("Note") or data.get("Information") or data.get("Error Message")
            raise DataError(f"{symbol} intraday: {note or 'unexpected response'}")
        time.sleep(AV_RATE_SLEEP)
        return data

    raw = cached_json(f"intraday_{symbol}_{interval}.json", max_age_hours, loader)
    series = raw[f"Time Series ({interval})"]
    bars = [
        Bar(
            date=d,
            open=float(v["1. open"]), high=float(v["2. high"]),
            low=float(v["3. low"]), close=float(v["4. close"]),
            volume=float(v["5. volume"]),
        )
        for d, v in series.items()
    ]
    bars.sort(key=lambda b: b.date)
    return bars


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
    )


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
    if t.sma200 is not None:
        rel = (t.close - t.sma200) / t.sma200 * 100
        if rel > 0:
            score += W_TREND; reasons.append("above 200d MA (uptrend intact)")
        elif rel > -3:
            score += W_TREND * 0.47; reasons.append("just below 200d MA")
        else:
            reasons.append(f"{abs(rel):.1f}% below 200d MA (downtrend)")

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
    if not bench or bench.sma200 is None:
        return Regime()
    spy_vs_200 = (bench.close - bench.sma200) / bench.sma200 * 100

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
    broken_trend = (
        tech.sma200 is not None
        and tech.close < tech.sma200
        and (tech.dd_from_high50 or 0) > 20
    )
    if broken_trend:
        rescued = news.available and news.score >= 0.15
        if not rescued:
            why = "no positive news to offset" if news.available else "news unavailable"
            notes.append(f"VETO: {tech.dd_from_high50:.0f}% off 50d high and below "
                         f"200d MA — {why} (falling knife)")
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
    stop = None
    if tech.atr_pct:
        stop = round(tech.close * (1 - 1.5 * tech.atr_pct / 100), 2)

    shares, risk_usd, notional = position_size(tech.close, stop)

    return Signal(
        symbol=symbol, tier=tier, final_score=round(score, 1), tech_score=tech_score,
        price=tech.close, reasons=reasons, notes=notes, news=news, flow=flow,
        tech=tech, stop_hint=stop, shares=shares, risk_usd=risk_usd, notional=notional,
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


def format_message(signals: list[Signal], regime: Regime, intraday: bool = False) -> str:
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
        if s.stop_hint:
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
    lines.append("<i>Signal output, not investment advice. Position size accordingly.</i>")
    return "\n".join(lines)


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

def cmd_scan(args) -> int:
    intraday = getattr(args, "intraday", False)
    symbols = args.symbols or (INTRADAY_WATCHLIST if intraday else WATCHLIST)
    mode = f"intraday {INTRADAY_INTERVAL}" if intraday else "end-of-day"
    print(f"Scanning {len(symbols)} symbols ({mode})...")

    tech_by_symbol: dict[str, Technicals] = {}
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

        tech_by_symbol[sym] = t
        warn = "" if t.sma200 is not None else "  (no 200d MA — trend filter off)"
        print(f"  ✓ {sym}: ${t.close:,.2f}  [{len(bars)} bars]{warn}")

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
                regime_input[BENCHMARK] = compute_technicals(bench_bars)
        except DataError as exc:
            print(f"  ! benchmark {BENCHMARK} unavailable: {exc}", file=sys.stderr)

    regime = compute_regime(regime_input)
    if regime.label == "unknown":
        print("  ! regime unknown — running without the risk-off gate", file=sys.stderr)
    print(f"\nMarket regime: {regime.label} "
          f"(SPY {regime.spy_vs_200:+.1f}% vs 200d, breadth {regime.breadth*100:.0f}%)")

    news_by_symbol = fetch_news(list(tech_by_symbol))

    signals: list[Signal] = []
    for sym, tech in tech_by_symbol.items():
        # Only spend SEC requests on names that already look technically interesting.
        prelim, _ = score_technicals(tech)
        flow = fetch_flow(sym) if (prelim >= TIER_WATCH - 10 and not args.no_flow) else FlowSignal()
        sig = evaluate(sym, tech, news_by_symbol.get(sym, NewsSignal()), flow, regime)
        if sig:
            signals.append(sig)

    signals.sort(key=lambda s: -s.final_score)

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

    message = format_message(fresh, regime, intraday=intraday)
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
    print(av_budget_report())
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


def cmd_backtest(args) -> int:
    """Walk the technical rules through history and measure forward returns.

    Only the technical layer is backtested: news sentiment and Form 4 history
    are not available point-in-time from these endpoints, so including them
    would leak lookahead bias into the results.
    """
    symbols = args.symbols or WATCHLIST
    horizons = [5, 10, 20]
    buckets: dict[str, dict[int, list[float]]] = {
        tier: {h: [] for h in horizons} for tier in ("WATCH", "STRONG", "HIGH CONVICTION")
    }
    baseline: dict[int, list[float]] = {h: [] for h in horizons}

    for sym in symbols:
        try:
            # Historical bars never change; cache them for a week.
            bars = fetch_prices(sym, full=True, max_age_hours=168)
        except DataError as exc:
            print(f"  ! {sym}: {exc}", file=sys.stderr)
            continue

        cutoff = datetime.now().date() - timedelta(days=int(365.25 * args.years))
        start_idx = next(
            (i for i, b in enumerate(bars)
             if datetime.strptime(b.date, "%Y-%m-%d").date() >= cutoff),
            0,
        )
        start_idx = max(start_idx, 220)  # need 200d MA warmup
        if start_idx >= len(bars) - max(horizons):
            continue

        hits = 0
        for i in range(start_idx, len(bars) - max(horizons)):
            window = bars[: i + 1]
            t = compute_technicals(window)
            sc, _ = score_technicals(t)

            entry = bars[i].close
            for h in horizons:
                baseline[h].append((bars[i + h].close - entry) / entry * 100)

            tier = ("HIGH CONVICTION" if sc >= TIER_HIGH
                    else "STRONG" if sc >= TIER_STRONG
                    else "WATCH" if sc >= TIER_WATCH else None)
            if not tier:
                continue
            hits += 1
            for h in horizons:
                buckets[tier][h].append((bars[i + h].close - entry) / entry * 100)
        print(f"  {sym}: {hits} signal-days over {args.years}y")

    def stats(xs: list[float]) -> str:
        if not xs:
            return "     n/a"
        mean = sum(xs) / len(xs)
        win = sum(1 for x in xs if x > 0) / len(xs) * 100
        return f"{mean:+6.2f}%  win {win:4.1f}%  n={len(xs)}"

    print(f"\n{'='*66}")
    print(f"BACKTEST — technical layer only, {args.years} years")
    print(f"{'='*66}")
    print(f"\n{'Baseline (every day)':<20}")
    for h in horizons:
        print(f"  {h:>2}d forward: {stats(baseline[h])}")
    for tier in ("WATCH", "STRONG", "HIGH CONVICTION"):
        print(f"\n{tier:<20}")
        for h in horizons:
            print(f"  {h:>2}d forward: {stats(buckets[tier][h])}")
    print(f"\n{'='*66}")
    print("A tier is only useful if its forward return and win rate beat the")
    print("baseline. If they don't, the thresholds need retuning.")
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
    s.set_defaults(func=cmd_scan)

    b = sub.add_parser("backtest", help="validate the scoring rules on history")
    b.add_argument("--years", type=float, default=3.0)
    b.add_argument("--symbols", nargs="*")
    b.set_defaults(func=cmd_backtest)

    t = sub.add_parser("test-telegram", help="verify bot credentials")
    t.set_defaults(func=cmd_test_telegram)

    f = sub.add_parser("refresh-flow", help="rebuild the insider-flow cache")
    f.add_argument("--symbols", nargs="*")
    f.set_defaults(func=cmd_refresh_flow)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

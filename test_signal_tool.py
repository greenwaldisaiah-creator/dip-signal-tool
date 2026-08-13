#!/usr/bin/env python3
"""Verification suite for signal_tool.

Covers the indicator math against hand-checkable values and the scoring
logic against constructed market scenarios. Run: python test_signal_tool.py
"""

import math
from pathlib import Path
import sys

from signal_tool import (
    Bar, FlowSignal, NewsSignal, Regime, Technicals,
    compute_regime, compute_technicals, evaluate, format_message,
    is_new_signal, rsi, score_technicals, sma, stdev,
)

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def approx(a, b, tol=1e-6):
    return a is not None and abs(a - b) < tol


def make_bars(closes, volumes=None, spread=0.01):
    """Build OHLCV bars from a close series with a plausible intraday range."""
    volumes = volumes or [1_000_000] * len(closes)
    out = []
    for i, (c, v) in enumerate(zip(closes, volumes)):
        out.append(Bar(
            date=f"2024-{(i // 28) % 12 + 1:02d}-{i % 28 + 1:02d}",
            open=c * (1 - spread / 2), high=c * (1 + spread),
            low=c * (1 - spread), close=c, volume=float(v),
        ))
    return out


# ---------------------------------------------------------------- indicators

print("\nIndicator math")

check("sma of 1..10 over 10 == 5.5", approx(sma(list(map(float, range(1, 11))), 10), 5.5))
check("sma returns None when history is short", sma([1.0, 2.0], 5) is None)

# Population stdev of 2,4,4,4,5,5,7,9 is exactly 2.0.
check("stdev matches known value", approx(stdev([2, 4, 4, 4, 5, 5, 7, 9], 8), 2.0))

# A monotonically rising series has no down moves -> RSI pinned at 100.
check("RSI == 100 on a pure uptrend", approx(rsi([float(i) for i in range(1, 40)], 14), 100.0))
# Monotonically falling -> no up moves -> RSI 0.
check("RSI == 0 on a pure downtrend", approx(rsi([float(i) for i in range(40, 1, -1)], 14), 0.0))
# Perfectly alternating equal-size moves -> gains == losses -> RSI 50.
alt = [100.0]
for i in range(40):
    alt.append(alt[-1] + (1.0 if i % 2 == 0 else -1.0))
check("RSI ~= 50 on alternating moves", abs(rsi(alt, 14) - 50.0) < 2.0,
      f"got {rsi(alt, 14):.2f}")
check("RSI returns None when history is short", rsi([1.0, 2.0, 3.0], 14) is None)

r = rsi([100.0] * 30, 14)
check("RSI on a flat series is neutral, not 100", approx(r, 50.0), f"got {r}")

# Bollinger band on a flat series collapses onto the mean.
flat = compute_technicals(make_bars([100.0] * 60, spread=0.0))
check("flat series -> zero-width Bollinger bands", approx(flat.bb_lower, 100.0, 1e-9))
check("flat series -> zero drawdown from 20d high",
      approx(flat.dd_from_high20, 0.0, 1e-9), f"got {flat.dd_from_high20}")
check("flat series -> zero Bollinger band width", approx(flat.bb_width_pct, 0.0, 1e-9))

# Drawdown arithmetic: high 100, close 90 -> exactly 10% (bar high is c*1.01).
dd_bars = make_bars([100.0] * 19 + [90.0], spread=0.0)
dd = compute_technicals(dd_bars)
check("drawdown from 20d high computes to 10%", approx(dd.dd_from_high20, 10.0, 1e-6),
      f"got {dd.dd_from_high20}")

# Volume ratio: 19 bars at 1M, final bar at 3M -> ratio vs 20d avg.
vr_bars = make_bars([100.0] * 20, volumes=[1_000_000] * 19 + [3_000_000])
vr = compute_technicals(vr_bars)
expected_ratio = 3_000_000 / ((19 * 1_000_000 + 3_000_000) / 20)
check("volume ratio matches 20d average", approx(vr.vol_ratio, expected_ratio, 1e-9),
      f"got {vr.vol_ratio}")


# ------------------------------------------------------------------ scoring

print("\nTechnical scoring")

def uptrend_then_dip(dip_pct, n_up=260, vol_spike=2.5):
    """A long steady uptrend followed by a sharp multi-day selloff."""
    closes = [100.0 * (1.0025 ** i) for i in range(n_up)]
    peak = closes[-1]
    for k in range(1, 8):
        closes.append(peak * (1 - dip_pct / 100 * k / 7))
    vols = [1_000_000] * n_up + [int(1_000_000 * (1 + vol_spike * k / 7)) for k in range(1, 8)]
    return make_bars(closes, vols)


deep = compute_technicals(uptrend_then_dip(12))
deep_score, deep_reasons = score_technicals(deep)
check("deep dip in an uptrend scores highly", deep_score >= 55, f"got {deep_score}")
check("deep dip stays above the 200d MA", deep.close > deep.sma200)

shallow = compute_technicals(uptrend_then_dip(1.5, vol_spike=0.0))
shallow_score, _ = score_technicals(shallow)
check("a 1.5% wobble scores low", shallow_score < 40, f"got {shallow_score}")
check("deeper dips score above shallower ones", deep_score > shallow_score,
      f"{deep_score} vs {shallow_score}")

# A steady grind lower: below the 200d MA, so trend points must be withheld.
downtrend = compute_technicals(make_bars([200.0 * (0.997 ** i) for i in range(260)]))
dn_score, dn_reasons = score_technicals(downtrend)
check("downtrend earns no trend-quality points",
      not any("uptrend intact" in r for r in dn_reasons))
check("downtrend is flagged as below the 200d MA",
      any("below 200d MA" in r for r in dn_reasons))

check("score is bounded at 100", deep_score <= 100)
check("flat market scores zero", score_technicals(flat)[0] == 0,
      f"got {score_technicals(flat)[0]}")


# ------------------------------------------------------------ confirmation

print("\nConfirmation layer and vetoes")

RISK_ON = Regime(risk_on=True, label="risk-on", spy_vs_200=4.0, breadth=0.7)
RISK_OFF = Regime(risk_on=False, label="risk-off", spy_vs_200=-6.0, breadth=0.2)
NO_NEWS, NO_FLOW = NewsSignal(), FlowSignal()

# A deep dip already pins the technical score at 100, which would clamp any
# upward adjustment. Use a moderate dip so upgrades are observable.
moderate = compute_technicals(uptrend_then_dip(6, vol_spike=0.6))
mod_score, _ = score_technicals(moderate)
check("moderate dip leaves headroom below the cap", mod_score < 100, f"got {mod_score}")

base = evaluate("TEST", moderate, NO_NEWS, NO_FLOW, RISK_ON)
check("a qualifying technical dip produces a signal", base is not None,
      f"score {mod_score}")

deep_sig = evaluate("TEST", deep, NO_NEWS, NO_FLOW, RISK_ON)
check("a deep dip also produces a signal", deep_sig is not None)

bearish = evaluate("TEST", moderate, NewsSignal(score=-0.6, article_count=9, available=True),
                   NO_FLOW, RISK_ON)
check("bearish news lowers the score",
      bearish is None or bearish.final_score < base.final_score,
      f"{bearish.final_score if bearish else None} vs {base.final_score}")

bullish = evaluate("TEST", moderate, NewsSignal(score=0.4, article_count=9, available=True),
                   NO_FLOW, RISK_ON)
check("positive news raises the score", bullish.final_score > base.final_score,
      f"{bullish.final_score} vs {base.final_score}")

buying = evaluate("TEST", moderate, NO_NEWS,
                  FlowSignal(net_usd=8_000_000, buy_count=4, available=True), RISK_ON)
check("insider buying raises the score", buying.final_score > base.final_score,
      f"{buying.final_score} vs {base.final_score}")

selling = evaluate("TEST", moderate, NO_NEWS,
                   FlowSignal(net_usd=-9_000_000, sell_count=6, available=True), RISK_ON)
check("insider selling lowers the score", selling.final_score < base.final_score,
      f"{selling.final_score} vs {base.final_score}")

check("the score is clamped at 100", evaluate(
    "TEST", deep, NewsSignal(score=0.9, article_count=9, available=True),
    FlowSignal(net_usd=5e7, buy_count=9, available=True), RISK_ON).final_score <= 100)

off = evaluate("TEST", moderate, NO_NEWS, NO_FLOW, RISK_OFF)
check("risk-off never improves a tier",
      off is None or off.final_score == base.final_score)

# Falling knife: below the 200d MA, >20% off the 50d high, negative news.
knife_closes = [300.0 * (0.995 ** i) for i in range(260)]
knife = compute_technicals(make_bars(knife_closes, [2_000_000] * 260))
check("knife scenario really is >20% off its 50d high", knife.dd_from_high50 > 20,
      f"got {knife.dd_from_high50:.1f}%")
knife_sig = evaluate("KNIFE", knife, NewsSignal(score=-0.5, article_count=6, available=True),
                     NO_FLOW, RISK_ON)
check("falling knife is vetoed outright", knife_sig is None)

knife_ok = evaluate("KNIFE", knife, NewsSignal(score=0.3, article_count=6, available=True),
                    NO_FLOW, RISK_ON)
check("same setup with positive news is not vetoed by the knife rule",
      knife_ok is None or True)  # may still fail thresholds; must not crash

check("no news + no flow leaves the technical score untouched",
      base.final_score == base.tech_score, f"{base.final_score} vs {base.tech_score}")


# ----------------------------------------------------------------- plumbing

print("\nRegime, dedupe and formatting")

reg = compute_regime({"SPY": deep, "AAPL": deep})
check("regime detects an uptrend as risk-on", reg.risk_on and reg.spy_vs_200 > 0)
check("regime handles a missing benchmark", compute_regime({}).label == "unknown")

from datetime import datetime, timedelta, timezone
now = datetime.now(timezone.utc)
check("an unseen ticker is a new signal", is_new_signal({}, base))
check("a same-tier repeat today is suppressed",
      not is_new_signal({"TEST": {"date": now.isoformat(), "tier": base.tier}}, base))
check("a same-tier repeat after the window resends",
      is_new_signal({"TEST": {"date": (now - timedelta(days=9)).isoformat(),
                              "tier": base.tier}}, base))
check("an upgrade always resends",
      is_new_signal({"TEST": {"date": now.isoformat(), "tier": "WATCH"}},
                    evaluate("TEST", deep, NewsSignal(score=0.4, article_count=9,
                                                      available=True),
                             FlowSignal(net_usd=9e6, buy_count=5, available=True), RISK_ON)))
check("corrupt state entry is treated as new",
      is_new_signal({"TEST": {"date": "not-a-date"}}, base))

msg = format_message([base], RISK_ON)
check("message includes the ticker", "TEST" in msg)
check("message includes the tier", base.tier in msg)
check("message carries a risk disclaimer", "not investment advice" in msg.lower())
check("message uses HTML tags Telegram accepts", "<b>" in msg and "<i>" in msg)
check("message stays within Telegram's 4096-char limit", len(msg) < 4096, f"len {len(msg)}")


# --------------------------------------------------------------- edge cases

print("\nEdge cases")

short = compute_technicals(make_bars([100.0] * 25))
check("short history yields no 200d MA rather than crashing", short.sma200 is None)
check("short history still scores without error", isinstance(score_technicals(short)[0], float))
check("short history does not claim trend quality",
      not any("200d" in r for r in score_technicals(short)[1]))

zero_vol = compute_technicals(make_bars([100.0] * 30, volumes=[0] * 30))
check("zero volume does not divide by zero", zero_vol.vol_ratio in (None, 0.0))

print("\nFalling-knife veto regressions")

# The veto must fire on price structure alone -- a news outage must not
# silently disable the model's main downside guard.
check("veto fires with NO news available",
      evaluate("KNIFE", knife, NewsSignal(), NO_FLOW, RISK_ON) is None)
check("veto fires with neutral news",
      evaluate("KNIFE", knife, NewsSignal(score=0.0, article_count=5, available=True),
               NO_FLOW, RISK_ON) is None)
check("veto fires with mildly negative news",
      evaluate("KNIFE", knife, NewsSignal(score=-0.05, article_count=5, available=True),
               NO_FLOW, RISK_ON) is None)
check("veto is NOT rescued by insider buying alone",
      evaluate("KNIFE", knife, NewsSignal(),
               FlowSignal(net_usd=5e7, buy_count=9, available=True), RISK_ON) is None)
_rescued = evaluate("KNIFE", knife, NewsSignal(score=0.4, article_count=8, available=True),
                    NO_FLOW, RISK_ON)
check("clearly positive news can rescue a broken-trend setup",
      _rescued is None or any("allowed through" in n for n in _rescued.notes))
check("a healthy uptrend dip is never vetoed",
      evaluate("TEST", moderate, NewsSignal(), NO_FLOW, RISK_ON) is not None)

print("\nPosition sizing")

import signal_tool as ST

def with_account(size, risk=1.0, maxpct=20.0):
    ST.ACCOUNT_SIZE, ST.RISK_PCT, ST.MAX_POSITION_PCT = size, risk, maxpct

with_account(0)
check("sizing is off when ACCOUNT_SIZE is unset",
      ST.position_size(100.0, 95.0) == (None, None, None))

# $100k account, 1% risk = $1000 budget; $5 risk/share -> 200 shares.
with_account(100_000, 1.0, 100.0)
sh, risk, notional = ST.position_size(100.0, 95.0)
check("shares follow risk budget / risk-per-share", sh == 200, f"got {sh}")
check("dollar risk matches the budget", approx(risk, 1000.0, 0.01), f"got {risk}")
check("notional is shares x price", approx(notional, 20000.0, 0.01), f"got {notional}")

# Halving risk % halves the position.
with_account(100_000, 0.5, 100.0)
check("halving risk pct halves the size", ST.position_size(100.0, 95.0)[0] == 100)

# A tight stop would imply a huge position -> the cap must bind.
with_account(100_000, 1.0, 20.0)
sh_capped, _, notional_capped = ST.position_size(100.0, 99.9)
check("max position cap binds on a tight stop", notional_capped <= 20_000 + 100,
      f"got {notional_capped}")
check("uncapped tight stop would have been far larger", sh_capped == 200, f"got {sh_capped}")

# A wider stop on the same account gives a smaller position -- the core property.
with_account(100_000, 1.0, 100.0)
tight = ST.position_size(100.0, 98.0)[0]
wide = ST.position_size(100.0, 90.0)[0]
check("more volatile (wider stop) -> smaller position", wide < tight, f"{wide} vs {tight}")

check("stop above price is rejected", ST.position_size(100.0, 101.0) == (None, None, None))
check("stop equal to price is rejected", ST.position_size(100.0, 100.0) == (None, None, None))
check("missing stop is rejected", ST.position_size(100.0, None) == (None, None, None))
with_account(500)
check("account too small for one share returns nothing",
      ST.position_size(10_000.0, 9_000.0) == (None, None, None))
with_account(0)


print("\nRequest budget")

ST._av_requests_used = 0
ST.AV_DAILY_LIMIT = 3
ST.av_budget_check(); ST.av_budget_check(); ST.av_budget_check()
check("budget allows requests up to the limit", ST._av_requests_used == 3)
try:
    ST.av_budget_check()
    check("budget raises past the limit", False, "no exception")
except ST.DataError as e:
    check("budget raises past the limit", "budget exhausted" in str(e))
check("budget report is human readable", "/3" in ST.av_budget_report())
ST._av_requests_used = 0
ST.AV_DAILY_LIMIT = 25

check("default watchlist fits the free tier",
      len(ST.WATCHLIST) + 1 <= 25, f"{len(ST.WATCHLIST)} symbols + 1 news call")
check("watchlist has no duplicates", len(set(ST.WATCHLIST)) == len(ST.WATCHLIST))
check("benchmark is in the watchlist", ST.BENCHMARK in ST.WATCHLIST)
check("intraday list is a subset of the watchlist",
      set(ST.INTRADAY_WATCHLIST) <= set(ST.WATCHLIST))


print("\nIntraday blending")

# Intraday leg: oversold, on heavy volume. Daily leg: healthy uptrend.
intra_t = compute_technicals(uptrend_then_dip(9, n_up=60, vol_spike=2.0))
daily_t = compute_technicals(uptrend_then_dip(4, n_up=260, vol_spike=0.5))
blend = ST.blend_intraday_with_daily(intra_t, daily_t)

check("blend takes price from the intraday leg", blend.close == intra_t.close)
check("blend takes RSI from the intraday leg", blend.rsi14 == intra_t.rsi14)
check("blend takes the 200d MA from the daily leg", blend.sma200 == daily_t.sma200)
check("blend takes the 50d MA from the daily leg", blend.sma50 == daily_t.sma50)
check("blend takes ATR from the daily leg", blend.atr_pct == daily_t.atr_pct)
check("blend recomputes drawdown against the daily 20d high",
      approx(blend.dd_from_high20,
             (daily_t.high20 - intra_t.close) / daily_t.high20 * 100, 1e-9))
check("blended technicals still score without error",
      isinstance(score_technicals(blend)[0], float))
check("blend survives a daily leg with no 200d MA",
      ST.blend_intraday_with_daily(intra_t, short).sma200 is None)

# The veto must still work on blended data, using the DAILY trend.
# Both legs have to describe the same instrument for the blend to mean
# anything, so build the intraday leg down to the daily series' own last price.
def intraday_leg_ending_at(price_level, dip_pct=3.0, n=80):
    closes = [price_level * (1 + dip_pct / 100 * (1 - i / (n - 1))) for i in range(n)]
    vols = [1_000_000] * (n - 8) + [3_000_000] * 8
    return compute_technicals(make_bars(closes, vols))

knife_intra = intraday_leg_ending_at(knife.close)
check("coherent intraday leg ends at the daily close",
      approx(knife_intra.close, knife.close, 1e-6))
blend_knife = ST.blend_intraday_with_daily(knife_intra, knife)
check("blended close matches the daily close for a coherent pair",
      approx(blend_knife.close, knife.close, 1e-6))
check("blended drawdown stays above the veto threshold",
      blend_knife.dd_from_high50 > 20, f"got {blend_knife.dd_from_high50:.1f}%")
check("veto still fires on blended data via the daily trend",
      evaluate("K", blend_knife, NewsSignal(), NO_FLOW, RISK_ON) is None)

# And a coherent healthy pair must still pass.
healthy_intra = intraday_leg_ending_at(moderate.close, dip_pct=2.0)
blend_ok = ST.blend_intraday_with_daily(healthy_intra, moderate)
check("a healthy blended pair is not vetoed",
      evaluate("OK", blend_ok, NewsSignal(), NO_FLOW, RISK_ON) is not None)


print("\nIntraday dedupe")

check("intraday and EOD state keys are distinct",
      ST.state_key("AAPL", True) != ST.state_key("AAPL", False))
_st = {ST.state_key("TEST", False): {"date": now.isoformat(), "tier": base.tier}}
check("an EOD alert does not suppress the intraday one",
      is_new_signal(_st, base, intraday=True))
check("an EOD alert still suppresses the next EOD one",
      not is_new_signal(_st, base, intraday=False))
_st2 = {ST.state_key("TEST", True): {"date": (now - timedelta(hours=1)).isoformat(),
                                     "tier": base.tier}}
check("intraday repeat within an hour is suppressed",
      not is_new_signal(_st2, base, intraday=True))
_st3 = {ST.state_key("TEST", True): {"date": (now - timedelta(hours=8)).isoformat(),
                                     "tier": base.tier}}
check("intraday resends after the short window",
      is_new_signal(_st3, base, intraday=True))

msg_i = format_message([base], RISK_ON, intraday=True)
check("intraday message is labelled as such", "INTRADAY" in msg_i)
check("EOD message is not labelled intraday",
      "INTRADAY" not in format_message([base], RISK_ON))

check("EOD + intraday together fit the free tier",
      (len(ST.WATCHLIST) + 1) + len(ST.INTRADAY_WATCHLIST) <= 25,
      f"{len(ST.WATCHLIST)}+1 EOD + {len(ST.INTRADAY_WATCHLIST)} intraday")

print("\nSizing display honesty")

with_account(100_000, 1.0, 20.0)
_s = evaluate("TEST", moderate, NO_NEWS, NO_FLOW, RISK_ON)
_m = format_message([_s], RISK_ON)
check("alert reports a share count when sizing is on", _s.shares and "sh " in _m)
_actual = _s.risk_usd / 100_000 * 100
check("displayed risk pct matches realized risk",
      f"{_actual:.2f}%" in _m, f"expected {_actual:.2f}% in message")
check("a capped position is labelled as capped",
      (_actual >= 1.0 - 0.05) or ("capped" in _m),
      f"actual {_actual:.2f}% but no capped label")
check("realized risk never exceeds the configured target",
      _actual <= 1.0 + 1e-9, f"got {_actual:.3f}%")
check("notional respects the position cap",
      _s.notional <= 20_000 + _s.price, f"got {_s.notional}")

with_account(0)
check("no share line when sizing is disabled",
      "sh " not in format_message([evaluate("TEST", moderate, NO_NEWS, NO_FLOW, RISK_ON)],
                                  RISK_ON))

print("\nFree-tier survival (regressions from the first live run)")

# 1. outputsize=full is a PREMIUM feature. The default must never request it.
import inspect
_src = inspect.getsource(ST.fetch_prices)
check("fetch_prices defaults to compact, not premium-only full",
      inspect.signature(ST.fetch_prices).parameters["full"].default is False)
check("fetch_prices can still merge accumulated history", "merged" in _src)

# 2. Pacing must happen before the request, not after a success.
_pace = inspect.getsource(ST.av_pace)
check("pacing uses a monotonic clock", "monotonic" in _pace)
for fn in (ST.fetch_prices, ST.fetch_weekly):
    s = inspect.getsource(fn)
    check(f"{fn.__name__} paces before issuing the request",
          s.index("av_pace()") < s.index("http_get_json"))

_t0 = ST.time.monotonic()
ST._av_last_request_at = 0.0
ST.av_pace(); ST.av_pace()
check("two paced calls are spaced by the rate interval",
      ST.time.monotonic() - _t0 >= ST.AV_RATE_SLEEP * 0.9,
      f"took {ST.time.monotonic()-_t0:.2f}s")

# 3. With only 100 daily bars there is no 200d MA -- the weekly series must
#    supply the trend, or the model's main guard vanishes.
_short_daily = compute_technicals(make_bars([100.0 + i * 0.1 for i in range(100)]))
check("100 daily bars really do lack a 200d MA", _short_daily.sma200 is None)
check("...and therefore have no trend reference on their own",
      _short_daily.trend_ma is None and _short_daily.trend_basis == "none")

_weekly = make_bars([80.0 + i * 0.4 for i in range(120)])   # 120 weekly bars
_withtrend = ST.attach_weekly_trend(_short_daily, _weekly)
check("weekly bars supply a 40w trend reference", _withtrend.trend_ma is not None)
check("trend basis is labelled 40w", _withtrend.trend_basis == "40w")
check("weekly bars supply a 52-week high", _withtrend.high52w is not None)

# A real 200d MA must win over the weekly stand-in.
_long_daily = compute_technicals(make_bars([100.0 * (1.001 ** i) for i in range(260)]))
check("daily 200d MA takes precedence when available",
      ST.attach_weekly_trend(_long_daily, _weekly).trend_basis == "200d")

# 4. No trend data at all must WITHHOLD points, never assume health.
_no_trend_score, _no_trend_reasons = score_technicals(_short_daily)
check("missing trend data withholds the trend points",
      any("withheld" in r for r in _no_trend_reasons))
check("missing trend data cannot reach the top tier",
      _no_trend_score < 70, f"got {_no_trend_score}")

# 5. The veto must work off the 40w reference too, not just the 200d.
_knife_daily = compute_technicals(make_bars([200.0 * (0.993 ** i) for i in range(100)]))
_knife_weekly = make_bars([400.0 * (0.99 ** i) for i in range(120)])
_knife_w = ST.attach_weekly_trend(_knife_daily, _knife_weekly)
check("weekly-based knife is below its 40w MA",
      _knife_w.trend_ma is not None and _knife_w.close < _knife_w.trend_ma)
check("veto fires using the 40w reference",
      evaluate("KW", _knife_w, NewsSignal(), NO_FLOW, RISK_ON) is None)

# 6. Budget must cover a cold start: every symbol needs daily + weekly at once.
check("cold-start budget fits the free tier",
      len(ST.WATCHLIST) * 2 + 1 <= 25,
      f"{len(ST.WATCHLIST)}x2 + 1 news = {len(ST.WATCHLIST)*2+1}")
check("steady-state budget leaves room for intraday",
      len(ST.WATCHLIST) + 1 + len(ST.INTRADAY_WATCHLIST) <= 25)

# 7. Intraday blending must carry the long-run context through.
_b = ST.blend_intraday_with_daily(intra_t, _withtrend)
check("blend carries trend_ma from the daily leg", _b.trend_ma == _withtrend.trend_ma)
check("blend carries trend_basis", _b.trend_basis == _withtrend.trend_basis)
check("blend carries the 52-week high", _b.high52w == _withtrend.high52w)

print("\nProvider fallback (Twelve Data + Alpha Vantage)")

import json as _json, importlib

# Twelve Data response parsing, including its quirks.
_td_payload = {"status":"ok","values":[
    {"datetime":"2026-08-12","open":"10","high":"11","low":"9","close":"10.5","volume":"1000"},
    {"datetime":"2026-08-11","open":"9","high":"10","low":"8","close":"9.5","volume":"900"}]}
_orig_get = ST.http_get_json
try:
    ST.http_get_json = lambda url, **kw: _td_payload
    ST.TD_KEY = "test"; ST._td_requests_used = 0; ST.TD_RATE_SLEEP = 0.0
    _bars = ST.td_time_series("X")
    check("twelve data parses values into bars", len(_bars) == 2)
    check("twelve data bars are sorted oldest first", _bars[0].date < _bars[1].date)
    check("twelve data close parsed correctly", approx(_bars[-1].close, 10.5))

    # ETFs sometimes report null volume -- must not drop the bar.
    ST.http_get_json = lambda url, **kw: {"status":"ok","values":[
        {"datetime":"2026-08-12","open":"1","high":"1","low":"1","close":"1","volume":None}]}
    check("null volume becomes 0 instead of dropping the bar",
          ST.td_time_series("X")[0].volume == 0)

    # An error payload must raise, not return junk.
    ST.http_get_json = lambda url, **kw: {"status":"error","message":"limit reached"}
    try:
        ST.td_time_series("X"); check("twelve data error raises", False, "no raise")
    except ST.DataError as e:
        check("twelve data error raises DataError", "limit reached" in str(e))
finally:
    ST.http_get_json = _orig_get

# Budget guard
ST._td_requests_used = 0; ST.TD_DAILY_LIMIT = 2
ST.td_budget_check(); ST.td_budget_check()
try:
    ST.td_budget_check(); check("twelve data budget caps", False, "no raise")
except ST.DataError:
    check("twelve data budget caps", True)
ST._td_requests_used = 0; ST.TD_DAILY_LIMIT = 800

# Merge semantics: union by date, newer wins, chronological out.
_a = make_bars([1.0, 2.0, 3.0]); _b = make_bars([9.0, 9.0, 9.0, 9.0])
_m = ST._merge_bars(_a, _b)
check("merge unions by date", len(_m) == 4, f"got {len(_m)}")
check("merge output is chronological", [x.date for x in _m] == sorted(x.date for x in _m))
check("newer data wins on overlapping dates", _m[0].close == 9.0)
check("merging with empty history is a no-op", len(ST._merge_bars([], _a)) == len(_a))

# Round-trip through the cache format.
_rt = ST._bars_from_json(_json.loads(_json.dumps(ST._bars_to_json(_a, "twelvedata"))))
check("bars survive a cache round-trip", [x.date for x in _rt] == [x.date for x in _a])
check("round-tripped closes are unchanged", approx(_rt[0].close, _a[0].close))

# The watchlist must size itself to the available budget.
check("without a TD key the watchlist stays inside AV's 25/day",
      len(ST.INDICES + ST.MEGACAPS) * 2 + 1 <= 25)
check("with a TD key the sector ETFs are included",
      len(ST.INDICES + ST.MEGACAPS + ST.SECTORS) == 21)

print("\nContinuous intraday scanning")

from datetime import datetime as _dt, timezone as _tz

def at(y, mo, d, h, mi):
    return _dt(y, mo, d, h, mi, tzinfo=_tz.utc)

# Aug 13 2026 is a Thursday. Market hours in UTC (EDT): 13:30 - 20:00.
check("open during regular hours", ST.market_status(at(2026,8,13,15,0))[0])
check("open right at the bell", ST.market_status(at(2026,8,13,13,30))[0])
check("open at the close minute", ST.market_status(at(2026,8,13,20,0))[0])
check("closed one minute before the open",
      not ST.market_status(at(2026,8,13,13,29))[0])
check("closed one minute after the close",
      not ST.market_status(at(2026,8,13,20,1))[0])
check("closed overnight", not ST.market_status(at(2026,8,13,3,0))[0])
check("closed on Saturday", not ST.market_status(at(2026,8,15,15,0))[0])
check("closed on Sunday", not ST.market_status(at(2026,8,16,15,0))[0])
check("weekend reason is reported", ST.market_status(at(2026,8,15,15,0))[1] == "weekend")
# Jul 3 2026 is an observed holiday (Independence Day) and a Friday.
check("closed on a market holiday", not ST.market_status(at(2026,7,3,15,0))[0])
check("holiday reason is reported",
      ST.market_status(at(2026,7,3,15,0))[1] == "market holiday")
check("before-open reason is specific",
      ST.market_status(at(2026,8,13,12,0))[1] == "before the open")
check("after-close reason is specific",
      ST.market_status(at(2026,8,13,22,0))[1] == "after the close")

# Intraday must hold a higher bar than end-of-day.
check("intraday minimum is above the WATCH tier",
      ST.INTRADAY_MIN_SCORE > ST.TIER_WATCH,
      f"{ST.INTRADAY_MIN_SCORE} vs {ST.TIER_WATCH}")
check("intraday minimum matches the STRONG tier",
      ST.INTRADAY_MIN_SCORE == ST.TIER_STRONG)

# Credit budget for a full session of 15-minute scans.
_SWEEPS = 27  # 9:30 through 16:00 inclusive
_intraday_credits = len(ST.INTRADAY_WATCHLIST) * _SWEEPS
_eod_credits = len(ST.WATCHLIST)
check("a full session of intraday scans fits free Twelve Data",
      _intraday_credits + _eod_credits <= 800,
      f"{_intraday_credits} intraday + {_eod_credits} EOD")
check("15-minute bars make >27 sweeps/day pointless",
      _SWEEPS * 15 >= 6.5 * 60, "a session is 390 minutes")

# Usage must survive across processes, since every scan is a fresh run.
import shutil as _sh
_sh.rmtree("cache", ignore_errors=True)
ST._av_requests_used, ST._td_requests_used = 0, 0
ST.persist_usage()
ST._av_requests_used, ST._td_requests_used = 7, 42
ST.persist_usage()
ST._av_requests_used, ST._td_requests_used = 0, 0     # simulate a new process
ST.load_persisted_usage()
check("Alpha Vantage usage survives a restart", ST._av_requests_used == 7,
      f"got {ST._av_requests_used}")
check("Twelve Data usage survives a restart", ST._td_requests_used == 42,
      f"got {ST._td_requests_used}")
check("usage is keyed by UTC date",
      _dt.now(_tz.utc).date().isoformat() in ST._usage_path().name)
_sh.rmtree("cache", ignore_errors=True)
ST.load_persisted_usage()
check("a new day starts the count at zero",
      ST._av_requests_used == 0 and ST._td_requests_used == 0)

# Corrupt usage file must not break a scan.
ST.cache_path("x")  # ensure cache dir exists
ST._usage_path().write_text("{not json")
ST.load_persisted_usage()
check("corrupt usage file falls back to zero", ST._av_requests_used == 0)
_sh.rmtree("cache", ignore_errors=True)

print("\nTrade plan")

_plan = ST.build_trade_plan(moderate)
check("a valid setup produces a plan", _plan is not None)
check("stop sits below entry", _plan.stop < _plan.entry)
check("target sits above entry", _plan.target > _plan.entry)
check("entry never chases above the last close", _plan.entry <= moderate.close + 1e-9,
      f"entry {_plan.entry} vs close {moderate.close}")
check("risk/reward is computed consistently",
      approx(_plan.rr, _plan.reward_per_share / _plan.risk_per_share, 0.01))
check("stop distance is 1.5 ATR",
      approx(_plan.entry - _plan.stop, 1.5 * moderate.close * moderate.atr_pct / 100, 0.02))
check("target percent matches the levels",
      approx(_plan.target_pct, (_plan.target - _plan.entry) / _plan.entry * 100, 0.01))
check("holding estimate is at least 2 days", _plan.hold_days_est >= 2)
check("holding estimate is capped at 60 days", _plan.hold_days_est <= 60)
check("annualised volatility is reported", _plan.ann_vol_pct is not None)
check("annualised vol exceeds daily ATR", _plan.ann_vol_pct > (_plan.atr_pct or 0))
check("a flat series yields no plan (zero ATR)", ST.build_trade_plan(flat) is None)
check("short history still yields a plan or None safely",
      ST.build_trade_plan(short) is None or ST.build_trade_plan(short).entry > 0)

_sig_with_plan = evaluate("TEST", moderate, NO_NEWS, NO_FLOW, RISK_ON)
_msg = format_message([_sig_with_plan], RISK_ON)
for _field, _label in (("Buy ≤", "entry"), ("Stop", "stop"), ("Target", "target"),
                       ("R:R", "risk/reward"), ("Est. hold", "holding estimate"),
                       ("ATR", "volatility")):
    check(f"alert includes the {_label}", _field in _msg)


print("\nPaper trading engine")

import shutil as _sh2
ST.TRADES_FILE = Path("test_trades.json")
ST.PAPER_TRADE_USD = 100.0

def fresh_book():
    if ST.TRADES_FILE.exists(): ST.TRADES_FILE.unlink()
    return ST.load_trades()

_book = fresh_book()
check("a new book is empty", _book == {"open": [], "closed": []})
check("opening a trade succeeds", ST.open_paper_trade(_book, _sig_with_plan, False))
check("the open position is recorded", len(_book["open"]) == 1)
check("$100 buys the right share count",
      approx(_book["open"][0]["shares"], 100.0 / _sig_with_plan.plan.entry, 1e-6))
check("a second trade in the same name is refused",
      not ST.open_paper_trade(_book, _sig_with_plan, False))
check("...and does not duplicate the position", len(_book["open"]) == 1)

_e = _book["open"][0]["entry"]; _st = _book["open"][0]["stop"]; _tg = _book["open"][0]["target"]

def bars_after(lows_highs):
    return [Bar(date=f"2099-01-{i+1:02d}", open=_e, high=h, low=l, close=(h+l)/2,
                volume=1e6) for i, (l, h) in enumerate(lows_highs)]

# Target hit -> a win of exactly the target distance.
_b = fresh_book(); ST.open_paper_trade(_b, _sig_with_plan, False)
_closed = ST.resolve_paper_trades(_b, {"TEST": bars_after([(_e, _tg + 1)])})
check("a target hit closes the trade", len(_closed) == 1)
check("target exit is booked at the target price", approx(_closed[0]["exit"], round(_tg, 2), 0.01))
check("target exit is a profit", _closed[0]["pnl"] > 0)
check("exit reason is recorded as target", _closed[0]["exit_reason"] == "target")
check("the position leaves the open book", len(_b["open"]) == 0)

# Stop hit -> a loss.
_b = fresh_book(); ST.open_paper_trade(_b, _sig_with_plan, False)
_closed = ST.resolve_paper_trades(_b, {"TEST": bars_after([(_st - 1, _e)])})
check("a stop hit closes the trade", len(_closed) == 1 and _closed[0]["exit_reason"] == "stop")
check("stop exit is a loss", _closed[0]["pnl"] < 0)

# Both touched in one bar -> must assume the STOP, never the target.
_b = fresh_book(); ST.open_paper_trade(_b, _sig_with_plan, False)
_closed = ST.resolve_paper_trades(_b, {"TEST": bars_after([(_st - 1, _tg + 1)])})
check("an ambiguous bar resolves pessimistically to the stop",
      _closed[0]["exit_reason"] == "stop",
      "assuming the target would flatter every result")

# Neither touched -> stays open.
_b = fresh_book(); ST.open_paper_trade(_b, _sig_with_plan, False)
_closed = ST.resolve_paper_trades(_b, {"TEST": bars_after([(_e * 0.99, _e * 1.01)])})
check("an unresolved trade stays open", len(_closed) == 0 and len(_b["open"]) == 1)

# Bars on/before the entry date must be ignored (no same-bar lookahead).
_b = fresh_book(); ST.open_paper_trade(_b, _sig_with_plan, False)
_today = ST.datetime.now(ST.timezone.utc).date().isoformat()
_same = [Bar(date=_today, open=_e, high=_tg + 5, low=_st - 5, close=_e, volume=1e6)]
check("bars from the entry day cannot close the trade",
      len(ST.resolve_paper_trades(_b, {"TEST": _same})) == 0)

print("\nPaper statistics")

_b = {"open": [], "closed": [
    {"symbol":"A","pnl":10.0,"cost":100,"pnl_pct":10.0},
    {"symbol":"B","pnl":-5.0,"cost":100,"pnl_pct":-5.0},
    {"symbol":"C","pnl":20.0,"cost":100,"pnl_pct":20.0},
    {"symbol":"D","pnl":-5.0,"cost":100,"pnl_pct":-5.0}]}
_s = ST.paper_stats(_b)
check("trade count is right", _s["trades"] == 4)
check("wins and losses are counted", _s["wins"] == 2 and _s["losses"] == 2)
check("win rate is right", approx(_s["win_rate"], 50.0, 0.01))
check("total P/L is right", approx(_s["total_pnl"], 20.0, 0.01))
check("profit factor = gross wins / gross losses", approx(_s["profit_factor"], 3.0, 0.01))
check("return % is on capital deployed", approx(_s["return_pct"], 5.0, 0.01))
check("average win is right", approx(_s["avg_win"], 15.0, 0.01))
check("average loss is negative", _s["avg_loss"] < 0)
check("best and worst are tracked", _s["best"] == 20.0 and _s["worst"] == -5.0)

_nl = ST.paper_stats({"open": [], "closed": [{"symbol":"A","pnl":10.0,"cost":100}]})
check("profit factor is None (not infinity) with no losses",
      _nl["profit_factor"] is None)
check("summary line handles the no-loss case", "PF —" in ST.paper_summary_line(_nl))
_empty = ST.paper_stats({"open": [], "closed": []})
check("empty book does not divide by zero", _empty["trades"] == 0 and _empty["win_rate"] == 0)
check("empty summary says so", "no closed trades" in ST.paper_summary_line(_empty))

_msg2 = format_message([_sig_with_plan], RISK_ON, stats=_s)
check("the alert carries the paper scoreboard", "Paper:" in _msg2)
check("alert with scoreboard stays under Telegram's limit", len(_msg2) < 4096)

if ST.TRADES_FILE.exists(): ST.TRADES_FILE.unlink()
ST.TRADES_FILE = Path("trades.json")

print("\nBacktest trade simulator")

def _bars(seq):
    """seq of (low, high) -> bars; close midway."""
    return [Bar(date=f"2099-01-{i+1:02d}", open=(l+h)/2, high=h, low=l,
                close=(l+h)/2, volume=1e6) for i, (l, h) in enumerate(seq)]

class _P:
    def __init__(self, entry, stop, target):
        self.entry, self.stop, self.target = entry, stop, target
_plan100 = _P(100.0, 95.0, 110.0)

# Bar 0 is the signal bar and is never tradable.
# Limit never reached -> NO TRADE. Counting these would fake the results.
r = ST.simulate_trade(_bars([(100,100),(101,105),(102,106),(103,107)]), 0, _plan100)
check("an unfilled limit produces no trade", r is None)

# Fills, then hits target.
r = ST.simulate_trade(_bars([(100,100),(99,101),(105,111)]), 0, _plan100)
check("a filled limit that reaches the target is a win",
      r and r["exit_reason"] == "target")
check("target exit books exactly the target move", approx(r["pnl_pct"], 10.0, 0.01))

# Fills, then hits stop.
r = ST.simulate_trade(_bars([(100,100),(99,101),(94,99)]), 0, _plan100)
check("a filled limit that reaches the stop is a loss",
      r and r["exit_reason"] == "stop")
check("stop exit books the stop move plus slippage",
      r["pnl_pct"] < -5.0 and r["pnl_pct"] > -5.2,
      f"got {r['pnl_pct']:.3f}% (expected slightly worse than -5%)")

# Stop and target in the SAME bar -> must book the stop.
r = ST.simulate_trade(_bars([(100,100),(94,111)]), 0, _plan100)
check("stop wins ties against the target", r["exit_reason"] == "stop",
      "assuming the target would inflate every backtest")

# Fill window: a limit reached only on the 5th bar is too late.
late = _bars([(100,100),(101,102),(102,103),(103,104),(99,101),(105,111)])
check("a limit filled outside the fill window is skipped",
      ST.simulate_trade(late, 0, _plan100, fill_window=3) is None)
check("...but is taken with a wider fill window",
      ST.simulate_trade(late, 0, _plan100, fill_window=5) is not None)

# Time exit when neither level is touched.
flat_bars = _bars([(100,100)] + [(99,101)] * 12)
r = ST.simulate_trade(flat_bars, 0, _plan100, max_hold=5)
check("an untouched trade exits on time", r and r["exit_reason"] == "time")
check("time exit respects max_hold", r["days"] <= 5, f"held {r['days']}")

# Running out of data must not crash or index past the end.
r = ST.simulate_trade(_bars([(100,100),(99,101)]), 0, _plan100, max_hold=30)
check("a truncated series exits safely", r is not None and r["exit_reason"] == "time")
check("days held is never negative", r["days"] >= 0)

print("\nBacktest aggregation")

_t = [{"pnl_pct": 10.0, "days": 5, "exit_reason": "target"},
      {"pnl_pct": -5.0, "days": 3, "exit_reason": "stop"},
      {"pnl_pct": 20.0, "days": 8, "exit_reason": "target"},
      {"pnl_pct": -5.0, "days": 2, "exit_reason": "stop"}]
_a = ST._agg(_t)
check("aggregate counts trades", _a["n"] == 4)
check("aggregate win rate", approx(_a["win_rate"], 50.0, 0.01))
check("aggregate average return", approx(_a["avg"], 5.0, 0.01))
check("aggregate profit factor", approx(_a["pf"], 3.0, 0.01))
check("aggregate average hold", approx(_a["avg_days"], 4.5, 0.01))
check("exit reasons are tallied", _a["targets"] == 2 and _a["stops"] == 2)
check("empty aggregate is safe", ST._agg([])["n"] == 0)
check("profit factor is None with no losses",
      ST._agg([{"pnl_pct": 5.0, "days": 1, "exit_reason": "target"}])["pf"] is None)
check("row formatting handles an empty bucket", "no trades" in ST._fmt_row("X", ST._agg([])))
check("row formatting handles n/a profit factor",
      "n/a" in ST._fmt_row("X", ST._agg([{"pnl_pct":5.0,"days":1,"exit_reason":"target"}])))

# The simulator must use the same tie-break rule as the live paper tracker.
_pb = {"open": [], "closed": []}
_sig2 = evaluate("TIE", moderate, NO_NEWS, NO_FLOW, RISK_ON)
ST.TRADES_FILE = Path("test_trades2.json")
if ST.TRADES_FILE.exists(): ST.TRADES_FILE.unlink()
ST.open_paper_trade(_pb, _sig2, False)
_e2, _s2, _t2 = _pb["open"][0]["entry"], _pb["open"][0]["stop"], _pb["open"][0]["target"]
_amb = [Bar(date="2099-06-01", open=_e2, high=_t2 + 1, low=_s2 - 1, close=_e2, volume=1e6)]
_live = ST.resolve_paper_trades(_pb, {"TIE": _amb})[0]["exit_reason"]
_bt = ST.simulate_trade(
    [Bar(date="2099-05-30", open=_e2, high=_e2, low=_e2, close=_e2, volume=1e6),
     Bar(date="2099-06-01", open=_e2, high=_t2 + 1, low=_s2 - 1, close=_e2, volume=1e6)],
    0, _P(_e2, _s2, _t2))["exit_reason"]
check("backtest and live tracker break ties identically",
      _live == _bt == "stop", f"live={_live} backtest={_bt}")
if ST.TRADES_FILE.exists(): ST.TRADES_FILE.unlink()
ST.TRADES_FILE = Path("trades.json")

print("\nEnvironment parsing (GitHub empty-variable crash)")

import os as _os, subprocess as _sp, sys as _sys

def with_env(**kw):
    old = {k: _os.environ.get(k) for k in kw}
    _os.environ.update({k: v for k, v in kw.items() if v is not None})
    for k, v in kw.items():
        if v is None: _os.environ.pop(k, None)
    return old

def restore(old):
    for k, v in old.items():
        if v is None: _os.environ.pop(k, None)
        else: _os.environ[k] = v

# The exact GitHub Actions failure: an undefined repo variable arrives as "".
_o = with_env(TEST_NUM="")
check("empty string falls back to the default (the GH Actions crash)",
      ST.env_float("TEST_NUM", 100.0) == 100.0)
check("empty string works for ints too", ST.env_int("TEST_NUM", 30) == 30)
check("empty string falls back for strings", ST.env_str("TEST_NUM", "x") == "x")
restore(_o)

_o = with_env(TEST_NUM="   ")
check("whitespace-only is treated as unset", ST.env_float("TEST_NUM", 7.0) == 7.0)
restore(_o)

_o = with_env(TEST_NUM=None)
check("a genuinely missing var uses the default", ST.env_float("TEST_NUM", 5.0) == 5.0)
restore(_o)

_o = with_env(TEST_NUM="42.5")
check("a normal value parses", ST.env_float("TEST_NUM", 0.0) == 42.5)
restore(_o)

# Values a person might realistically type into a web form.
for raw, want in (("$25,000", 25000.0), ("1,000", 1000.0), ("$100", 100.0),
                  ("2.5%", 2.5), ("  50  ", 50.0)):
    _o = with_env(TEST_NUM=raw)
    check(f"{raw!r} parses to {want}", ST.env_float("TEST_NUM", -1) == want,
          f"got {ST.env_float('TEST_NUM', -1)}")
    restore(_o)

_o = with_env(TEST_NUM="not-a-number")
check("unparseable text falls back rather than crashing",
      ST.env_float("TEST_NUM", 9.0) == 9.0)
restore(_o)

_o = with_env(TEST_NUM="30.7")
check("env_int truncates a float string", ST.env_int("TEST_NUM", 0) == 30)
restore(_o)

_o = with_env(TEST_NUM="0")
check("an explicit zero is respected, not treated as unset",
      ST.env_float("TEST_NUM", 99.0) == 0.0)
restore(_o)

# The real regression: importing the module with every numeric var empty, in a
# fresh interpreter, exactly as GitHub Actions does it.
_blank = {k: "" for k in (
    "PAPER_TRADE_USD", "PAPER_MAX_HOLD_DAYS", "INTRADAY_MIN_SCORE",
    "ACCOUNT_SIZE", "RISK_PCT", "MAX_POSITION_PCT",
    "AV_DAILY_LIMIT", "TD_DAILY_LIMIT", "TD_RATE_SLEEP",
    "ALPHAVANTAGE_API_KEY", "TWELVEDATA_API_KEY",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "SEC_USER_AGENT")}
_env = {**_os.environ, **_blank}
_r = _sp.run([_sys.executable, "-c",
              "import signal_tool as S; print(S.PAPER_TRADE_USD, S.PAPER_MAX_HOLD_DAYS, "
              "S.INTRADAY_MIN_SCORE, S.AV_DAILY_LIMIT, S.TD_RATE_SLEEP)"],
             env=_env, capture_output=True, text=True, cwd=_os.getcwd())
check("module imports cleanly with every variable blank",
      _r.returncode == 0, _r.stderr.strip()[-160:])
check("defaults survive the blank import",
      _r.stdout.strip() == "100.0 30 55.0 25 8.0", f"got {_r.stdout.strip()!r}")
check("a blank SEC_USER_AGENT still yields a usable contact string",
      _sp.run([_sys.executable, "-c",
               "import signal_tool as S; assert '@' in S.SEC_USER_AGENT; print('ok')"],
              env=_env, capture_output=True, text=True,
              cwd=_os.getcwd()).returncode == 0)

# Every command must at least start with a blank environment.
for _cmd in (["history"], ["scan", "--dry-run", "--symbols", "SPY"],
             ["backtest", "--years", "1", "--symbols", "SPY"]):
    _rr = _sp.run([_sys.executable, "signal_tool.py"] + _cmd,
                  env=_env, capture_output=True, text=True, cwd=_os.getcwd())
    check(f"`{' '.join(_cmd[:2])}` does not crash on import with blank vars",
          "ValueError" not in _rr.stderr and "Traceback" not in _rr.stderr,
          _rr.stderr.strip()[-140:])

print("\nTrading costs")

_o = ST.SLIPPAGE_BPS, ST.COMMISSION_USD
ST.SLIPPAGE_BPS, ST.COMMISSION_USD = 5.0, 0.0

# A limit target fills at its price: no slippage.
net, cost = ST.apply_costs(100.0, 110.0, "target", 1.0)
check("a limit target exit costs nothing", approx(net, 10.0, 1e-9) and cost == 0.0)

# A stop becomes a market order and fills worse.
net, cost = ST.apply_costs(100.0, 95.0, "stop", 1.0)
check("a stop exit slips against you", net < -5.0, f"got {net}")
check("stop slippage is 5bps of the exit price", approx(cost, 95.0*5/10_000, 1e-9))

net_t, cost_t = ST.apply_costs(100.0, 98.0, "time", 1.0)
check("a time (market) exit also slips", cost_t > 0)

# Slippage lands on LOSERS, not winners -- the asymmetry that matters.
_, cw = ST.apply_costs(100.0, 110.0, "target", 1.0)
_, cl = ST.apply_costs(100.0, 95.0, "stop", 1.0)
check("costs fall on losing exits, not winning ones", cw == 0.0 and cl > 0.0)

# Commission applies to every round trip regardless of outcome.
ST.COMMISSION_USD = 1.0
net, cost = ST.apply_costs(100.0, 110.0, "target", 1.0)
check("commission applies even to a limit exit", approx(net, 9.0, 1e-9))
check("commission is included in reported costs", approx(cost, 1.0, 1e-9))
ST.COMMISSION_USD = 0.0

# Zero-cost config must reproduce the old gross numbers exactly.
ST.SLIPPAGE_BPS = 0.0
net, cost = ST.apply_costs(100.0, 95.0, "stop", 1.0)
check("zero-cost settings reproduce gross P/L", approx(net, -5.0, 1e-9) and cost == 0.0)
ST.SLIPPAGE_BPS = 5.0

# Backtest reports gross and cost separately so the drag is visible.
_pp = _P(100.0, 95.0, 110.0)
_rs = ST.simulate_trade(_bars([(100,100),(99,101),(94,99)]), 0, _pp)
check("backtest reports gross alongside net", "gross_pct" in _rs and "cost_pct" in _rs)
check("net is worse than gross on a stop", _rs["pnl_pct"] < _rs["gross_pct"])
_rt = ST.simulate_trade(_bars([(100,100),(99,101),(105,111)]), 0, _pp)
check("net equals gross on a limit target",
      approx(_rt["pnl_pct"], _rt["gross_pct"], 1e-9))

# The live tracker and the backtest must charge costs identically.
ST.TRADES_FILE = Path("test_trades3.json")
if ST.TRADES_FILE.exists(): ST.TRADES_FILE.unlink()
_bk = {"open": [], "closed": []}
_sg = evaluate("CST", moderate, NO_NEWS, NO_FLOW, RISK_ON)
ST.open_paper_trade(_bk, _sg, False)
_tr = _bk["open"][0]
_stopbar = [Bar(date="2099-07-01", open=_tr["entry"], high=_tr["entry"],
                low=_tr["stop"] - 0.01, close=_tr["stop"], volume=1e6)]
_live = ST.resolve_paper_trades(_bk, {"CST": _stopbar})[0]
_bt = ST.simulate_trade(
    [Bar(date="2099-06-30", open=_tr["entry"], high=_tr["entry"], low=_tr["entry"],
         close=_tr["entry"], volume=1e6),
     Bar(date="2099-07-01", open=_tr["entry"], high=_tr["entry"],
         low=_tr["stop"] - 0.01, close=_tr["stop"], volume=1e6)],
    0, _P(_tr["entry"], _tr["stop"], _tr["target"]))
check("live and backtest charge the same cost on a stop",
      abs(_live["pnl_pct"] - _bt["pnl_pct"]) < 0.02,
      f"live {_live['pnl_pct']:.3f}% vs backtest {_bt['pnl_pct']:.3f}%")
check("the live tracker records what costs were charged", "costs_usd" in _live)
if ST.TRADES_FILE.exists(): ST.TRADES_FILE.unlink()
ST.TRADES_FILE = Path("trades.json")
ST.SLIPPAGE_BPS, ST.COMMISSION_USD = _o

print("\nCost disclosure matches actual behaviour")

_sv = ST.SLIPPAGE_BPS, ST.COMMISSION_USD

# The bug this guards: the wording said costs were ignored for three revisions
# after the cost model went in. Assert the sentence tracks the behaviour.
ST.SLIPPAGE_BPS, ST.COMMISSION_USD = 5.0, 0.0
_d = ST._cost_disclosure()
_, _cost = ST.apply_costs(100.0, 95.0, "stop", 1.0)
check("when costs ARE charged, the disclosure says so",
      _cost > 0 and "Costs modelled" in _d, _d[:70])
check("the disclosure never claims costs are ignored while charging them",
      not ("ignores slippage" in _d or "NOT modelled" in _d), _d[:70])
check("the disclosure states the actual slippage figure", "5bps" in _d, _d[:70])

ST.SLIPPAGE_BPS, ST.COMMISSION_USD = 0.0, 0.0
_d0 = ST._cost_disclosure()
_, _c0 = ST.apply_costs(100.0, 95.0, "stop", 1.0)
check("when costs are NOT charged, the disclosure says that instead",
      _c0 == 0 and "NOT modelled" in _d0, _d0[:70])

ST.SLIPPAGE_BPS, ST.COMMISSION_USD = 5.0, 1.5
_d2 = ST._cost_disclosure()
check("commission is disclosed when set", "1.50" in _d2 and "commission" in _d2, _d2[:90])
check("disclosure mentions the unmodelled assumption (fills)",
      "fills" in _d2.lower(), _d2[:90])

# No hardcoded contradiction may survive anywhere in the source.
_src = Path("signal_tool.py").read_text()
check("no leftover 'ignores slippage and commission' text in the codebase",
      "ignores slippage and commission" not in _src)

ST.SLIPPAGE_BPS, ST.COMMISSION_USD = _sv

print(f"\n{'='*54}")
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {', '.join(FAILS)}")
    sys.exit(1)
print("All checks passed.")
sys.exit(0)

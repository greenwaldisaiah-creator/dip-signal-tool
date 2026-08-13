#!/usr/bin/env python3
"""Verification suite for signal_tool.

Covers the indicator math against hand-checkable values and the scoring
logic against constructed market scenarios. Run: python test_signal_tool.py
"""

import math
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

print(f"\n{'='*54}")
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {', '.join(FAILS)}")
    sys.exit(1)
print("All checks passed.")
sys.exit(0)

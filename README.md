# Dip Signal Tool

Scans a watchlist for oversold pullbacks, confirms them against news sentiment
and insider flow, sizes the position, and pushes alerts to Telegram.

## How the signal works

The model is deliberately **trigger-then-confirm**, not a blended score. Price
structure decides *whether to look*; the other layers decide *whether to trust it*.

**1. Technical trigger (0–100)** — a dip must fire here first.

| Factor | Weight | Rationale |
|---|---|---|
| RSI(14) oversold | 30 | Momentum exhaustion |
| Close below lower Bollinger band | 20 | Statistically stretched |
| Drawdown from 20-day high | 20 | Depth of the pullback |
| Volume spike vs 20-day average | 15 | Capitulation, not drift |
| Price above 200-day MA | 15 | Trend quality |

**2. Confirmation filter** — adjusts, never triggers.

- Bearish news flow: −7 to −15
- Positive news flow: +4 to +8
- Insiders net buying >$1M (90d): +10
- Insiders net selling >$1M (90d): −8

**3. Falling-knife veto** — a hard block, not a downgrade. If price is below the
200-day MA *and* more than 20% off its 50-day high, the signal is suppressed
outright. Only clearly positive news (≥ +0.15) can rescue it. This fires on price
alone: a news outage must never disable the model's main downside guard.

**4. Regime gate** — when SPY is below its 200-day MA, every tier threshold rises
by 10 points. Dip-buying in a downtrend is a different trade.

**Tiers:** WATCH ≥ 40 · STRONG ≥ 55 · HIGH CONVICTION ≥ 70

## Watchlist

21 symbols: 3 indices (SPY, QQQ, IWM), 8 megacaps (AAPL, MSFT, NVDA, AMZN,
GOOGL, META, TSLA, AVGO) and 10 sector ETFs (XLK, XLF, XLE, XLV, XLY, XLI, XLP,
XLU, XLB, XLC). Edit `INDICES` / `MEGACAPS` / `SECTORS` at the top of
`signal_tool.py`, and read the request budget below before adding more.

ETFs have no insiders and thin news coverage, so they score on technicals alone —
the tool detects this and degrades rather than inventing a confirmation signal.

## Intraday mode

```bash
python signal_tool.py scan --intraday
```

Uses 15-minute bars, **but keeps trend context on daily bars**. A 200-period
average of 15-minute bars spans about eight sessions — that's noise, not a trend.
So the oversold, stretched and volume factors come from intraday data, while the
moving averages, the falling-knife veto and the ATR stop stay on daily bars.

Intraday and end-of-day alerts dedupe independently, so a midday alert won't
suppress the evening one. The intraday resend window is 6 hours rather than 5 days.

## Position sizing

Set `ACCOUNT_SIZE` and alerts include a share count:

```
📐 221 sh ≈ $19,977 · risk $465 (0.47% capped)
```

Shares are set so a stop-out costs `RISK_PCT` of the account — meaning a volatile
name automatically gets a smaller position than a quiet one for the same dollar
risk. `MAX_POSITION_PCT` then caps concentration, so a very tight stop can't imply
an absurd position. When that cap binds, the alert reports the **realized** risk
percentage and says `capped`, rather than the target you configured. Leave
`ACCOUNT_SIZE` unset to omit sizing entirely.

## Request budget — read this before widening anything

Alpha Vantage's free tier allows **25 requests/day**, and that is the binding
constraint on this whole tool:

| Job | Requests |
|---|---|
| End-of-day scan | 21 price + 1 batched news = **22** |
| Intraday sweep (3 symbols) | **3** |
| **Total** | **25** |

That is exact, with **no headroom**. Intraday costs only one request per symbol
because it reuses the daily bars the EOD scan already cached. SEC EDGAR calls are
free and unlimited, and are only spent on names already near a tier boundary.

The tool counts its own requests and stops cleanly when the budget is spent,
rather than failing halfway through with a confusing rate-limit error. If you
want a wider watchlist or more frequent intraday sweeps, get a premium key
(75 req/min) and raise `AV_DAILY_LIMIT`.

## Data sources

| Layer | Source | Cost |
|---|---|---|
| Prices | Alpha Vantage `TIME_SERIES_DAILY` / `TIME_SERIES_INTRADAY` | Free |
| News sentiment | Alpha Vantage `NEWS_SENTIMENT` | Free, same key |
| Insider flow | SEC EDGAR Form 4 | Free, no key |
| Alerts | Telegram Bot API | Free |

**On "whale wallets":** that concept is on-chain and has no equity equivalent.
The stock analogue implemented here is **SEC Form 4 open-market insider
transactions** — actual cash executives chose to put in or take out. Grants,
option exercises and tax withholding are deliberately excluded because they carry
no directional information.

**On Twitter:** X's API starts around $200/month for search access. Alpha
Vantage's news endpoint already aggregates Bloomberg, MarketWatch, Reuters,
Barron's and others *with sentiment pre-scored per ticker*, on the key you
already have. The whitelist lives in `REPUTABLE_SOURCES`; aggregators and
promotional newsletters are filtered out. If you later get an X bearer token,
that feed can be added alongside.

## Setup

**New to Python? Read `SETUP.md` instead — it is a click-by-click walkthrough
that needs no Python and nothing installed.**

```bash
cp .env.example .env      # add your Telegram token + chat id
set -a && source .env && set +a

python signal_tool.py test-telegram          # verify credentials
python signal_tool.py scan --dry-run         # scan, print, don't send
python signal_tool.py scan                   # end-of-day scan and alert
python signal_tool.py scan --intraday        # 15-minute bars
python signal_tool.py backtest --years 3     # validate the rules
python test_signal_tool.py                   # 102-check verification suite
```

### Getting your Telegram chat id

Message your bot once, then open
`https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` and read
`result[0].message.chat.id`.

## Deploying on GitHub Actions

Runs free on GitHub's infrastructure — no machine of your own left on.

1. Create a **private** repo and push this folder to it.
2. **Settings → Secrets and variables → Actions → Secrets → New repository secret**, add:
   - `ALPHAVANTAGE_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `SEC_USER_AGENT` (e.g. `DipSignalTool your@email.com`)
3. Optionally, under the **Variables** tab (not Secrets), add `ACCOUNT_SIZE`,
   `RISK_PCT`, `MAX_POSITION_PCT` to enable position sizing.
4. **Actions** tab → **Dip Scan** → **Run workflow** to test immediately.

The workflow runs the EOD scan weekdays at 5:15pm ET and one intraday sweep at
11:30am ET, caches `cache/` and `state.json` between runs, and runs the test
suite on every push.

> **Note on schedules:** the crons are written in UTC for US Eastern *daylight*
> time. When the US falls back to EST in November, shift both to `15 22` and
> `30 16` to keep the same local times.

Alternatively, cron on any always-on machine:

```
15 17 * * 1-5 cd /path/to/stocksignal && /usr/bin/python3 signal_tool.py scan >> scan.log 2>&1
30 11 * * 1-5 cd /path/to/stocksignal && /usr/bin/python3 signal_tool.py scan --intraday >> scan.log 2>&1
```

## Design notes

- **Caching is load-bearing.** Prices cache 20h, intraday 25min, news 8h, SEC
  filings 24h, the CIK map 30 days. Form 4 XML is cached permanently — filings
  never change once submitted.
- **Graceful degradation.** If a source fails the scan continues on a stale cache
  and says so on stderr rather than dying. A missing news feed weakens
  confirmation but never fabricates it.
- **Dedupe.** The same ticker+tier won't re-alert inside the window, but a tier
  *upgrade* always sends — that's new information.

## Verification status

`test_signal_tool.py` runs **102 checks** covering indicator math against
hand-computable values (RSI on pure up/down/alternating/flat series, Bollinger,
drawdown, volume ratio), scoring across constructed scenarios, every confirmation
adjustment, the veto's failure modes, intraday blending, position-sizing edge
cases, the request budget, dedupe, and Telegram's 4096-character limit.

Four real bugs were caught this way and fixed:

1. A zero-volatility series collected free Bollinger points from collapsed bands.
2. The falling-knife veto was gated on news availability — it switched itself off
   exactly when the news feed was down.
3. `scan` fetched only 100 bars, too few for a 200-day MA, silently disabling the
   trend filter in production.
4. Position sizing displayed the configured risk target even when the position
   cap had cut actual risk to less than half of it.

**The empirical backtest has not been run.** Its mechanics are tested, but
validating the *thresholds* needs real multi-year history and network access.
Run `python signal_tool.py backtest --years 3` once deployed, and compare each
tier's forward return and win rate against the baseline row. **If a tier doesn't
beat the baseline, the thresholds are miscalibrated and should be retuned before
you trade it.** Treat the current weights as a starting hypothesis, not a
validated edge.

---

Signal output, not investment advice.

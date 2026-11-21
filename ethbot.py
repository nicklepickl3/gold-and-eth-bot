#!/usr/bin/env python3
"""
ethbot.py - Ethereum SIGNALS ONLY bot -> ntfy alerts, with timed entries.
Price data: Yahoo Finance (ETH-USD). No trades, no login, no token.

Flow: SETUP (enter at HH:MM) -> ENTER NOW or CANCELLED at that time.

pip install requests
env: NTFY_TOPIC, optional TIMEZONE, COPY_TOPIC, NEWS_RSS
python ethbot.py --check   (test data)
python ethbot.py           (run)
"""
import datetime as dt
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import requests

# ================= CONFIG =================
YAHOO = "ETH-USD"
NAME = "ETH"
BAR_MIN = 1
CANDLES = 120
FAST, SLOW, TREND = 9, 21, 50
RSI_N, ATR_N = 14, 14
HTF_MIN = 15                  # higher-timeframe trend check
CHECK_EVERY_SEC = 15
FRESH_MIN = 10

# timed entry
ENTRY_DELAY_MIN = 5           # enter this many minutes after the setup
ENTRY_WINDOW_MIN = 3          # entry stays valid this long after the entry time
MAX_CHASE_ATR = 0.5           # cancel if price moved more than this many ATRs from setup

# your time zone for alert times, e.g. "America/New_York", "Europe/London", "UTC"
TIMEZONE = os.environ.get("TIMEZONE", "UTC")

# "signal" = SETUP/ENTER/CANCELLED alerts, "status" = bot started, "alert" = price alerts
SEND_TO_NTFY = {"signal", "status"}

MIN_SCORE = 2                 # analyst checks passed, out of 3
MIN_CONFIDENCE = 55           # skip signals scoring below this (0-100)
RSI_OB, RSI_OS = 70, 30
# screener (crypto is 24/7, so volatility only)
ATR_MIN_PCT = 0.0002
ATR_MAX_PCT = 0.010
# signal limits
MAX_SIGNALS_DAY = 8
MIN_GAP_MIN = 15              # minimum minutes between setups
# news: blackout windows ("YYYY-MM-DD HH:MM" UTC, minutes each side)
NEWS_BLACKOUTS = [
    # ("2026-12-10 13:30", 30),
]
NEWS_RSS = os.environ.get("NEWS_RSS", "")
BULL = ["etf inflow", "etf approval", "ethereum upgrade", "eth rises", "eth jumps", "eth rallies",
        "rate cut", "institutional buying", "staking demand"]
BEAR = ["etf outflow", "exchange hack", "sec lawsuit", "eth falls", "eth drops", "eth slips",
        "rate hike", "crypto ban", "liquidations"]
COPY_TOPIC = os.environ.get("COPY_TOPIC", "")

STATE_FILE = "ethbot_state.json"
TOPIC = os.environ.get("NTFY_TOPIC", "")
UA = {"User-Agent": "Mozilla/5.0"}
# ==========================================

try:
    TZ = ZoneInfo(TIMEZONE)
except Exception:
    TZ = dt.timezone.utc


def fmt_time(ts):
    return dt.datetime.fromtimestamp(ts, TZ).strftime("%H:%M %Z")


def notify(title, msg, kind="alert", priority="default", tags=""):
    print(f"[{dt.datetime.now():%H:%M:%S}] {title}: {msg}", flush=True)
    if not TOPIC or kind not in SEND_TO_NTFY:
        return
    try:
        requests.post(f"https://ntfy.sh/{TOPIC}", data=msg.encode("utf-8"),
                      headers={"Title": title, "Priority": priority, "Tags": tags}, timeout=10)
    except Exception as e:
        print("ntfy failed:", e)


def default_state():
    return {
        "contract_id": None, "dir": None, "last_cross_epoch": 0,
        "price_above": None, "price_below": None,
        "paused": False, "kill": False, "pending": None,
        "day": "", "day_signals": 0, "last_signal_ts": 0,
        "last_price": None, "balance": None, "day_pnl": 0.0, "updated": None,
    }


def load_state():
    st = default_state()
    try:
        with open(STATE_FILE) as f:
            st.update(json.load(f))
    except Exception:
        pass
    return st


def save_state(st):
    st["updated"] = dt.datetime.now().isoformat(timespec="seconds")
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, STATE_FILE)


# ---------- DATA (Yahoo Finance) ----------
def get_candles(gran=BAR_MIN * 60, count=CANDLES):
    interval = {60: "1m", 900: "15m"}.get(gran, "1m")
    rng = "2d" if interval == "1m" else "5d"
    last = None
    for host in ("query1", "query2"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{YAHOO}",
                             params={"interval": interval, "range": rng},
                             headers=UA, timeout=15)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            out = []
            for i, ts in enumerate(res["timestamp"]):
                o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
                if None in (o, h, l, c):
                    continue
                out.append({"epoch": ts, "open": o, "high": h, "low": l, "close": c})
            return out[-count:]
        except Exception as e:
            last = e
    raise last


# ---------- INDICATORS ----------
def ema(values, n):
    k = 2 / (n + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def rsi(closes, n=RSI_N):
    ch = [closes[i] - closes[i - 1] for i in range(-n, 0)]
    g = sum(c for c in ch if c > 0) / n
    l = -sum(c for c in ch if c < 0) / n
    return 100.0 if l == 0 else 100 - 100 / (1 + g / l)


def atr(cs, n=ATR_N):
    trs = []
    for i in range(-n, 0):
        h, l, pc = float(cs[i]["high"]), float(cs[i]["low"]), float(cs[i - 1]["close"])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / n


def levels(sig, price, a):
    if sig == "up":
        return price - 1.5 * a, price + 3.0 * a
    return price + 1.5 * a, price - 3.0 * a


# ---------- ANALYST ----------
def analyst(closed, direction):
    closes = [float(c["close"]) for c in closed]
    f, s, t = ema(closes, FAST), ema(closes, SLOW), ema(closes, TREND)
    r = rsi(closes)
    if direction == "up":
        checks = [f[-1] > s[-1], closes[-1] > t[-1], 50 <= r < RSI_OB]
    else:
        checks = [f[-1] < s[-1], closes[-1] < t[-1], RSI_OS < r <= 50]
    return sum(checks), r


def confidence(candles, htf, sig, nb=0):
    """Graded 0-100 setup-quality score (not a win probability)."""
    closes = [float(c["close"]) for c in candles[:-1]]
    a = atr(candles)
    if a == 0:
        return 0
    f, s, t = ema(closes, FAST), ema(closes, SLOW), ema(closes, TREND)
    sgn = 1 if sig == "up" else -1
    clamp = lambda v, hi: max(0.0, min(hi, v))
    r = rsi(closes)
    hc = [float(c["close"]) for c in htf[:-1]]
    hf, hs = ema(hc, FAST), ema(hc, SLOW)
    ha = atr(htf) or a

    pts = 0.0
    pts += clamp(abs(f[-1] - s[-1]) / a * 200, 20)
    pts += clamp(sgn * (closes[-1] - t[-1]) / a * 10, 20)
    pts += clamp(20 - abs(r - (60 if sig == "up" else 40)), 20)
    pts += clamp(sgn * (s[-1] - s[-6]) / a * 30, 15)
    pts += clamp(sgn * (hf[-1] - hs[-1]) / ha * 15, 15)
    pts += 10 if nb == sgn else 5 if nb == 0 else 0
    return int(round(pts))


# ---------- SCREENER (24/7 market: volatility only) ----------
def screener(candles):
    pct = atr(candles) / float(candles[-1]["close"])
    if pct < ATR_MIN_PCT:
        return False, "too quiet"
    if pct > ATR_MAX_PCT:
        return False, "too volatile"
    return True, "ok"


# ---------- HIGHER-TIMEFRAME TREND ----------
def htf_trend(htf):
    closes = [float(c["close"]) for c in htf[:-1]]
    if len(closes) < TREND + 2:
        return None
    f, s, t = ema(closes, FAST), ema(closes, SLOW), ema(closes, TREND)
    if f[-1] > s[-1] and closes[-1] > t[-1]:
        return "up"
    if f[-1] < s[-1] and closes[-1] < t[-1]:
        return "down"
    return None


# ---------- NEWS ----------
_news_cache = (0, 0)


def news_blackout():
    now = dt.datetime.utcnow()
    for ts, mins in NEWS_BLACKOUTS:
        t = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M")
        if abs((now - t).total_seconds()) <= mins * 60:
            return True, ts
    return False, ""


def news_bias():
    global _news_cache
    if not NEWS_RSS:
        return 0
    if time.time() - _news_cache[0] < 600:
        return _news_cache[1]
    val = 0
    try:
        root = ET.fromstring(requests.get(NEWS_RSS, timeout=8).content)
        titles = [(i.findtext("title") or "").lower() for i in root.iter("item")][:15]
        bull = sum(any(k in t for k in BULL) for t in titles)
        bear = sum(any(k in t for k in BEAR) for t in titles)
        val = 1 if bull > bear + 1 else -1 if bear > bull + 1 else 0
    except Exception:
        pass
    _news_cache = (time.time(), val)
    return val


# ---------- COPYCAT (optional: follow another ntfy topic) ----------
def copycat():
    if not COPY_TOPIC:
        return 0
    last = 0
    try:
        r = requests.get(f"https://ntfy.sh/{COPY_TOPIC}/json?poll=1&since=15m", timeout=8)
        for line in r.text.strip().splitlines():
            m = json.loads(line)
            text = (m.get("title", "") + " " + m.get("message", "")).upper()
            if "BUY" in text:
                last = 1
            elif "SELL" in text:
                last = -1
    except Exception:
        pass
    return last


# ---------- GATES ----------
def gates(st, candles, direction):
    ok, why = screener(candles)
    if not ok:
        return False, "screener: " + why
    if st["day_signals"] >= MAX_SIGNALS_DAY:
        return False, "max signals today"
    if time.time() - st["last_signal_ts"] < MIN_GAP_MIN * 60:
        return False, "too soon after last signal"
    bad, when = news_blackout()
    if bad:
        return False, "news blackout " + when
    want = 1 if direction == "up" else -1
    nb = news_bias()
    if nb and nb != want:
        return False, "news bias against"
    score, _ = analyst(candles[:-1], direction)
    if score < MIN_SCORE:
        return False, f"analyst score {score}/3"
    if htf_trend(get_candles(HTF_MIN * 60, 80)) != direction:
        return False, f"{HTF_MIN}m trend disagrees"
    cc = copycat()
    if cc and cc != want:
        return False, "copycat source disagrees"
    return True, "ok"


# ---------- TIMED ENTRY ----------
def handle_pending(st, candles, price):
    p = st["pending"]
    now = time.time()
    if now < p["enter_at"]:
        return  # still waiting for the entry time

    word = "BUY" if p["dir"] == "up" else "SELL"
    valid_until = p["enter_at"] + ENTRY_WINDOW_MIN * 60
    closes = [float(c["close"]) for c in candles[:-1]]
    f, s = ema(closes, FAST), ema(closes, SLOW)
    still = f[-1] > s[-1] if p["dir"] == "up" else f[-1] < s[-1]
    moved = (price - p["price"]) * (1 if p["dir"] == "up" else -1)  # + = in trade direction
    limit = MAX_CHASE_ATR * p["atr"]

    reason = None
    if now > valid_until:
        reason = "entry window passed"
    elif not still:
        reason = "trend flipped, EMA cross is gone"
    elif moved > limit:
        reason = f"price already ran to {price:.2f}, don't chase"
    elif moved < -limit:
        reason = f"price moved against the setup ({price:.2f})"

    if reason:
        notify(f"CANCELLED: {word} {NAME} setup",
               f"Skip this one: {reason}.\nSetup price was {p['price']:.2f}.",
               "signal", "high", "x")
    else:
        a = atr(candles)
        sl_px, tp_px = levels(p["dir"], price, a)
        msg = (f"Enter {word} now at ~{price:.2f}\n"
               f"Valid until {fmt_time(valid_until)}\n"
               f"Stop:   {sl_px:.2f}\n"
               f"Target: {tp_px:.2f}  (2R)\n"
               f"Setup confidence: {p['conf']}%")
        notify(f"ENTER NOW: {word} {NAME}", msg, "signal", "urgent",
               "white_check_mark")
    st["pending"] = None


# ---------- MAIN LOOP ----------
def cycle(st):
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    if st["day"] != today:
        st.update(day=today, day_signals=0)

    candles = get_candles()
    price = float(candles[-1]["close"])
    st["last_price"] = price

    if st["price_above"] is not None and price >= st["price_above"]:
        notify("ALERT price", f"{NAME} {price:.2f} >= {st['price_above']}", "alert", "high", "bell")
        st["price_above"] = None
    if st["price_below"] is not None and price <= st["price_below"]:
        notify("ALERT price", f"{NAME} {price:.2f} <= {st['price_below']}", "alert", "high", "bell")
        st["price_below"] = None

    if st["kill"]:
        st["paused"], st["pending"], st["kill"] = True, None, False
    if st["paused"]:
        return

    if st.get("pending"):
        handle_pending(st, candles, price)
        return

    closed = candles[:-1]
    if len(closed) < TREND + 2:
        return
    if time.time() - (closed[-1]["epoch"] + BAR_MIN * 60) > FRESH_MIN * 60:
        return  # stale data
    closes = [float(c["close"]) for c in closed]
    f, s = ema(closes, FAST), ema(closes, SLOW)
    sig = None
    if f[-2] <= s[-2] and f[-1] > s[-1]:
        sig = "up"
    elif f[-2] >= s[-2] and f[-1] < s[-1]:
        sig = "down"
    if not sig or closed[-1]["epoch"] <= st["last_cross_epoch"]:
        return
    st["last_cross_epoch"] = closed[-1]["epoch"]

    ok, why = gates(st, candles, sig)
    if not ok:
        print(f"Skipped {sig} cross: {why}", flush=True)
        return

    htf = get_candles(HTF_MIN * 60, 80)
    conf = confidence(candles, htf, sig, news_bias())
    if conf < MIN_CONFIDENCE:
        print(f"Skipped {sig} cross: confidence {conf}%", flush=True)
        return

    # --- create the timed setup ---
    now = time.time()
    a = atr(candles)
    enter_at = math.ceil((now + ENTRY_DELAY_MIN * 60) / 60) * 60
    zone = MAX_CHASE_ATR * a
    sl_px, tp_px = levels(sig, price, a)
    r = rsi(closes)
    word = "BUY" if sig == "up" else "SELL"
    msg = (f"Do NOT enter now. Setup price: {price:.2f}\n"
           f"Enter at: {fmt_time(enter_at)} (in {ENTRY_DELAY_MIN} min)\n"
           f"Only if price is {price - zone:.2f} - {price + zone:.2f}\n"
           f"Est. stop {sl_px:.2f} | target {tp_px:.2f}\n"
           f"Confidence: {conf}% | EMA{FAST}/{SLOW} cross | RSI {r:.0f} | {HTF_MIN}m trend agrees\n"
           f"I'll send ENTER NOW or CANCELLED then.")
    notify(f"SETUP: {word} {NAME} - enter {fmt_time(enter_at)}", msg, "signal", "high",
           "hourglass_flowing_sand")
    st["pending"] = {"dir": sig, "price": price, "atr": a, "conf": conf,
                     "created": now, "enter_at": enter_at}
    st["day_signals"] += 1
    st["last_signal_ts"] = now


def main():
    if "--check" in sys.argv:
        print("Last candle:", get_candles()[-1])
        print("Time zone:", TIMEZONE, "| now:", fmt_time(time.time()))
        return
    notify("ETHBot running",
           f"Bot started | {NAME} | Yahoo Finance {YAHOO} | signals only | times in {TIMEZONE}",
           "status", "default", "robot")
    last_err = 0
    while True:
        try:
            st = load_state()
            cycle(st)
            save_state(st)
        except KeyboardInterrupt:
            print("Stopped.")
            return
        except Exception as e:
            if time.time() - last_err > 300:
                print(f"Bot error: {type(e).__name__}: {e} - retrying", flush=True)
                last_err = time.time()
            time.sleep(10)
            continue
        time.sleep(CHECK_EVERY_SEC)


if __name__ == "__main__":
    main()

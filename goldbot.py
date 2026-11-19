#!/usr/bin/env python3
"""
goldbot.py - Gold SIGNALS ONLY bot (Deriv price data -> ntfy alerts).
Places NO trades. A Deriv token with Read scope is enough.

pip install websockets requests
env: DERIV_TOKEN, NTFY_TOPIC, optional COPY_TOPIC, NEWS_RSS
python goldbot.py --check   (test login + gold symbol)
python goldbot.py           (run)
Works with goldctl.py (same state file).
"""
import asyncio
import datetime as dt
import itertools
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import requests
import websockets

# ================= CONFIG =================
SYMBOL = "frxXAUUSD"
BAR_MIN = 1
CANDLES = 120
FAST, SLOW, TREND = 9, 21, 50
RSI_N, ATR_N = 14, 14
HTF_MIN = 15                  # higher-timeframe trend check
CHECK_EVERY_SEC = 15
FRESH_MIN = 10

SEND_TO_NTFY = {"signal"}     # add "alert" to also get price alerts from goldctl

MIN_SCORE = 2                 # analyst checks passed, out of 3
MIN_CONFIDENCE = 55           # skip signals scoring below this (0-100)
RSI_OB, RSI_OS = 70, 30
# screener (UTC)
SESSION_UTC = (7, 20)
ATR_MIN_PCT = 0.0001
ATR_MAX_PCT = 0.003
# signal limits
MAX_SIGNALS_DAY = 8
MIN_GAP_MIN = 15              # minimum minutes between signals
# news: blackout windows ("YYYY-MM-DD HH:MM" UTC, minutes each side)
NEWS_BLACKOUTS = [
    # ("2026-12-10 13:30", 30),
]
NEWS_RSS = os.environ.get("NEWS_RSS", "")
BULL = ["rate cut", "safe haven", "weaker dollar", "dollar falls", "gold rises", "gold jumps", "gold gains"]
BEAR = ["rate hike", "stronger dollar", "dollar rises", "gold falls", "gold drops", "gold slips", "hawkish"]
COPY_TOPIC = os.environ.get("COPY_TOPIC", "")

STATE_FILE = "goldbot_state.json"
TOKEN = os.environ.get("DERIV_TOKEN", "")
TOPIC = os.environ.get("NTFY_TOPIC", "")
APP_ID = os.environ.get("DERIV_APP_ID", "1089")
WS_URL = os.environ.get("DERIV_WS_URL", f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}")
# ==========================================


class DerivError(Exception):
    pass


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
        "paused": False, "kill": False,
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


# ---------- SCREENER ----------
def screener(candles):
    now = dt.datetime.utcnow()
    if now.weekday() >= 5:
        return False, "weekend"
    if not SESSION_UTC[0] <= now.hour < SESSION_UTC[1]:
        return False, "outside session"
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


# ---------- DERIV (read-only market data) ----------
class Deriv:
    def __init__(self):
        self.ws = None
        self.ids = itertools.count(1)

    async def connect(self):
        self.ws = await websockets.connect(WS_URL, ping_interval=20)

    async def call(self, payload):
        rid = next(self.ids)
        await self.ws.send(json.dumps(dict(payload, req_id=rid)))
        while True:
            msg = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=30))
            if msg.get("req_id") == rid:
                if msg.get("error"):
                    e = msg["error"]
                    raise DerivError(f"{e.get('code')}: {e.get('message')}")
                return msg

    async def authorize(self):
        return (await self.call({"authorize": TOKEN}))["authorize"]

    async def candles(self, gran=BAR_MIN * 60, count=CANDLES):
        r = await self.call({"ticks_history": SYMBOL, "adjust_start_time": 1, "count": count,
                             "end": "latest", "granularity": gran, "style": "candles"})
        return r["candles"]


# ---------- MAIN LOOP ----------
async def gates(d, st, candles, direction):
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
    nb = await asyncio.to_thread(news_bias)
    if nb and nb != want:
        return False, "news bias against"
    score, _ = analyst(candles[:-1], direction)
    if score < MIN_SCORE:
        return False, f"analyst score {score}/3"
    if htf_trend(await d.candles(HTF_MIN * 60, 80)) != direction:
        return False, f"{HTF_MIN}m trend disagrees"
    cc = await asyncio.to_thread(copycat)
    if cc and cc != want:
        return False, "copycat source disagrees"
    return True, "ok"


async def cycle(d, st):
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    if st["day"] != today:
        st.update(day=today, day_signals=0)

    candles = await d.candles()
    price = float(candles[-1]["close"])
    st["last_price"] = price

    if st["price_above"] is not None and price >= st["price_above"]:
        notify("ALERT price", f"Gold {price:.2f} >= {st['price_above']}", "alert", "high", "bell")
        st["price_above"] = None
    if st["price_below"] is not None and price <= st["price_below"]:
        notify("ALERT price", f"Gold {price:.2f} <= {st['price_below']}", "alert", "high", "bell")
        st["price_below"] = None

    if st["kill"]:
        st["paused"], st["kill"] = True, False
    if st["paused"]:
        return

    closed = candles[:-1]
    if len(closed) < TREND + 2:
        return
    if time.time() - (closed[-1]["epoch"] + BAR_MIN * 60) > FRESH_MIN * 60:
        return  # stale data / market closed
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

    ok, why = await gates(d, st, candles, sig)
    if not ok:
        print(f"Skipped {sig} cross: {why}", flush=True)
        return

    htf = await d.candles(HTF_MIN * 60, 80)
    nb = await asyncio.to_thread(news_bias)
    conf = confidence(candles, htf, sig, nb)
    if conf < MIN_CONFIDENCE:
        print(f"Skipped {sig} cross: confidence {conf}%", flush=True)
        return

    word = "BUY" if sig == "up" else "SELL"
    a = atr(candles)
    r = rsi(closes)
    sl_px = price - 1.5 * a if sig == "up" else price + 1.5 * a
    tp_px = price + 3.0 * a if sig == "up" else price - 3.0 * a
    msg = (f"Entry:  {price:.2f}\n"
           f"Stop:   {sl_px:.2f}\n"
           f"Target: {tp_px:.2f}  (2R)\n"
           f"Confidence: {conf}%\n"
           f"EMA{FAST}/{SLOW} cross | RSI {r:.0f} | {HTF_MIN}m trend agrees")

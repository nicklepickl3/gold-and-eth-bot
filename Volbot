#!/usr/bin/env python3
"""
volbot.py - Volatility 10/25/50/75/100 Index SIGNALS ONLY bot -> ntfy alerts.
Price data: Deriv public WebSocket (no login, no token). Places NO trades.

pip install websockets requests
env: NTFY_TOPIC
python volbot.py --check   (test data for all 5 indices)
python volbot.py           (run)
"""
import asyncio
import datetime as dt
import itertools
import json
import os
import sys
import time

import requests
import websockets

# ================= CONFIG =================
SYMBOLS = {                       # symbol: (name, price decimals)
    "R_10": ("Volatility 10", 3),
    "R_25": ("Volatility 25", 3),
    "R_50": ("Volatility 50", 4),
    "R_75": ("Volatility 75", 4),
    "R_100": ("Volatility 100", 2),
}
BAR_MIN = 1
CANDLES = 120
FAST, SLOW, TREND = 9, 21, 50
RSI_N, ATR_N = 14, 14
HTF_MIN = 15                      # higher-timeframe trend check
CHECK_EVERY_SEC = 15
FRESH_MIN = 10

# "signal" = ENTER NOW alerts, "status" = bot started message
SEND_TO_NTFY = {"signal", "status"}

MIN_SCORE = 2                     # analyst checks passed, out of 3
MIN_CONFIDENCE = 55               # skip signals scoring below this (0-100)
RSI_OB, RSI_OS = 70, 30
MAX_SIGNALS_DAY = 8               # per index
MIN_GAP_MIN = 15                  # minutes between signals, per index

STATE_FILE = "volbot_state.json"
TOPIC = os.environ.get("NTFY_TOPIC", "")
APP_ID = os.environ.get("DERIV_APP_ID", "1089")
WS_URLS = [
    f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}",
    f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}",
]
# ==========================================


class DerivError(Exception):
    pass


def notify(title, msg, kind="signal", priority="default", tags=""):
    print(f"[{dt.datetime.now():%H:%M:%S}] {title}: {msg}", flush=True)
    if not TOPIC or kind not in SEND_TO_NTFY:
        return
    try:
        requests.post(f"https://ntfy.sh/{TOPIC}", data=msg.encode("utf-8"),
                      headers={"Title": title, "Priority": priority, "Tags": tags}, timeout=10)
    except Exception as e:
        print("ntfy failed:", e)


# ---------- STATE ----------
def load_state():
    st = {"paused": False, "kill": False, "syms": {}, "updated": None}
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


def analyst(closed, direction):
    closes = [float(c["close"]) for c in closed]
    f, s, t = ema(closes, FAST), ema(closes, SLOW), ema(closes, TREND)
    r = rsi(closes)
    if direction == "up":
        checks = [f[-1] > s[-1], closes[-1] > t[-1], 50 <= r < RSI_OB]
    else:
        checks = [f[-1] < s[-1], closes[-1] < t[-1], RSI_OS < r <= 50]
    return sum(checks), r


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


def confidence(candles, htf, sig):
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
    return int(round(pts * 100 / 90))


# ---------- DERIV (public market data, no login) ----------
class Deriv:
    def __init__(self):
        self.ws = None
        self.ids = itertools.count(1)

    async def connect(self):
        env = os.environ.get("DERIV_WS_URL")
        last = None
        for url in ([env] if env else WS_URLS):
            try:
                self.ws = await websockets.connect(
                    url, ping_interval=20, origin="https://deriv.com", open_timeout=15)
                return
            except Exception as e:
                last = e
                print(f"Deriv connect failed ({url.split('?')[0]}): {e}", flush=True)
        raise last

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

    async def candles(self, sym, gran=BAR_MIN * 60, count=CANDLES):
        r = await self.call({"ticks_history": sym, "adjust_start_time": 1, "count": count,
                             "end": "latest", "granularity": gran, "style": "candles"})
        return r["candles"]


# ---------- PER-INDEX CHECK ----------
async def check_symbol(d, st, sym):
    name, dec = SYMBOLS[sym]
    s = st["syms"].setdefault(sym, {"last_cross_epoch": 0, "day": "",
                                    "day_signals": 0, "last_signal_ts": 0})
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    if s["day"] != today:
        s.update(day=today, day_signals=0)

    candles = await d.candles(sym)
    closed = candles[:-1]
    if len(closed) < TREND + 2:
        return
    if time.time() - (closed[-1]["epoch"] + BAR_MIN * 60) > FRESH_MIN * 60:
        return  # stale data
    closes = [float(c["close"]) for c in closed]
    f, sl = ema(closes, FAST), ema(closes, SLOW)
    sig = None
    if f[-2] <= sl[-2] and f[-1] > sl[-1]:
        sig = "up"
    elif f[-2] >= sl[-2] and f[-1] < sl[-1]:
        sig = "down"
    if not sig or closed[-1]["epoch"] <= s["last_cross_epoch"]:
        return
    s["last_cross_epoch"] = closed[-1]["epoch"]

    if s["day_signals"] >= MAX_SIGNALS_DAY:
        return print(f"{name}: skipped {sig} cross: max signals today", flush=True)
    if time.time() - s["last_signal_ts"] < MIN_GAP_MIN * 60:
        return print(f"{name}: skipped {sig} cross: too soon after last signal", flush=True)
    score, r = analyst(closed, sig)
    if score < MIN_SCORE:
        return print(f"{name}: skipped {sig} cross: analyst {score}/3", flush=True)
    htf = await d.candles(sym, HTF_MIN * 60, 80)
    if htf_trend(htf) != sig:
        return print(f"{name}: skipped {sig} cross: {HTF_MIN}m trend disagrees", flush=True)
    conf = confidence(candles, htf, sig)
    if conf < MIN_CONFIDENCE:
        return print(f"{name}: skipped {sig} cross: confidence {conf}%", flush=True)

    price = float(candles[-1]["close"])
    a = atr(candles)
    sl_px = price - 1.5 * a if sig == "up" else price + 1.5 * a
    tp_px = price + 3.0 * a if sig == "up" else price - 3.0 * a
    word = "BUY" if sig == "up" else "SELL"
    msg = (f"Enter {word} now at ~{price:.{dec}f}\n"
           f"Stop:   {sl_px:.{dec}f}\n"
           f"Target: {tp_px:.{dec}f}  (2R)\n"
           f"Confidence: {conf}% | EMA{FAST}/{SLOW} cross | RSI {r:.0f} | {HTF_MIN}m trend agrees")
    notify(f"ENTER NOW: {word} {name} ({conf}%)", msg, "signal", "urgent",
           "chart_with_upwards_trend" if sig == "up" else "chart_with_downwards_trend")
    s["day_signals"] += 1
    s["last_signal_ts"] = time.time()


async def check_mode():
    d = Deriv()
    await d.connect()
    for sym, (name, dec) in SYMBOLS.items():
        c = (await d.candles(sym))[-1]
        print(f"{name} ({sym}): last close {float(c['close']):.{dec}f}")


async def main():
    if "--check" in sys.argv:
        return await check_mode()
    notify("VolBot running", f"Bot started | {len(SYMBOLS)} Volatility indices | signals only",
           "status", "default", "robot")
    last_err = 0
    while True:
        try:
            d = Deriv()
            await d.connect()
            while True:
                st = load_state()
                if st["kill"]:
                    st["paused"], st["kill"] = True, False
                if not st["paused"]:
                    for sym in SYMBOLS:
                        try:
                            await check_symbol(d, st, sym)
                        except DerivError as e:
                            print(f"{sym} error: {e}", flush=True)
                save_state(st)
                await asyncio.sleep(CHECK_EVERY_SEC)
        except KeyboardInterrupt:
            print("Stopped.")
            return
        except Exception as e:
            if time.time() - last_err > 300:
                print(f"Bot error: {type(e).__name__}: {e} - retrying", flush=True)
                last_err = time.time()
            await asyncio.sleep(10)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")

#!/usr/bin/env python3
"""
goldbot.py - EMA 9/21 crossover signal bot for gold on Deriv.
SIGNALS ONLY: Fetches candles, calculates EMA 9/21 crossover, and sends ntfy alerts.

Setup:
    pip install websockets requests
    export DERIV_TOKEN="your_deriv_token"
    export NTFY_TOPIC="your_ntfy_topic"
    python goldbot.py --check               (verify login + gold symbol)
    python goldbot.py                       (run the bot)
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

# ---------------- CONFIG ----------------
SYMBOL = "frxXAUUSD"          # Deriv gold/USD. Verify with --check
INTERVAL_MIN = 1              # candle size in minutes
CANDLES = 80                  # candles fetched per check
FAST, SLOW = 9, 21            # EMA lengths
CHECK_EVERY_SEC = 15
FRESH_MIN = 10                # ignore data older than this (market closed)

TOKEN = os.environ.get("DERIV_TOKEN", "")
TOPIC = os.environ.get("NTFY_TOPIC", "")
WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"  # Standard default App ID
# ----------------------------------------


class DerivError(Exception):
    pass


def notify(title, msg, priority="default", tags=""):
    print(f"[{dt.datetime.now():%H:%M:%S}] {title}: {msg}", flush=True)
    if not TOPIC or not title.startswith("SIGNAL"):
        return
    try:
        requests.post(
            f"https://ntfy.sh/{TOPIC}",
            data=msg.encode("utf-8"),
            headers={"Title": title, "Priority": priority, "Tags": tags},
            timeout=10,
        )
    except Exception as e:
        print("ntfy failed:", e)


def ema(values, n):
    k = 2 / (n + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


class Deriv:
    def __init__(self):
        self.ws = None
        self.ids = itertools.count(1)

    async def connect(self):
        self.ws = await websockets.connect(WS_URL, ping_interval=20)

    async def call(self, payload):
        rid = next(self.ids)
        payload = dict(payload, req_id=rid)
        await self.ws.send(json.dumps(payload))
        while True:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=30)
            msg = json.loads(raw)
            if msg.get("req_id") == rid:
                if msg.get("error"):
                    e = msg["error"]
                    raise DerivError(f"{e.get('code')}: {e.get('message')}")
                return msg

    async def authorize(self):
        r = await self.call({"authorize": TOKEN})
        return r["authorize"]

    async def candles(self):
        r = await self.call({
            "ticks_history": SYMBOL,
            "adjust_start_time": 1,
            "count": CANDLES,
            "end": "latest",
            "granularity": INTERVAL_MIN * 60,
            "style": "candles",
        })
        return r["candles"]


async def cycle(d, state):
    # 1) Fetch market data
    candles = await d.candles()
    price = float(candles[-1]["close"])

    # 2) Signal evaluation on CLOSED candles only
    closed = candles[:-1]
    if len(closed) < SLOW + 2:
        return

    # Check if candle data is fresh (market open)
    age = time.time() - (closed[-1]["epoch"] + INTERVAL_MIN * 60)
    if age > FRESH_MIN * 60:
        return

    closes = [float(c["close"]) for c in closed]
    f, s = ema(closes, FAST), ema(closes, SLOW)

    sig = None
    if f[-2] <= s[-2] and f[-1] > s[-1]:
        sig = "up"
    elif f[-2] >= s[-2] and f[-1] < s[-1]:
        sig = "down"

    # Avoid duplicate signals on the same candle epoch
    if not sig or closed[-1]["epoch"] <= state["last_cross_epoch"]:
        return

    state["last_cross_epoch"] = closed[-1]["epoch"]

    # 3) Dispatch Alert
    direction_str = "BUY" if sig == "up" else "SELL"
    tag_str = "chart_with_upwards_trend" if sig == "up" else "chart_with_downwards_trend"
    
    notify(
        f"SIGNAL: {direction_str} gold",
        f"Price ~{price:.2f} | EMA{FAST} crossed {'above' if sig == 'up' else 'below'} EMA{SLOW}",
        priority="high",
        tags=tag_str
    )


async def check_mode():
    d = Deriv()
    await d.connect()
    a = await d.authorize()
    print("Logged in:", a["loginid"], "| virtual:", a.get("is_virtual"),
          "| balance:", a["balance"], a["currency"])
    r = await d.call({"active_symbols": "brief", "product_type": "basic"})
    for s in r["active_symbols"]:
        name = json.dumps(s)
        if "XAU" in name:
            print("Gold symbol:", s.get("symbol") or s.get("underlying_symbol"),
                  "|", s.get("display_name") or s.get("underlying_symbol_name"))
    print("Last candle:", (await d.candles())[-1])


async def main():
    if not TOKEN:
        sys.exit("Set DERIV_TOKEN environment variable first.")
    if "--check" in sys.argv:
        return await check_mode()

    state = {"last_cross_epoch": 0}
    last_err = 0

    while True:
        try:
            d = Deriv()
            await d.connect()
            a = await d.authorize()
            notify(
                "GoldBot Signal Monitor Started",
                f"Account: {a['loginid']} | Symbol: {SYMBOL} | {INTERVAL_MIN}m EMA{FAST}/{SLOW}",
                tags="robot"
            )
            while True:
                await cycle(d, state)
                await asyncio.sleep(CHECK_EVERY_SEC)
        except Exception as e:
            if time.time() - last_err > 300:
                print(f"[{dt.datetime.now():%H:%M:%S}] Connection error: {e} - retrying in 10s...", flush=True)
                last_err = time.time()
            await asyncio.sleep(10)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")

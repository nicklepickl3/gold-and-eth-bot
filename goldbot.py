#!/usr/bin/env python3
"""
goldbot.py - EMA 9/21 crossover bot for gold on Deriv.
SIGNALS_ONLY = True  -> sends BUY/SELL alerts to ntfy, places NO trades.
SIGNALS_ONLY = False -> trades multiplier contracts on your Deriv account.
Control it with goldctl.py.

Setup:
    pip install websockets requests
    export DERIV_TOKEN="your_api_token"     (Read scope is enough for signals)
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
SIGNALS_ONLY = True           # True = ntfy alerts only, no trades

SYMBOL = "frxXAUUSD"          # Deriv gold/USD. Verify with --check
INTERVAL_MIN = 1              # candle size in minutes (BAR_MIN)
CANDLES = 80                  # candles fetched per check
FAST, SLOW = 9, 21            # EMA lengths
CHECK_EVERY_SEC = 15
FRESH_MIN = 10                # ignore data older than this (market closed)

MULTIPLIER = 100              # must be allowed for gold on your account
RISK = 0.01                   # max loss per trade = 1% of balance (stop loss)
TP_R = 2.0                    # take profit = 2x the risk
MAX_POS = 0.50                # stake never exceeds 50% of balance
MIN_STAKE = 1.0
MAX_DAILY_LOSS = 0.03         # pause for the day at -3% of start balance

STATE_FILE = "goldbot_state.json"
TOKEN = os.environ.get("DERIV_TOKEN", "")
TOPIC = os.environ.get("NTFY_TOPIC", "")
APP_ID = os.environ.get("DERIV_APP_ID", "1089")
WS_URL = os.environ.get("DERIV_WS_URL", f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}")
# ----------------------------------------


class DerivError(Exception):
    pass


def notify(title, msg, priority="default", tags=""):
    print(f"[{dt.datetime.now():%H:%M:%S}] {title}: {msg}", flush=True)
    if not TOPIC:
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


def default_state():
    return {
        "contract_id": None, "dir": None, "last_cross_epoch": 0,
        "price_above": None, "price_below": None,
        "paused": False, "kill": False,
        "day": "", "day_start_balance": 0.0, "day_pnl": 0.0,
        "last_price": None, "balance": None, "updated": None,
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

    async def balance(self):
        r = await self.call({"balance": 1})
        return float(r["balance"]["balance"])

    async def candles(self):
        r = await self.call({
            "ticks_history": SYMBOL, "adjust_start_time": 1,
            "count": CANDLES, "end": "latest",
            "granularity": INTERVAL_MIN * 60, "style": "candles",
        })
        return r["candles"]

    async def buy(self, direction, stake, sl, tp, currency):
        params = {
            "amount": round(stake, 2), "basis": "stake",
            "contract_type": "MULTUP" if direction == "up" else "MULTDOWN",
            "currency": currency, "multiplier": MULTIPLIER,
            "limit_order": {"take_profit": round(tp, 2), "stop_loss": round(sl, 2)},
        }
        last = None
        for key in ("underlying_symbol", "symbol"):   # new vs old API field name
            try:
                r = await self.call({"buy": 1, "price": round(stake, 2),
                                     "parameters": {**params, key: SYMBOL}})
                return r["buy"]
            except DerivError as e:
                last = e
        raise last

    async def sell(self, contract_id):
        return (await self.call({"sell": contract_id, "price": 0}))["sell"]

    async def contract(self, contract_id):
        r = await self.call({"proposal_open_contract": 1, "contract_id": contract_id})
        return r["proposal_open_contract"]


async def record_close(d, st, reason):
    """Fetch final result of the open contract, update P&L, notify."""
    cid = st["contract_id"]
    poc = await d.contract(cid)
    profit = float(poc.get("profit", 0))
    st["day_pnl"] += profit
    notify(
        f"Closed {st['dir'].upper()} ({reason})",
        f"P&L {profit:+.2f} | Day {st['day_pnl']:+.2f} | Contract {cid}",
        priority="high", tags="moneybag" if profit >= 0 else "warning",
    )
    st["contract_id"], st["dir"] = None, None


async def close_position(d, st, reason):
    try:
        await d.sell(st["contract_id"])
    except DerivError as e:
        notify("Sell issue", str(e), tags="warning")  # may already be sold
    await record_close(d, st, reason)


async def open_position(d, st, direction, price, balance, currency):
    risk_amt = balance * RISK
    stake = max(MIN_STAKE, risk_amt * 2)
    stake = min(stake, balance * MAX_POS)
    sl = min(risk_amt, stake)
    tp = risk_amt * TP_R
    b = await d.buy(direction, stake, sl, tp, currency)
    st["contract_id"], st["dir"] = b["contract_id"], direction
    notify(
        f"Opened {direction.upper()} gold",
        f"Price ~{price:.2f} | Stake {stake:.2f} | SL {sl:.2f} | TP {tp:.2f} "
        f"| x{MULTIPLIER} | Contract {b['contract_id']}",
        priority="high", tags="chart_with_upwards_trend" if direction == "up" else "chart_with_downwards_trend",
    )


async def cycle(d, st, currency):
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    bal = await d.balance()
    if st["day"] != today:
        st.update(day=today, day_start_balance=bal, day_pnl=0.0)
        if not SIGNALS_ONLY:
            notify("New day", f"Balance {bal:.2f} {currency}")
    st["balance"] = bal

    # 1) update open contract (stopped out / TP hit?)
    if st["contract_id"] and not SIGNALS_ONLY:
        poc = await d.contract(st["contract_id"])
        if poc.get("is_sold"):
            await record_close(d, st, "SL/TP hit")

    # 2) market data
    candles = await d.candles()
    price = float(candles[-1]["close"])
    st["last_price"] = price

    # 3) price alerts
    if st["price_above"] is not None and price >= st["price_above"]:
        notify("Price alert", f"Gold {price:.2f} >= {st['price_above']}", "high", "bell")
        st["price_above"] = None
    if st["price_below"] is not None and price <= st["price_below"]:
        notify("Price alert", f"Gold {price:.2f} <= {st['price_below']}", "high", "bell")
        st["price_below"] = None

    # 4) kill switch
    if st["kill"]:
        if st["contract_id"] and not SIGNALS_ONLY:
            await close_position(d, st, "kill switch")
        st["paused"], st["kill"] = True, False
        notify("KILL SWITCH", "Bot paused.", "urgent", "rotating_light")
        return

    if st["paused"]:
        return

    # 5) daily loss limit (trading mode only)
    if (not SIGNALS_ONLY and st["day_start_balance"]
            and st["day_pnl"] <= -MAX_DAILY_LOSS * st["day_start_balance"]):
        if st["contract_id"]:
            await close_position(d, st, "daily loss limit")
        st["paused"] = True
        notify("Daily loss limit", f"Day P&L {st['day_pnl']:.2f}. Bot paused.", "urgent", "rotating_light")
        return

    # 6) signal on CLOSED candles only
    closed = candles[:-1]
    if len(closed) < SLOW + 2:
        return
    age = time.time() - (closed[-1]["epoch"] + INTERVAL_MIN * 60)
    if age > FRESH_MIN * 60:
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

    # --- signals only: alert and stop here, no orders ---
    if SIGNALS_ONLY:
        notify(f"SIGNAL: {'BUY' if sig == 'up' else 'SELL'} gold",
               f"Price ~{price:.2f} | EMA{FAST} crossed {'above' if sig == 'up' else 'below'} EMA{SLOW}",
               priority="high",
               tags="chart_with_upwards_trend" if sig == "up" else "chart_with_downwards_trend")
        return

    if st["contract_id"] and st["dir"] != sig:
        await close_position(d, st, "reversal")
    if not st["contract_id"]:
        await open_position(d, st, sig, price, bal, currency)


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
        sys.exit("Set DERIV_TOKEN first.")
    if "--check" in sys.argv:
        return await check_mode()

    last_err = 0
    while True:
        try:
            d = Deriv()
            await d.connect()
            a = await d.authorize()
            mode = "SIGNALS ONLY" if SIGNALS_ONLY else ("DEMO" if a.get("is_virtual") else "REAL")
            notify("GoldBot started",
                   f"{mode} | {a['loginid']} | {SYMBOL} | {INTERVAL_MIN}m EMA{FAST}/{SLOW}",
                   tags="robot")
            while True:
                st = load_state()
                await cycle(d, st, a["currency"])
                save_state(st)
                await asyncio.sleep(CHECK_EVERY_SEC)
        except Exception as e:
            if time.time() - last_err > 300:
                notify("Bot error", f"{type(e).__name__}: {e} - retrying", "high", "warning")
                last_err = time.time()
            await asyncio.sleep(10)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")

import sys, time, json, os
import numpy as np, pandas as pd, requests, yfinance as yf

TICKER = "GC=F"
INTERVAL = "15m"
PERIOD = "60d"
TOPIC = os.environ.get("NTFY_TOPIC", "change-this-to-something-random-gold-x7q2")
CASH = 10000.0
MAX_POS = 0.50
RISK = 0.01
FEE = 0.0005
CHECK_EVERY_MIN = 15
PRICE_ABOVE = None
PRICE_BELOW = None
STATE_FILE = "goldbot_state.json"

def load():
    df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=True)
    c = df["Close"].squeeze().dropna()
    return c

def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, 1e-9))

def signals(c, st):
    t = st[0]
    if t == "sma":
        f, s = c.rolling(st[1]).mean(), c.rolling(st[2]).mean()
        E = (f > s) & (f.shift(1) <= s.shift(1)); X = f < s
    elif t == "rsi":
        r = rsi(c, st[1]); E = r < st[2]; X = r > st[3]
    else:
        n = st[1]
        E = c > c.shift(1).rolling(n).max()
        X = c < c.shift(1).rolling(max(3, n // 2)).min()
    return E.fillna(False).values, X.fillna(False).values

def label(st):
    if st[0] == "sma": return f"SMA {st[1]}/{st[2]} cross"
    if st[0] == "rsi": return f"RSI {st[1]} buy<{st[2]} sell>{st[3]}"
    return f"Breakout {st[1]}-bar"

def backtest(c, E, X, stop, rr):
    p = c.values; frac = min(MAX_POS, RISK / stop)
    cash, sh, ep, pe, px = CASH, 0.0, 0.0, False, False
    eq, trades = [], 0
    for i in range(len(p)):
        if sh == 0 and pe:

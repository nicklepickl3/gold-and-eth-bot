import sys, time, json, os
import numpy as np, pandas as pd, requests, yfinance as yf

TICKER = "GC=F"
INTERVAL = "1m"         # 1-minute bars for high-frequency trades
PERIOD = "7d"           # yfinance limit for 1m data
TOPIC = "goldbot-trader-x9k2p-secret-7711" # Updated secret topic name
CASH = 10000.0
MAX_POS = 0.50
RISK = 0.01
FEE = 0.0002
CHECK_EVERY_SEC = 15    # Check every 15 seconds
BAR_MIN = 1 
FRESH_MIN = 10
PRICE_ABOVE = None
PRICE_BELOW = None
STATE_FILE = "goldbot_state.json"

def load(period=PERIOD):
    df = yf.download(TICKER, period=period, interval=INTERVAL, progress=False, auto_adjust=True)
    return df["Close"].squeeze().dropna()

def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, 1e-9))

def signals(c, st):
    t = st[0]
    if t == "sma":
        f, s = c.rolling(st[1]).mean(), c.rolling(st[2]).mean()
        E = (f > s) & (f.shift(1) <= s.shift(1))
        X = f < s
    elif t == "rsi":
        r = rsi(c, st[1])
        E = r < st[2]
        X = r > st[3]
    elif t == "band":
        z = (c - c.rolling(st[1]).mean()) / c.rolling(st[1]).std()
        E = z < -st[2]
        X = z > 0
    else:
        n = st[1]
        E = c > c.shift(1).rolling(n).max()
        X = c < c.shift

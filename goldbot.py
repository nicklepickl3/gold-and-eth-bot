import sys, time, json, os
import numpy as np, pandas as pd, requests, yfinance as yf

TICKER = "GC=F"
INTERVAL = "5m"
PERIOD = "60d"
TOPIC = os.environ.get("NTFY_TOPIC", "change-this-to-something-random-gold-x7q2")
CASH = 10000.0
MAX_POS = 0.50
RISK = 0.01
FEE = 0.0002
CHECK_EVERY_SEC = 60
BAR_MIN = 5
FRESH_MIN = 20
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
        X = c < c.shift(1).rolling(max(3, n // 2)).min()
    return E.fillna(False).values, X.fillna(False).values

def label(st):
    if st[0] == "sma":
        return f"SMA {st[1]}/{st[2]} cross"
    if st[0] == "rsi":
        return f"RSI {st[1]} buy<{st[2]} sell>{st[3]}"
    if st[0] == "band":
        return f"Band {st[1]} bars, {st[2]} dev"
    return f"Breakout {st[1]}-bar"

def backtest(c, E, X, stop, rr):
    p = c.values
    frac = min(MAX_POS, RISK / stop)
    cash, sh, ep, pe, px = CASH, 0.0, 0.0, False, False
    eq, trades = [], 0
    for i in range(len(p)):
        buying = (sh == 0 and pe)
        selling = (sh > 0 and px)
        if buying:
            amt = cash * frac
            sh = amt * (1 - FEE) / p[i]
            cash -= amt
            ep = p[i]
        if selling:
            cash += sh * p[i] * (1 - FEE)
            sh = 0.0
            trades += 1
        pe = False
        px = False
        if sh > 0:
            px = bool(p[i] <= ep * (1 - stop) or p[i] >= ep * (1 + rr * stop) or X[i])
        else:
            pe = bool(E[i])
        eq.append(cash + sh * p[i])
    eq = np.array(eq)
    ret = eq[-1] / CASH - 1
    dd = (1 - eq / np.maximum.accumulate(eq)).max()
    return dict(ret=ret, dd=dd, n=trades, pe=pe, px=px, inpos=sh > 0, ep=ep, frac=frac)

def grid():
    S = [("sma", f, s) for f in (2, 3, 5, 8, 10) for s in (5, 8, 10, 20, 30, 50) if f < s]
    S += [("rsi", p, l, h) for p in (2, 4, 7, 14) for l in (25, 30, 35, 40, 45) for h in (50, 55, 60, 65)]
    S += [("brk", n) for n in (3, 5, 10, 20, 30)]
    S += [("band", n, k) for n in (10, 20, 40) for k in (0.5, 0.8, 1.0, 1.5)]
    return [(st, sp, rr) for st in S for sp in (0.002, 0.003, 0.005, 0.0075, 0.01) for rr in (1, 1.5, 2, 3)]

def optimize(c):
    cut = int(len(c) * 0.7)
    train, test = c.iloc[:cut], c.iloc[cut:]
    days = max(1.0, (c.index[-1] - c.index[0]).total_seconds() / 86400 * 5 / 7)
    rows = []
    for st, sp, rr in grid():
        a = backtest(train, *signals(train, st), sp, rr)
        if a["n"] < 10:
            continue
        b = backtest(test, *signals(test, st), sp, rr)
        score = a["ret"] / (a["dd"] + 0.05)
        perday = 2 * (a["n"] + b["n"]) / days
        rows.append(dict(st=st, stop=sp, rr=rr, train=a, test=b, score=score, perday=perday))
    tiers = [
        ("A", lambda r: r["train"]["ret"] > 0 and r["test"]["ret"] > 0 and r["test"]["dd"] <= 0.08),
        ("B", lambda r: r["train"]["ret"] > 0 and r["test"]["ret"] > -0.01 and r["test"]["dd"] <= 0.10),
        ("C", lambda r: r["train"]["ret"] > 0),
    ]
    for grade, good in tiers:
        for want in (4, 3, 2, 1):
            ok = [r for r in rows if r["perday"] >= want and r["test"]["n"] >= 3 and good(r)]
            if ok:
                for r in ok:
                    r["grade"] = grade
                ok.sort(key=lambda r: r["score"], reverse=True)
                return ok
    return []

def push(title, msg, prio="default"):
    try:
        requests.post(f"https://ntfy.sh/{TOPIC}", data=msg.encode("utf-8"),
                      headers={"Title": title, "Priority": prio}, timeout=15)
    except Exception as e:
        print("push failed:", e)

def load_

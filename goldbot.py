import sys, time, json, os
import numpy as np, pandas as pd, requests, yfinance as yf

TICKER = "GC=F"
INTERVAL = "1h"
PERIOD = "730d"
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
            amt = cash * frac; sh = amt * (1 - FEE) / p[i]; cash -= amt; ep = p[i]
        elif sh > 0 and px:
            cash += sh * p[i] * (1 - FEE); sh = 0.0; trades += 1
        pe = px = False
        if sh > 0:
            if p[i] <= ep * (1 - stop) or p[i] >= ep * (1 + rr * stop) or X[i]: px = True
        elif E[i]: pe = True
        eq.append(cash + sh * p[i])
    eq = np.array(eq); ret = eq[-1] / CASH - 1
    dd = (1 - eq / np.maximum.accumulate(eq)).max()
    return dict(ret=ret, dd=dd, n=trades, pe=pe, px=px, inpos=sh > 0, ep=ep, frac=frac)

def grid():
    S = [("sma", f, s) for f in (5, 8, 10, 15, 20) for s in (20, 30, 50, 100) if f < s]
    S += [("rsi", p, l, h) for p in (7, 14) for l in (25, 30, 35) for h in (55, 65, 70)]
    S += [("brk", n) for n in (10, 20, 30, 50)]
    return [(st, sp, rr) for st in S for sp in (0.01, 0.015, 0.02) for rr in (2, 3)]

def optimize(c):
    cut = int(len(c) * 0.7); train, test = c.iloc[:cut], c.iloc[cut:]
    rows = []
    for st, sp, rr in grid():
        a = backtest(train, *signals(train, st), sp, rr)
        if a["n"] < 4: continue
        b = backtest(test, *signals(test, st), sp, rr)
        rows.append(dict(st=st, stop=sp, rr=rr, train=a, test=b,
                         score=a["ret"] / (a["dd"] + 0.05)))
    ok = [r for r in rows if r["test"]["ret"] > 0 and r["test"]["n"] >= 2]
    ok.sort(key=lambda r: r["score"], reverse=True)
    return ok

def push(title, msg, prio="default"):
    try:
        requests.post(f"https://ntfy.sh/{TOPIC}", data=msg.encode("utf-8"),
                      headers={"Title": title, "Priority": prio}, timeout=15)
    except Exception as e:
        print("push failed:", e)

def load_state():
    return json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}

def save_state(s): json.dump(s, open(STATE_FILE, "w"))

def check(best, state):
    full = load(); live = float(full.iloc[-1]); c = full.iloc[:-1]
    st, sp, rr = best["st"], best["stop"], best["rr"]
    r = backtest(c, *signals(c, st), sp, rr)
    bar = str(c.index[-1]); name = f"{label(st)}, {sp*100:.1f}% stop, {rr}:1 target"
    size = CASH * r["frac"]
    if r["pe"] and state.get("bar") != bar:
        push("GOLD: BUY signal",
             f"{name}\nPrice ~{live:.2f}\nStop {live*(1-sp):.2f}  Target {live*(1+rr*sp):.2f}\n"
             f"Size ${size:.0f} of ${CASH:.0f}", "high")
        state["bar"] = bar
    elif r["px"] and state.get("bar") != bar:
        push("GOLD: SELL signal", f"{name}\nExit now, price ~{live:.2f}", "high"); state["bar"] = bar
    if PRICE_ABOVE and live >= PRICE_ABOVE and state.get("above") != PRICE_ABOVE:
        push("GOLD above level", f"{live:.2f} >= {PRICE_ABOVE}"); state["above"] = PRICE_ABOVE
    if PRICE_BELOW and live <= PRICE_BELOW and state.get("below") != PRICE_BELOW:
        push("GOLD below level", f"{live:.2f} <= {PRICE_BELOW}"); state["below"] = PRICE_BELOW
    save_state(state)
    print(time.strftime("%H:%M"), f"gold {live:.2f}", label(st),
          "BUY" if r["pe"] else "SELL" if r["px"] else "HOLD" if r["inpos"] else "WAIT")

def pick():
    res = optimize(load())
    if not res: raise SystemExit("No setup held up on unseen data. Try a different interval.")
    return res

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "test":
        push("Gold bot test", "If you see this, alerts work."); raise SystemExit
    res = pick()
    if mode == "backtest":
        for r in res[:10]:
            print(f"{label(r['st']):28} stop {r['stop']*100:.1f}% rr {r['rr']}  "
                  f"train {r['train']['ret']*100:6.1f}% (dd {r['train']['dd']*100:.1f}%)  "
                  f"unseen {r['test']['ret']*100:6.1f}% ({r['test']['n']} trades)")
        raise SystemExit
    if mode == "once":
        check(res[0], load_state()); raise SystemExit
    best, last_opt, state = res[0], time.time(), load_state()
    push("Gold bot started", f"Using {label(best['st'])}")
    while True:
        try:
            if time.time() - last_opt > 86400: best, last_opt = pick()[0], time.time()
            check(best, state)
        except Exception as e:
            print("error:", e)
        time.sleep(CHECK_EVERY_MIN * 60)

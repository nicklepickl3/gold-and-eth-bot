import sys, time, json, os
import numpy as np, pandas as pd, requests, yfinance as yf

TICKER = "GC=F"
INTERVAL = "1m"         # Changed from "5m" to "1m" for high-frequency short trades
PERIOD = "7d"           # yfinance limits 1m data to 7 days max
TOPIC = os.environ.get("NTFY_TOPIC", "change-this-to-something-random-gold-x7q2")
CASH = 10000.0
MAX_POS = 0.50
RISK = 0.01
FEE = 0.0002
CHECK_EVERY_SEC = 15    # Checked more frequently (every 15 seconds)
BAR_MIN = 1             # Adjusted for 1m bars
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
        X = c < c.shift(1).rolling(max(2, n // 2)).min()
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
    # Expanded with faster, ultra-short parameters for high frequency
    S = [("sma", f, s) for f in (1, 2, 3, 5) for s in (3, 5, 8, 12) if f < s]
    S += [("rsi", p, l, h) for p in (2, 3, 5) for l in (35, 40, 45) for h in (55, 60, 65)]
    S += [("brk", n) for n in (2, 3, 5, 8)]
    S += [("band", n, k) for n in (5, 10, 15) for k in (0.3, 0.5, 0.8)]
    return [(st, sp, rr) for st in S for sp in (0.001, 0.0015, 0.002, 0.003) for rr in (0.8, 1, 1.5)]

def optimize(c):
    cut = int(len(c) * 0.7)
    train, test = c.iloc[:cut], c.iloc[cut:]
    days = max(1.0, (c.index[-1] - c.index[0]).total_seconds() / 86400 * 5 / 7)
    rows = []
    for st, sp, rr in grid():
        a = backtest(train, *signals(train, st), sp, rr)
        if a["n"] < 5:  # Lowered minimum trade requirement for fast setups
            continue
        b = backtest(test, *signals(test, st), sp, rr)
        score = a["ret"] / (a["dd"] + 0.05)
        perday = 2 * (a["n"] + b["n"]) / days
        rows.append(dict(st=st, stop=sp, rr=rr, train=a, test=b, score=score, perday=perday))
    
    # Relaxed tier constraints to allow high-frequency execution
    tiers = [
        ("A", lambda r: r["train"]["ret"] > -0.01 and r["test"]["ret"] > -0.02 and r["test"]["dd"] <= 0.12),
        ("B", lambda r: r["train"]["ret"] > -0.03 and r["test"]["dd"] <= 0.15),
        ("C", lambda r: True),
    ]
    for grade, good in tiers:
        for want in (10, 5, 2, 1):
            ok = [r for r in rows if r["perday"] >= want and r["test"]["n"] >= 2 and good(r)]
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

def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {}

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"))

def check(best, state):
    full = load("1d") # Fetch quick recent 1m/few-hours data for checking live updates
    live = float(full.iloc[-1])
    c = full.iloc[:-1]
    st, sp, rr = best["st"], best["stop"], best["rr"]
    r = backtest(c, *signals(c, st), sp, rr)
    bar = str(c.index[-1])
    ts = c.index[-1]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    age = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60 - BAR_MIN
    name = f"{label(st)}, {sp*100:.2f}% stop, {rr}:1 target"
    size = CASH * r["frac"]
    fresh = state.get("bar") != bar and age <= FRESH_MIN
    
    if r["pe"] and fresh:
        msg = (f"{name}\nPrice ~{live:.2f}\nStop {live*(1-sp):.2f}  Target {live*(1+rr*sp):.2f}\n"
               f"Size ${size:.0f} of ${CASH:.0f}\nConfidence: {best['grade']}")
        push("GOLD: BUY signal", msg, "high")
        state["bar"] = bar
    elif r["px"] and fresh:
        push("GOLD: SELL signal", f"{name}\nExit now, price ~{live:.2f}\nConfidence: {best['grade']}", "high")
        state["bar"] = bar
        
    save_state(state)
    status = "BUY" if r["pe"] else "SELL" if r["px"] else "HOLD" if r["inpos"] else "WAIT"
    print(time.strftime("%H:%M:%S"), f"gold {live:.2f}", label(st), status, f"~{best['perday']:.1f} signals/day", "grade", best["grade"], flush=True)

def pick():
    res = optimize(load())
    if not res:
        raise SystemExit("No setup found. Try loosening constraints further.")
    return res

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "test":
        push("Gold bot test", "If you see this, alerts work.")
        raise SystemExit
    res = pick()
    if mode == "backtest":
        for r in res[:10]:
            print(label(r["st"]), r["stop"], r["rr"], round(r["train"]["ret"] * 100, 1), round(r["test"]["ret"] * 100, 1))
        raise SystemExit
    if mode == "once":
        check(res[0], load_state())
        raise SystemExit
    best, last_opt, state = res[0], time.time(), load_state()
    print("Using", label(best["st"]), best["stop"], best["rr"], f"~{best['perday']:.1f} signals/day", "grade", best["grade"], flush=True)
    push("Gold bot running", f"Using {label(best['st'])}, ~{best['perday']:.1f} signals/day, grade {best['grade']}", "low")
    while True:
        t0 = time.time()
        try:
            if time.time() - last_opt > 86400:
                best, last_opt = pick()[0], time.time()
            check(best, state)
        except Exception as e:
            print("error:", e, flush=True)
        time.sleep(max(5, CHECK_EVERY_SEC - (time.time() - t0)))

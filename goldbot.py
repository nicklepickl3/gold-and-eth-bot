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

def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"inpos": False, "ep": 0.0, "stop_price": 0.0, "target_price": 0.0, "bar": ""}

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"))

def check(best, state):
    full = load("5d")
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

    inpos = state.get("inpos", False)
    ep = state.get("ep", 0.0)
    stop_price = state.get("stop_price", 0.0)
    target_price = state.get("target_price", 0.0)

    # Check price level alerts
    if PRICE_ABOVE and live >= PRICE_ABOVE and state.get("above") != PRICE_ABOVE:
        push("GOLD above level", f"{live:.2f} >= {PRICE_ABOVE}")
        state["above"] = PRICE_ABOVE
    if PRICE_BELOW and live <= PRICE_BELOW and state.get("below") != PRICE_BELOW:
        push("GOLD below level", f"{live:.2f} <= {PRICE_BELOW}")
        state["below"] = PRICE_BELOW

    # Trading logic & notifications based on actual position state
    if not inpos:
        # Looking for buy entry
        if r["pe"] and fresh:
            ep = live
            stop_price = ep * (1 - sp)
            target_price = ep * (1 + rr * sp)
            
            state["inpos"] = True
            state["ep"] = ep
            state["stop_price"] = stop_price
            state["target_price"] = target_price
            state["bar"] = bar
            
            msg = (f"Strategy: {name}\n"
                   f"Entry Price: ~{ep:.2f}\n"
                   f"Stop Loss: {stop_price:.2f} (-{sp*100:.2f}%)\n"
                   f"Take Profit: {target_price:.2f} (+{rr*sp*100:.2f}%)\n"
                   f"Position Size: ${size:.0f} of ${CASH:.0f}\n"
                   f"Confidence: Grade {best['grade']} (A best, C weakest)")
            push("BUY GOLD NOW", msg, "high")
    else:
        # Currently in a position, monitor exits (Stop Loss, Take Profit, Technical Exit)
        pnl_pct = (live - ep) / ep * 100
        pnl_dlr = (live - ep) * (CASH * r["frac"] / ep)
        
        hit_stop = live <= stop_price
        hit_target = live >= target_price
        tech_exit = r["px"] and fresh

        if hit_stop:
            state["inpos"] = False
            state["bar"] = bar
            msg = (f"Strategy: {name}\n"
                   f"Exit Price: ~{live:.2f} (Stop Loss Hit)\n"
                   f"Entry Price: {ep:.2f}\n"
                   f"P&L: {pnl_pct:+.2f}% (${pnl_dlr:+.0f})")
            push("STOP LOSS", msg, "high")
        elif hit_target:
            state["inpos"] = False
            state["bar"] = bar
            msg = (f"Strategy: {name}\n"
                   f"Exit Price: ~{live:.2f} (Take Profit Hit)\n"
                   f"Entry Price: {ep:.2f}\n"
                   f"P&L: {pnl_pct:+.2f}% (${pnl_dlr:+.0f})")
            push("TAKE PROFIT", msg, "high")
        elif tech_exit:
            state["inpos"] = False
            state["bar"] = bar
            msg = (f"Strategy: {name}\n"
                   f"Exit Price: ~{live:.2f} (Technical Exit Signal)\n"
                   f"Entry Price: {ep:.2f}\n"
                   f"P&L: {pnl_pct:+.2f}% (${pnl_dlr:+.0f})")
            push("EXIT TRADE", msg, "high")

    save_state(state)

    # Console status output
    if state.get("inpos"):
        current_pnl = (live - ep) / ep * 100
        status = f"IN POSITION (Entry: {ep:.2f} | Live: {live:.2f} | P&L: {current_pnl:+.2f}%)"
    else:
        status = "WAITING FOR BUY"
        
    if age > 60:
        status += " (market closed or data stale)"
        
    print(time.strftime("%H:%M:%S"), f"gold {live:.2f}", label(st), status, f"~{best['perday']:.1f} signals/day", "grade", best["grade"], flush=True)

def pick():
    res = optimize(load())
    if not res:
        raise SystemExit("No setup held up on unseen data. Try a different interval.")
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

import sys, time, json, os
import numpy as np, pandas as pd, requests, yfinance as yf

TICKER = "ETH-USD"
ASSET = "ETH"
INTERVAL = "5m"
PERIOD = "60d"
TOPIC = os.environ.get("NTFY_TOPIC", "change-this-to-something-random-gold-x7q2")
CASH = 10000.0
MAX_POS = 0.50
RISK = 0.01
FEE = 0.0006
CHECK_EVERY_SEC = 60
BAR_MIN = 5
FRESH_MIN = 20
PRICE_ABOVE = None
PRICE_BELOW = None
STATE_FILE = "ethbot_state.json"

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

def backtest(c, E, X, stop, rr, hold=9999):
    p = c.values
    frac = min(MAX_POS, RISK / stop)
    cash, sh, ep, pe, px = CASH, 0.0, 0.0, False, False
    eq, trades, entry_i, held = [], 0, -1, 0
    for i in range(len(p)):
        buying = (sh == 0 and pe)
        selling = (sh > 0 and px)
        if buying:
            amt = cash * frac
            sh = amt * (1 - FEE) / p[i]
            cash -= amt
            ep = p[i]
            entry_i = i
            held = 0
        if selling:
            cash += sh * p[i] * (1 - FEE)
            sh = 0.0
            trades += 1
        pe = False
        px = False
        if sh > 0:
            held += 1
            px = bool(p[i] <= ep * (1 - stop) or p[i] >= ep * (1 + rr * stop) or X[i] or held >= hold)
        else:
            pe = bool(E[i])
        eq.append(cash + sh * p[i])
    eq = np.array(eq)
    ret = eq[-1] / CASH - 1
    dd = (1 - eq / np.maximum.accumulate(eq)).max()
    return dict(ret=ret, dd=dd, n=trades, pe=pe, px=px, inpos=sh > 0, ep=ep, frac=frac, entry_i=entry_i)

def grid():
    S = [("sma", f, s) for f in (2, 3, 5, 8, 10) for s in (5, 8, 10, 20, 30, 50) if f < s]
    S += [("rsi", p, l, h) for p in (2, 4, 7, 14) for l in (25, 30, 35, 40, 45) for h in (50, 55, 60, 65)]
    S += [("brk", n) for n in (3, 5, 10, 20, 30)]
    S += [("band", n, k) for n in (10, 20, 40) for k in (0.3, 0.5, 0.8, 1.0)]
    return [(st, sp, rr, hd) for st in S for sp in (0.004, 0.006, 0.01, 0.015) for rr in (1, 1.5, 2) for hd in (8, 16, 36, 72)]

def optimize(c):
    cut = int(len(c) * 0.7)
    train, test = c.iloc[:cut], c.iloc[cut:]
    days = max(1.0, (c.index[-1] - c.index[0]).total_seconds() / 86400)
    rows = []
    for st, sp, rr, hd in grid():
        a = backtest(train, *signals(train, st), sp, rr, hd)
        if a["n"] < 10:
            continue
        b = backtest(test, *signals(test, st), sp, rr, hd)
        score = a["ret"] / (a["dd"] + 0.05)
        perday = 2 * (a["n"] + b["n"]) / days
        rows.append(dict(st=st, stop=sp, rr=rr, hold=hd, train=a, test=b, score=score, perday=perday))
    good = {
        "A": lambda r: r["train"]["ret"] > 0 and r["test"]["ret"] > 0 and r["test"]["dd"] <= 0.08,
        "B": lambda r: r["train"]["ret"] > 0 and r["test"]["ret"] > -0.01 and r["test"]["dd"] <= 0.10,
        "C": lambda r: r["train"]["ret"] > 0,
    }
    order = [(g, w) for w in (8, 6, 4, 3, 2, 1) for g in ("A", "B")]
    order += [("C", w) for w in (8, 6, 4, 3, 2, 1)]
    for grade, want in order:
        ok = [r for r in rows if r["perday"] >= want and r["test"]["n"] >= 3 and good[grade](r)]
        if ok:
            for r in ok:
                r["grade"] = grade
            ok.sort(key=lambda r: r["score"], reverse=True)
            return ok
    return []

def push(title, msg, prio="default", tags=""):
    try:
        requests.post(f"https://ntfy.sh/{TOPIC}", data=msg.encode("utf-8"),
                      headers={"Title": title, "Priority": prio, "Tags": tags}, timeout=15)
    except Exception as e:
        print("push failed:", e)

def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {}

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"))

def check(best, state):
    full = load("5d")
    live = float(full.iloc[-1])
    c = full.iloc[:-1]
    st, sp, rr, hd = best["st"], best["stop"], best["rr"], best["hold"]
    E, X = signals(c, st)
    bar = str(c.index[-1])
    ts = c.index[-1]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    age = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60 - BAR_MIN
    name = f"{label(st)}, {sp*100:.2f}% stop, {rr}:1 target, max {hd*BAR_MIN} min"
    pos = state.get("pos")
    if pos is None:
        if E[-1] and state.get("bar") != bar and age <= FRESH_MIN:
            stop_p, tgt_p = live * (1 - sp), live * (1 + rr * sp)
            size = CASH * min(MAX_POS, RISK / sp)
            msg = (f"Open a BUY on {ASSET} now.\n"
                   f"Price: about {live:.2f}\n"
                   f"Set STOP LOSS at: {stop_p:.2f}\n"
                   f"Set TAKE PROFIT at: {tgt_p:.2f}\n"
                   f"Suggested size: ${size:.0f} of ${CASH:.0f}\n"
                   f"Quick trade: get out within {hd*BAR_MIN} minutes.\n"
                   f"I will message you when to get out.\n"
                   f"Confidence: {best['grade']} (A best, C weakest)\n"
                   f"Setup: {name}")
            push(f"BUY {ASSET} NOW", msg, "high", "green_circle")
            state["pos"] = {"entry": live, "stop": stop_p, "target": tgt_p, "bar": bar, "seen": bar, "n": 0}
            state["bar"] = bar
    else:
        why = None
        if pos.get("seen") != bar:
            pos["seen"] = bar
            pos["n"] = pos.get("n", 0) + 1
        if live <= pos["stop"]:
            why = (f"STOP LOSS - SELL {ASSET} NOW", f"Price hit your stop. Close your {ASSET} BUY now to limit the loss.", "red_circle")
        elif live >= pos["target"]:
            why = (f"TAKE PROFIT - SELL {ASSET} NOW", f"Price hit your target. Close your {ASSET} BUY now to lock in the profit.", "white_check_mark")
        elif X[-1] and bar != pos["bar"]:
            why = (f"EXIT TRADE - SELL {ASSET} NOW", f"The setup says the move is over. Close your {ASSET} BUY now.", "warning")
        elif pos.get("n", 0) >= hd:
            why = (f"EXIT TRADE - SELL {ASSET} NOW", f"Time is up on this quick trade. Close your {ASSET} BUY now.", "warning")
        if why:
            chg = (live / pos["entry"] - 1) * 100
            msg = (f"{why[1]}\n"
                   f"You bought at about {pos['entry']:.2f}\n"
                   f"Price now: about {live:.2f} ({chg:+.2f}%)\n"
                   f"If you never took the BUY, ignore this.")
            push(why[0], msg, "high", why[2])
            state["pos"] = None
            state["bar"] = bar
    if PRICE_ABOVE and live >= PRICE_ABOVE and state.get("above") != PRICE_ABOVE:
        push(f"{ASSET} above level", f"{live:.2f} >= {PRICE_ABOVE}")
        state["above"] = PRICE_ABOVE
    if PRICE_BELOW and live <= PRICE_BELOW and state.get("below") != PRICE_BELOW:
        push(f"{ASSET} below level", f"{live:.2f} <= {PRICE_BELOW}")
        state["below"] = PRICE_BELOW
    save_state(state)
    status = "BUY" if (pos is None and E[-1]) else "IN TRADE" if state.get("pos") else "WAIT"
    if age > 60:
        status += " (data stale)"
    print(time.strftime("%H:%M:%S"), f"{ASSET} {live:.2f}", label(st), status, f"~{best['perday']:.1f} signals/day", "grade", best["grade"], flush=True)

def adopt_open_trade(best, state):
    full = load("5d")
    c = full.iloc[:-1]
    sp, rr, hd = best["stop"], best["rr"], best["hold"]
    r = backtest(c, *signals(c, best["st"]), sp, rr, hd)
    if not r["inpos"] or r["entry_i"] < 0:
        return
    ts = c.index[r["entry_i"]]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    mins = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60
    if mins <= 300:
        ep = r["ep"]
        state["pos"] = {"entry": ep, "stop": ep * (1 - sp), "target": ep * (1 + rr * sp), "bar": str(c.index[-1]), "seen": str(c.index[-1]), "n": len(c) - 1 - r["entry_i"]}
        msg = (f"The bot restarted. An {ASSET} BUY from about {ep:.2f} may still be open.\n"
               f"If you took it: STOP LOSS {ep*(1-sp):.2f}, TAKE PROFIT {ep*(1+rr*sp):.2f}.\n"
               f"I will tell you when to get out. If you never bought, ignore this.")
        push(f"{ASSET} trade may still be open", msg, "default", "information_source")

def pick():
    return optimize(load())

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "test":
        push("ETH bot test", "If you see this, alerts work.")
        raise SystemExit
    res = pick()
    if not res and mode != "run":
        raise SystemExit("No profitable setup found right now.")
    told = False
    while not res:
        print("No profitable setup after fees. Checking again in 1 hour.", flush=True)
        if not told:
            push(f"{ASSET} bot waiting", "No setup is profitable after fees right now. I will keep checking every hour and start alerting when one is.", "low")
            told = True
        time.sleep(3600)
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
    try:
        adopt_open_trade(best, state)
    except Exception as e:
        print("adopt error:", e, flush=True)
    push("ETH bot running", f"Using {label(best['st'])}, ~{best['perday']:.1f} signals/day, grade {best['grade']}", "low")
    while True:
        t0 = time.time()
        try:
            if time.time() - last_opt > 86400:
                new = pick()
                if new:
                    best = new[0]
                last_opt = time.time()
            check(best, state)
        except Exception as e:
            print("error:", e, flush=True)
        time.sleep(max(5, CHECK_EVERY_SEC - (time.time() - t0)))

if __name__ == "__main__":
    main()

import sys, time, json, os
import numpy as np, pandas as pd, requests, yfinance as yf

try:
    from numba import njit
except Exception:
    def njit(*a, **k):
        if len(a) == 1 and callable(a[0]):
            return a[0]
        return lambda f: f

TICKER = "GC=F"           # gold futures
ASSET = "GOLD"
INTERVAL = "5m"
PERIOD = "30d"            # smaller = faster optimize
TOPIC = os.environ.get("NTFY_TOPIC", "PUT-YOUR-GOLD-NTFY-TOPIC-HERE")
CASH = 10000.0
MAX_POS = 0.50
RISK = 0.01
FEE = 0.0006
ALLOW_SHORT = True        # True = SELL signals open short trades; False = BUY only
CHECK_EVERY_SEC = 60
QUIET_RETRY_MIN = 5       # minutes between retries when no setup is found
RUN_MIN = 350             # bot stops after this many minutes
BAR_MIN = 5
FRESH_MIN = 20
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
    """Returns entry-long, exit-long, entry-short, exit-short arrays."""
    t = st[0]
    if t == "sma":
        f, s = c.rolling(st[1]).mean(), c.rolling(st[2]).mean()
        EL = (f > s) & (f.shift(1) <= s.shift(1))
        XL = f < s
        ES = (f < s) & (f.shift(1) >= s.shift(1))
        XS = f > s
    elif t == "rsi":
        r = rsi(c, st[1])
        EL = r < st[2]
        XL = r > st[3]
        ES = r > 100 - st[2]
        XS = r < 100 - st[3]
    elif t == "band":
        z = (c - c.rolling(st[1]).mean()) / c.rolling(st[1]).std()
        EL = z < -st[2]
        XL = z > 0
        ES = z > st[2]
        XS = z < 0
    else:
        n, m = st[1], max(3, st[1] // 2)
        EL = c > c.shift(1).rolling(n).max()
        XL = c < c.shift(1).rolling(m).min()
        ES = c < c.shift(1).rolling(n).min()
        XS = c > c.shift(1).rolling(m).max()
    out = [x.fillna(False).values.astype(np.bool_) for x in (EL, XL, ES, XS)]
    if not ALLOW_SHORT:
        out[2][:] = False
    return out


def label(st):
    if st[0] == "sma":
        return f"SMA {st[1]}/{st[2]} cross"
    if st[0] == "rsi":
        return f"RSI {st[1]} buy<{st[2]} sell>{100-st[2]}"
    if st[0] == "band":
        return f"Band {st[1]} bars, {st[2]} dev"
    return f"Breakout {st[1]}-bar"


@njit(cache=True)
def _bt(p, EL, XL, ES, XS, stop, rr, hold, frac, cash0, fee):
    cash, units, ep = cash0, 0.0, 0.0
    side, pend = 0, 0            # side: 1 long, -1 short, 0 flat
    px = False
    trades, entry_i, held = 0, -1, 0
    peak, maxdd, last = cash0, 0.0, cash0
    for i in range(len(p)):
        if side == 0 and pend != 0:
            amt = cash * frac
            units = amt * (1 - fee) / p[i]
            cash -= amt
            ep = p[i]
            side = pend
            entry_i = i
            held = 0
        elif side != 0 and px:
            if side == 1:
                val = units * p[i]
            else:
                val = units * (2 * ep - p[i])
            cash += val * (1 - fee)
            units = 0.0
            side = 0
            trades += 1
        pend = 0
        px = False
        if side == 1:
            held += 1
            px = (p[i] <= ep * (1 - stop)) or (p[i] >= ep * (1 + rr * stop)) or XL[i] or held >= hold
        elif side == -1:
            held += 1
            px = (p[i] >= ep * (1 + stop)) or (p[i] <= ep * (1 - rr * stop)) or XS[i] or held >= hold
        else:
            if EL[i]:
                pend = 1
            elif ES[i]:
                pend = -1
        if side == 1:
            last = cash + units * p[i]
        elif side == -1:
            last = cash + units * (2 * ep - p[i])
        else:
            last = cash
        if last > peak:
            peak = last
        dd = 1 - last / peak
        if dd > maxdd:
            maxdd = dd
    return last / cash0 - 1, maxdd, trades, side, ep, entry_i


def backtest(c, sig, stop, rr, hold=9999):
    frac = min(MAX_POS, RISK / stop)
    EL, XL, ES, XS = sig
    ret, dd, n, side, ep, ei = _bt(c.values.astype(np.float64), EL, XL, ES, XS, stop, rr, hold, frac, CASH, FEE)
    return dict(ret=ret, dd=dd, n=n, side=side, ep=ep, entry_i=ei)


def grid():
    S = [("sma", f, s) for f in (2, 3, 5, 8, 10) for s in (5, 8, 10, 20, 30, 50) if f < s]
    S += [("rsi", p, l, h) for p in (2, 4, 7, 14) for l in (25, 30, 35, 40, 45) for h in (50, 55, 60, 65)]
    S += [("brk", n) for n in (3, 5, 10, 20, 30)]
    S += [("band", n, k) for n in (10, 20, 40) for k in (0.3, 0.5, 0.8, 1.0)]
    return S


def optimize(c):
    cut = int(len(c) * 0.7)
    train, test = c.iloc[:cut], c.iloc[cut:]
    days = max(1.0, (c.index[-1] - c.index[0]).total_seconds() / 86400)
    rows = []
    for st in grid():
        sig_t = signals(train, st)       # signals computed once per strategy
        sig_s = signals(test, st)
        for sp in (0.004, 0.006, 0.01, 0.015):
            for rr in (1, 1.5, 2):
                for hd in (8, 16, 36, 72):
                    a = backtest(train, sig_t, sp, rr, hd)
                    if a["n"] < 10:
                        continue
                    b = backtest(test, sig_s, sp, rr, hd)
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


def push(title, msg, prio="high", tags=""):
    try:
        requests.post(f"https://ntfy.sh/{TOPIC}", data=msg.encode("utf-8"),
                      headers={"Title": title, "Priority": prio, "Tags": tags}, timeout=15)
    except Exception as e:
        print("push failed:", e, flush=True)


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE))
        except Exception:
            pass
    return {}


def save_state(s):
    json.dump(s, open(STATE_FILE, "w"))


def check(best, state):
    full = load("3d")
    live = float(full.iloc[-1])
    c = full.iloc[:-1]
    st, sp, rr, hd = best["st"], best["stop"], best["rr"], best["hold"]
    EL, XL, ES, XS = signals(c, st)
    bar = str(c.index[-1])
    ts = c.index[-1]
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    age = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60 - BAR_MIN
    name = f"{label(st)}, {sp*100:.2f}% stop, {rr}:1 target"
    pos = state.get("pos")
    if pos is None:
        side = 1 if EL[-1] else (-1 if ES[-1] else 0)
        if side and state.get("bar") != bar and age <= FRESH_MIN:
            if side == 1:
                stop_p, tgt_p = live * (1 - sp), live * (1 + rr * sp)
            else:
                stop_p, tgt_p = live * (1 + sp), live * (1 - rr * sp)
            word = "BUY" if side == 1 else "SELL"
            size = CASH * min(MAX_POS, RISK / sp)
            msg = (f"{name}\n"
                   f"Price ~{live:.2f}\n"
                   f"Stop {stop_p:.2f}  Target {tgt_p:.2f}\n"
                   f"Size ${size:.0f} of ${CASH:.0f}\n"
                   f"Confidence: {best['grade']} (A best, C weakest)")
            push(f"{ASSET}: {word} signal", msg, "high", "green_circle" if side == 1 else "red_circle")
            state["pos"] = {"side": side, "entry": live, "stop": stop_p, "target": tgt_p, "bar": bar, "seen": bar, "n": 0}
            state["bar"] = bar
    else:
        side = pos.get("side", 1)
        why = None
        if pos.get("seen") != bar:
            pos["seen"] = bar
            pos["n"] = pos.get("n", 0) + 1
        xnow = XL[-1] if side == 1 else XS[-1]
        hit_stop = live <= pos["stop"] if side == 1 else live >= pos["stop"]
        hit_tgt = live >= pos["target"] if side == 1 else live <= pos["target"]
        if hit_stop:
            why = "stop hit"
        elif hit_tgt:
            why = "target hit"
        elif xnow and bar != pos["bar"]:
            why = "setup over"
        elif pos.get("n", 0) >= hd:
            why = "time up"
        if why:
            word = "BUY" if side == 1 else "SELL"
            pnl = side * (live / pos["entry"] - 1) * 100
            msg = (f"{name}\n"
                   f"Close your {word} now ({why}), price ~{live:.2f}\n"
                   f"Entered ~{pos['entry']:.2f} ({pnl:+.2f}%)\n"
                   f"Confidence: {best['grade']}")
            push(f"{ASSET}: EXIT TRADE", msg, "high", "warning")
            state["pos"] = None
            state["bar"] = bar
    save_state(state)
    p = state.get("pos")
    status = ("IN " + ("BUY" if p.get("side", 1) == 1 else "SELL")) if p else ("BUY" if EL[-1] else "SELL" if ES[-1] else "WAIT")
    print(time.strftime("%H:%M:%S"), f"{ASSET} {live:.2f}", label(st), status, flush=True)


def pick():
    return optimize(load())


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "test":
        push(f"{ASSET} bot test", "If you see this, alerts work.", "default")
        raise SystemExit
    start = time.time()
    end = start + RUN_MIN * 60
    res = pick()
    if mode == "backtest":
        for r in res[:10]:
            print(label(r["st"]), r["stop"], r["rr"], round(r["train"]["ret"] * 100, 1), round(r["test"]["ret"] * 100, 1))
        raise SystemExit
    if mode != "once":
        push(f"{ASSET} bot: running", f"The bot is running for the next {RUN_MIN} min. You will get BUY, SELL and EXIT TRADE signals.", "low", "robot")
    while not res and time.time() < end:        # quiet period: log every minute, retry every 5 min
        for _ in range(QUIET_RETRY_MIN):
            print(time.strftime("%H:%M:%S"), f"{ASSET} no profitable setup, waiting", flush=True)
            time.sleep(60)
        res = pick()
    state = load_state()
    if res:
        best = res[0]
        print("Using", label(best["st"]), best["stop"], best["rr"], "grade", best["grade"], flush=True)
        if mode == "once":
            check(best, state)
            raise SystemExit
        last_opt = time.time()
        while time.time() < end:
            t0 = time.time()
            try:
                if time.time() - last_opt > 21600:   # re-optimize every 6h
                    new = pick()
                    if new:
                        best = new[0]
                    last_opt = time.time()
                check(best, state)
            except Exception as e:
                print("error:", e, flush=True)
            time.sleep(max(5, CHECK_EVERY_SEC - (time.time() - t0)))
    open_note = " A trade may still be open." if state.get("pos") else ""
    push(f"{ASSET} bot: timer finished",
         f"{RUN_MIN} min are up. The bot has stopped, restart it to keep getting signals.{open_note}",
         "default", "stopwatch")


if __name__ == "__main__":
    main()

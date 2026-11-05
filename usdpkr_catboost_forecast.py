"""
AI-Enabled FX Hedging Decision System for USD/PKR
Using NDF and Forward Premium Analysis for Treasury Risk Management
-------------------------------------------------------------------
CatBoost forecasting module  (Stage 1: macro factors  ->  Stage 2: USD/PKR)

WHAT THE SCRIPT DOES
  1. Asks you to select your train file (xlsx / csv) from your PC.
  2. Splits the data in time order: first 80 % = train, last 20 % = test (no shuffling).
  3. BACK-TEST on the 20 % test part (CatBoost vs a "no change" naive forecast).
  4. Retrains on ALL data and asks for a future month (e.g. Dec-2026, 06/2026) or a number
     of months ahead (e.g. 12).
  5. STAGE 1: CatBoost predicts every factor (reserves, REER, Brent, SBP rate, Fed rate,
     PK CPI, US CPI); the inflation differential is calculated = PK CPI - US CPI.
  6. STAGE 2: a second CatBoost model turns those predicted factors into the
     INTERBANK USD/PKR close forecast.
  7. Prints a table, saves forecast_<Mon><Year>.xlsx and two PNG charts next to your file.

Expected columns (names are matched by keywords, so small differences are fine):
  Date | Foreign reserves | REER | Brent Crude Oil Price | SBP Policy rates |
  Federal Funds Rate - Midpoint (%) | Pakistan CPI Inflation YoY (%) | US CPI - YoY Inflation % |
  Inflation rate differential | INTERBANK USD/PKR - Close

Usage:   python usdpkr_catboost_forecast.py                      (opens a file window)
         python usdpkr_catboost_forecast.py data.xlsx            (file given, asks month)
         python usdpkr_catboost_forecast.py data.xlsx 6 12       (no questions, 6 & 12 months)
"""

import os
import re
import sys
import subprocess
import importlib
import warnings


# ----------------------------------------------------------------------
# 0. AUTO-INSTALL MISSING PACKAGES
# ----------------------------------------------------------------------
def ensure_packages():
    needed = {"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib",
              "catboost": "catboost", "openpyxl": "openpyxl"}
    for mod, pip_name in needed.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"Installing missing package: {pip_name} ... (please wait)")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


ensure_packages()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

# ----------------------------------------------------------------------
# 1. SETTINGS
# ----------------------------------------------------------------------
TRAIN_RATIO = 0.80          # first 80 % of months = train, last 20 % = test
N_SEEDS = 3                 # CatBoost models with different seeds are averaged
WARMUP = 12                 # first 12 months are not used as origins (rolling features need them)

VARS = ["reserves", "reer", "brent", "sbp", "fed", "pk_cpi", "us_cpi", "infl_diff", "usdpkr"]
FACTORS = ["reserves", "reer", "brent", "sbp", "fed", "pk_cpi", "us_cpi"]   # Stage-1 targets
LOG_VARS = {"reserves", "reer", "brent", "usdpkr"}   # level-type -> log-return; rates -> difference

NICE = {
    "reserves": "Foreign reserves",
    "reer": "REER",
    "brent": "Brent Crude Oil Price",
    "sbp": "SBP Policy rate",
    "fed": "Federal Funds Rate (midpoint)",
    "pk_cpi": "Pakistan CPI Inflation YoY (%)",
    "us_cpi": "US CPI YoY Inflation (%)",
    "infl_diff": "Inflation rate differential",
    "usdpkr": "INTERBANK USD/PKR - Close",
}

KEYWORDS = [
    ("infl_diff", ["differential"]),
    ("usdpkr", ["usd/pkr"]),
    ("reserves", ["reserve"]),
    ("reer", ["reer", "real effective"]),
    ("brent", ["brent"]),
    ("sbp", ["sbp", "policy rate"]),
    ("fed", ["federal fund", "fed fund"]),
    ("pk_cpi", ["pakistan cpi", "pak cpi", "pakistan inflation"]),
    ("us_cpi", ["us cpi", "u.s. cpi"]),
]


# ----------------------------------------------------------------------
# 2. LOAD TRAIN FILE FROM PC
# ----------------------------------------------------------------------
def pick_file():
    if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]):
        return sys.argv[1]
    print("A window will open -> select your TRAIN data file (xlsx / csv) ...")
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select your TRAIN data file",
            filetypes=[("Excel / CSV", "*.xlsx *.xls *.csv"), ("All files", "*.*")])
        root.destroy()
        if path:
            return path
    except Exception:
        pass
    path = input("Type / paste the full path of your train file: ").strip().strip('"')
    if not path or not os.path.exists(path):
        raise SystemExit("File not found. Please run again and select the file.")
    return path


def load_data(path):
    raw = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    if "Date" not in raw.columns:
        raise ValueError("Your file must have a column named 'Date'.")

    mapping = {}
    for key, words in KEYWORDS:
        for col in raw.columns:
            if col == "Date" or col in mapping.values():
                continue
            if any(w in col.lower() for w in words):
                mapping[key] = col
                break

    missing = [v for v in VARS if v not in mapping and v != "infl_diff"]
    if missing:
        raise ValueError("These columns were not found in your file: "
                         + ", ".join(NICE[m] for m in missing))

    df = pd.DataFrame({"Date": pd.to_datetime(raw["Date"], errors="coerce")})
    for key in VARS:
        if key in mapping:
            df[key] = pd.to_numeric(raw[mapping[key]], errors="coerce")
    if "infl_diff" not in df.columns:
        df["infl_diff"] = df["pk_cpi"] - df["us_cpi"]

    df = (df.dropna(subset=["Date"]).sort_values("Date")
            .drop_duplicates("Date").set_index("Date"))
    df.index = df.index.to_period("M").to_timestamp()          # month-start dates
    df = df[~df.index.duplicated(keep="last")]
    df = df[VARS].interpolate().ffill().bfill()
    if (df[list(LOG_VARS)] <= 0).any().any():
        raise ValueError("Reserves / REER / Brent / USD-PKR must be positive numbers.")
    if len(df) < 48:
        raise ValueError("Need at least 48 monthly rows to train.")
    return df


# ----------------------------------------------------------------------
# 3. FEATURES  (only past / current information -> no leakage)
# ----------------------------------------------------------------------
def change(df, v, k):
    """Change of v over the PAST k months (log-return for level vars, difference for rates)."""
    return np.log(df[v]).diff(k) if v in LOG_VARS else df[v].diff(k)


def make_features(df):
    f = pd.DataFrame(index=df.index)
    for v in VARS:
        f[v] = df[v]
        d1 = change(df, v, 1)
        f[v + "_d1"] = d1
        f[v + "_d1_lag1"] = d1.shift(1)
        f[v + "_d3"] = change(df, v, 3)
        f[v + "_d6"] = change(df, v, 6)
        f[v + "_d12"] = change(df, v, 12)
        f[v + "_mean3"] = d1.rolling(3, min_periods=1).mean()
        f[v + "_mean6"] = d1.rolling(6, min_periods=1).mean()
    f["policy_spread"] = df["sbp"] - df["fed"]
    f["real_policy_rate"] = df["sbp"] - df["pk_cpi"]
    f["usdpkr_dev12"] = np.log(df["usdpkr"]) - np.log(df["usdpkr"].rolling(12, min_periods=3).mean())
    f["month"] = df.index.month
    return f.fillna(0.0)


def future_change(df, v, h):
    """Change of v over the NEXT h months."""
    if v in LOG_VARS:
        return np.log(df[v]).shift(-h) - np.log(df[v])
    return df[v].shift(-h) - df[v]


def stage2_features(cur, fut):
    """
    Stage-2 features for USD/PKR.  cur = factor levels NOW (t), fut = factor levels at the
    TARGET month (t+h); both are DataFrames with the same index.
    Training uses real future factor values, forecasting uses Stage-1 predicted values.
    """
    X = pd.DataFrame(index=cur.index)
    for v in FACTORS:
        X[v + "_chg_h"] = (np.log(fut[v]) - np.log(cur[v])) if v in LOG_VARS else (fut[v] - cur[v])
    X["infl_diff_fut"] = fut["pk_cpi"] - fut["us_cpi"]
    X["infl_diff_now"] = cur["pk_cpi"] - cur["us_cpi"]
    X["policy_spread_fut"] = fut["sbp"] - fut["fed"]
    X["real_policy_fut"] = fut["sbp"] - fut["pk_cpi"]
    return X


def stage2_state(feats):
    """Current USD/PKR momentum features (known at time t)."""
    cols = ["usdpkr_d1", "usdpkr_d3", "usdpkr_d6", "usdpkr_d12", "usdpkr_dev12"]
    return feats[cols]


# ----------------------------------------------------------------------
# 4. CATBOOST MODELS
# ----------------------------------------------------------------------
def make_model(seed=42, iterations=300):
    return CatBoostRegressor(
        loss_function="RMSE", iterations=iterations, learning_rate=0.05,
        depth=3, l2_leaf_reg=10, random_strength=1.0, bagging_temperature=0.5,
        random_seed=seed, verbose=False, allow_writing_files=False)


def fit_ensemble(X, y, n_seeds, iterations=300):
    return [make_model(42 + k, iterations).fit(X, y) for k in range(n_seeds)]


def predict_ensemble(ens, X):
    return float(np.mean([m.predict(X)[0] for m in ens]))


def train_direct(df, h, n_seeds=N_SEEDS, last_target_row=None, iterations=300):
    """
    Train Stage-1 (one model per factor) and Stage-2 (USD/PKR) for a horizon of h months.
    last_target_row: only use origins t whose target month t+h is < this row number
                     (used by the 80/20 back-test so test months are never seen).
    """
    n = len(df)
    feats = make_features(df)
    stop = (n - h) if last_target_row is None else min(n - h, last_target_row - h)
    rows = list(range(WARMUP, stop))
    if len(rows) < 20:
        raise ValueError(f"Horizon of {h} months is too long for this much data "
                         f"(only {len(rows)} training rows). Ask for a nearer month.")
    idx = df.index[rows]
    X = feats.loc[idx]

    stage1 = {}
    for v in FACTORS:
        y = future_change(df, v, h).loc[idx]
        stage1[v] = fit_ensemble(X, y, n_seeds, iterations)

    fut = df[FACTORS].shift(-h).loc[idx]
    cur = df[FACTORS].loc[idx]
    X2 = pd.concat([stage2_features(cur, fut), stage2_state(feats).loc[idx]], axis=1)
    y2 = future_change(df, "usdpkr", h).loc[idx]
    stage2 = fit_ensemble(X2, y2, n_seeds, iterations)
    return {"stage1": stage1, "stage2": stage2, "h": h}


def predict_direct(models, df):
    """Forecast every variable h months after the LAST row of df."""
    feats = make_features(df)
    X = feats.iloc[[-1]]
    last = df.iloc[-1]
    out = {}
    for v in FACTORS:
        step = predict_ensemble(models["stage1"][v], X)
        out[v] = float(last[v] * np.exp(step) if v in LOG_VARS else last[v] + step)

    # keep factors inside their (slightly widened) historical range; rates cannot be < 0
    for v in FACTORS:
        lo, hi = df[v].min(), df[v].max()
        pad = 0.05 * (hi - lo)
        out[v] = float(np.clip(out[v], lo - pad, hi + pad))
    out["sbp"] = max(out["sbp"], 0.0)
    out["fed"] = max(out["fed"], 0.0)
    out["infl_diff"] = out["pk_cpi"] - out["us_cpi"]

    # Stage 2: predicted factors -> USD/PKR
    fut = pd.DataFrame([{v: out[v] for v in FACTORS}], index=X.index)
    cur = pd.DataFrame([{v: last[v] for v in FACTORS}], index=X.index)
    X2 = pd.concat([stage2_features(cur, fut), stage2_state(feats).iloc[[-1]]], axis=1)
    step = predict_ensemble(models["stage2"], X2)
    out["usdpkr"] = float(last["usdpkr"] * np.exp(step))
    return out


# ----------------------------------------------------------------------
# 5. BACK-TEST ON THE 20 % TEST PART
# ----------------------------------------------------------------------
def backtest(df, horizons=(1, 3, 6, 12), plot_dir=None):
    n = len(df)
    split = int(TRAIN_RATIO * n)
    print(f"\nTRAIN / TEST SPLIT:  train = {df.index[0]:%b-%Y} to {df.index[split - 1]:%b-%Y} "
          f"({split} months)   test = {df.index[split]:%b-%Y} to {df.index[-1]:%b-%Y} "
          f"({n - split} months)")
    print("Models are trained ONLY on the train part and judged on the unseen test part.\n")

    mape = lambda a, p: np.mean(np.abs((a - p) / a)) * 100
    print("USD/PKR  (full pipeline: predicted factors -> USD/PKR)")
    print(f"  {'Horizon':<10}{'CatBoost MAPE':>15}{'Naive MAPE':>13}{'RMSE (Rs)':>12}"
          f"{'Direction hit':>15}")
    factor_rows = []
    plot_data = None
    for h in horizons:
        try:
            models = train_direct(df, h, n_seeds=1, last_target_row=split, iterations=200)
        except ValueError:
            continue
        act, pred, naive = [], [], []
        fa = {v: [] for v in FACTORS}
        fp = {v: [] for v in FACTORS}
        fn = {v: [] for v in FACTORS}
        dates = []
        for t in range(split - h, n - h):                    # origins whose target is in the test part
            hist = df.iloc[:t + 1]
            out = predict_direct(models, hist)
            act.append(df["usdpkr"].iloc[t + h])
            pred.append(out["usdpkr"])
            naive.append(hist["usdpkr"].iloc[-1])
            dates.append(df.index[t + h])
            for v in FACTORS:
                fa[v].append(df[v].iloc[t + h]); fp[v].append(out[v]); fn[v].append(hist[v].iloc[-1])
        act, pred, naive = map(np.array, (act, pred, naive))
        hit = np.mean(np.sign(pred - naive) == np.sign(act - naive)) * 100
        rmse = np.sqrt(np.mean((act - pred) ** 2))
        print(f"  {str(h) + ' month':<10}{mape(act, pred):>14.2f}%{mape(act, naive):>12.2f}%"
              f"{rmse:>12.2f}{hit:>14.0f}%")
        for v in FACTORS:
            a, p, nv = map(np.array, (fa[v], fp[v], fn[v]))
            factor_rows.append((h, v, np.mean(np.abs(a - p)), np.mean(np.abs(a - nv))))
        if plot_data is None:
            plot_data = (h, dates, act, pred)

    if factor_rows:
        print("\nFACTORS  (Mean Absolute Error on the test part; lower is better)")
        print(f"  {'Factor':<34}{'Horizon':>9}{'CatBoost MAE':>14}{'Naive MAE':>12}")
        for h, v, m, nv in factor_rows:
            print(f"  {NICE[v]:<34}{str(h) + ' m':>9}{m:>14.3f}{nv:>12.3f}")

    if plot_data is not None and plot_dir:
        h, dates, act, pred = plot_data
        plt.figure(figsize=(12, 6))
        plt.plot(dates, act, label="Actual USD/PKR (test part)", color="royalblue", lw=2)
        plt.plot(dates, pred, label=f"CatBoost predicted ({h}-month ahead)",
                 color="darkorange", linestyle="--", lw=2)
        plt.title("USD/PKR back-test on the unseen 20% test data")
        plt.xlabel("Date"); plt.ylabel("USD/PKR"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(plot_dir, "backtest_usdpkr.png")
        plt.savefig(p, dpi=130)
        print(f"\nChart saved: {p}")
        plt.close()


# ----------------------------------------------------------------------
# 6. USER QUESTIONS
# ----------------------------------------------------------------------
def parse_target(text, last_date):
    """Month-year (Dec-2026, 2027-03, 06/2026) OR a number = months ahead (e.g. 18)."""
    text = text.strip()
    if text.isdigit():
        n = int(text)
        return (last_date + pd.DateOffset(months=n)).to_period("M").to_timestamp() if n > 0 else None
    m = re.fullmatch(r"(\d{1,2})[/\-.](\d{4})", text)
    if m and 1 <= int(m.group(1)) <= 12:
        return pd.Timestamp(int(m.group(2)), int(m.group(1)), 1)
    try:
        return pd.to_datetime(text).to_period("M").to_timestamp()
    except Exception:
        return None


def show_and_save(df, row, target, h, out_dir):
    last_date = df.index[-1]
    print(f"\n================ FORECAST FOR {target:%B %Y}  ({h} months ahead) ================")
    print(f"  {'':<36}{'Last (' + format(last_date, '%b-%Y') + ')':>14}{'Forecast':>14}")
    for v in VARS:
        print(f"  {NICE[v]:<36}{df[v].iloc[-1]:>14,.3f}{row[v]:>14,.3f}")
    print("=" * 66, flush=True)

    try:
        xl = os.path.join(out_dir, f"forecast_{target:%b%Y}.xlsx")
        pd.DataFrame({"Factor": [NICE[v] for v in VARS],
                      f"Last ({last_date:%b-%Y})": [df[v].iloc[-1] for v in VARS],
                      f"Forecast {target:%b-%Y}": [row[v] for v in VARS]}
                     ).round(4).to_excel(xl, index=False)
        print(f"Saved: {xl}")
    except Exception as e:
        print(f"(Could not save Excel: {e})")

    try:
        png = os.path.join(out_dir, f"forecast_{target:%b%Y}.png")
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["usdpkr"], label="Historical USD/PKR", color="royalblue", lw=2)
        plt.plot([last_date, target], [df["usdpkr"].iloc[-1], row["usdpkr"]],
                 color="darkorange", linestyle="--", lw=2, label="CatBoost forecast")
        plt.scatter([target], [row["usdpkr"]], color="red", zorder=5)
        plt.annotate(f"{row['usdpkr']:.2f}", (target, row["usdpkr"]),
                     textcoords="offset points", xytext=(-10, 10))
        plt.title(f"USD/PKR forecast for {target:%b-%Y} (CatBoost)")
        plt.xlabel("Date"); plt.ylabel("USD/PKR"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(png, dpi=130)
        print(f"Chart saved: {png}\n")
        plt.close()
    except Exception as e:
        print(f"(Could not save chart: {e})\n")


def run_forecast(df, target, h, cache, out_dir):
    if h > 36:
        print(f"  Note: {h} months ahead - the further the date, the less reliable any forecast is.")
    if h not in cache:
        print(f"  Training CatBoost for {h}-month-ahead horizon ...", flush=True)
        cache[h] = train_direct(df, h)
    row = predict_direct(cache[h], df)
    show_and_save(df, row, target, h, out_dir)


def main():
    path = pick_file()
    print(f"\nUsing file: {path}")
    df = load_data(path)
    out_dir = os.path.dirname(os.path.abspath(path))
    last_date = df.index[-1]
    print(f"Training data: {len(df)} months ({df.index[0]:%b-%Y} to {last_date:%b-%Y})")

    print("Running back-test (about a minute) ...")
    try:
        backtest(df, plot_dir=out_dir)
    except Exception as e:
        print(f"(back-test skipped: {e})")
    print("\nReady.\n", flush=True)

    cache = {}
    cli_targets = sys.argv[2:]
    if cli_targets:                                   # non-interactive mode
        for txt in cli_targets:
            target = parse_target(txt, last_date)
            if target is None or target <= last_date:
                print(f"  '{txt}' is not a valid month after {last_date:%b-%Y}.")
                continue
            h = (target.year - last_date.year) * 12 + (target.month - last_date.month)
            try:
                run_forecast(df, target, h, cache, out_dir)
            except Exception as e:
                print(f"  Forecast error: {e}\n")
        return

    while True:
        txt = input(f"Enter FUTURE month-year after {last_date:%b-%Y} (e.g. Dec-2027 or 06/2026)\n"
                    f"or just the NUMBER of months ahead (e.g. 18)   [q = quit]: ")
        if txt.strip().lower() in ("q", "quit", "exit"):
            break
        target = parse_target(txt, last_date)
        if target is None or target <= last_date:
            print(f"  Please give a valid month AFTER {last_date:%b-%Y}.\n")
            continue
        h = (target.year - last_date.year) * 12 + (target.month - last_date.month)
        try:
            run_forecast(df, target, h, cache, out_dir)
        except Exception as e:
            print(f"  Forecast error: {e}\n")


if __name__ == "__main__":
    main()

"""
baseline_test.py -- how well can a simple baseline predict each site's hourly
load? (script 7). An LFM activation is settled against a baseline, so the
baseline's own error is noise on every delivered kW.

Baseline: mean load in the same clock hour on the previous 10 comparable days
(weekday -> previous weekdays, weekend -> previous weekend days), looking back
at most `max_lookback_days`; at least `min_days` such days must have a value.
Load = gross where lpc_11 finds a generation file, else net exchange (labelled).
Outputs: baseline_error.csv (per site and season: distribution of the hourly
error = actual - baseline, share of hours with |error| > 100 kW),
baseline_error_hourly.csv, baseline_error_hist_<site>.png/.pdf.
Site codes only (lpc_11 CONFIG["site_code_key"]).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG = {
    "lpc_module": "lpc_11",
    "sites": "selected",            # "selected" | "all" | [site ids as in site_comparison.csv]
    "n_days": 10,
    "min_days": 5,
    "max_lookback_days": 30,
    "weekend_days": [5, 6],
    "error_limit_kw": 100.0,
    "output_dir": "baseline_test_out",
    "csv_sep": ";",
    "csv_decimal": ",",
}


def baseline(h, cfg):
    """Previous-N-comparable-days mean for every hour of an hourly series."""
    df = pd.DataFrame({"kW": h})
    df["date"] = df.index.normalize()
    df["hod"] = df.index.hour
    df["we"] = np.isin(df.index.dayofweek, cfg["weekend_days"])
    piv = df.pivot_table(index="date", columns="hod", values="kW", aggfunc="mean")
    piv = piv.reindex(pd.date_range(piv.index.min(), piv.index.max(), freq="D"))
    we = np.isin(piv.index.dayofweek, cfg["weekend_days"])
    out = pd.DataFrame(np.nan, index=piv.index, columns=piv.columns)
    for flag in (False, True):
        sub = piv[we == flag]
        dates = sub.index
        for i, d in enumerate(dates):
            prev = sub.iloc[max(0, i - 3 * cfg["n_days"]):i]
            prev = prev[prev.index >= d - pd.Timedelta(days=cfg["max_lookback_days"])].tail(cfg["n_days"])
            cnt = prev.notna().sum()
            out.loc[d] = prev.mean().where(cnt >= cfg["min_days"])
    b = out.stack()
    b.index = [d + pd.Timedelta(hours=int(hh)) for d, hh in b.index]
    return pd.Series(b.to_numpy(), index=pd.DatetimeIndex(b.index)).reindex(h.index)


def main(cfg=CONFIG):
    sys.path.insert(0, os.getcwd())
    L = __import__(cfg["lpc_module"])
    lcfg = L.CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)
    codes = L._code_map(lcfg)
    df = L.read_comparison(lcfg)
    if df is None:
        raise SystemExit("no site_comparison.csv")
    if cfg["sites"] == "selected":
        pick = df[df["selected"].astype(str).str.lower().isin(["true", "1", "1.0"])]
    elif cfg["sites"] == "all":
        pick = df
    else:
        want = {str(s).strip().lower() for s in cfg["sites"]}
        pick = df[df[L.SITE_KEY].astype(str).str.strip().str.lower().isin(want)]
    som = {m: k for k, ms in lcfg["seasons"].items() for m in ms}
    summ, hourly_rows = [], []
    for _, r in pick.iterrows():
        sid = str(r[L.SITE_KEY])
        code = codes.get(sid.strip().lower(), sid if not codes else f"UNCODED:{sid}")
        path = L.find_meter_file(r, lcfg)
        if path is None:
            summ.append({"site": code, "season": "year", "note": "meter file not found"})
            continue
        c = dict(lcfg, csv_path=path, show_plot=False)
        net, info = L.load_net_series(c)
        gross = L.gross_series(net, c)
        load, series = (gross, "gross") if gross is not None else (net, "net exchange")
        h = L.hourly_mean(load, c)
        h = h[~h.index.duplicated()]
        b = baseline(h, cfg)
        e = (h - b).dropna()
        seas = pd.Series([som.get(m) for m in e.index.month], index=e.index)
        hourly_rows.append(pd.DataFrame({"site": code, "hour": e.index.strftime("%Y-%m-%d %H:00"),
                                         "actual_kw": h.reindex(e.index).round(2).to_numpy(),
                                         "baseline_kw": b.reindex(e.index).round(2).to_numpy(),
                                         "error_kw": e.round(2).to_numpy(), "season": seas.to_numpy()}))
        for season in list(lcfg["seasons"]) + ["year"]:
            v = e if season == "year" else e[(seas == season).to_numpy()]
            if not len(v):
                continue
            summ.append({"site": code, "season": season, "series": series, "n_hours": len(v),
                         "mean_error_kw": v.mean(), "mae_kw": v.abs().mean(),
                         "p5_kw": v.quantile(0.05), "p50_kw": v.median(), "p95_kw": v.quantile(0.95),
                         "share_abs_error_gt_limit": (v.abs() > cfg["error_limit_kw"]).mean(),
                         "error_limit_kw": cfg["error_limit_kw"],
                         "mae_relative_to_mean_load": v.abs().mean() / h.mean() if h.mean() else np.nan,
                         "baseline": f"mean of same hour, previous {cfg['n_days']} comparable days "
                                     f"(>= {cfg['min_days']} within {cfg['max_lookback_days']} days)",
                         "source_resolution": "15 min", "block_length_h": 1, "percentile": "P5/P50/P95 of errors",
                         "f": "n/a"})
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.hist(e, bins=60, color="0.5", edgecolor="0.2")
        for x in (-cfg["error_limit_kw"], cfg["error_limit_kw"]):
            ax.axvline(x, color="k", ls="--", lw=0.9)
        ax.set_xlabel("hourly baseline error, actual - baseline (kW)")
        ax.set_ylabel("hours")
        ax.set_title(f"Site {code}: baseline error ({series}); dashed = +/- {cfg['error_limit_kw']:.0f} kW")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in code)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(cfg["output_dir"], f"baseline_error_hist_{safe}.{ext}"), dpi=200,
                        bbox_inches="tight")
        plt.close(fig)
    w = dict(sep=cfg["csv_sep"], decimal=cfg["csv_decimal"], index=False, encoding="utf-8-sig")
    s = pd.DataFrame(summ).round(4)
    s.to_csv(os.path.join(cfg["output_dir"], "baseline_error.csv"), **w)
    if hourly_rows:
        pd.concat(hourly_rows).to_csv(os.path.join(cfg["output_dir"], "baseline_error_hourly.csv"), **w)
    print(s[s["season"] == "year"][["site", "n_hours", "mean_error_kw", "mae_kw", "share_abs_error_gt_limit"]]
          .to_string(index=False))
    print(f"[baseline] outputs in {os.path.abspath(cfg['output_dir'])}")


if __name__ == "__main__":
    main()

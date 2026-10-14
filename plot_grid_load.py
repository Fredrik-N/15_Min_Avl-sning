"""
plot_grid_load.py -- plots of the grid consumption file (natbelastning).

Reads the file exactly as lpc_10 does (CONFIG["grid_consumption"] in lpc_10.py),
so the constrained window drawn here is the one the other scripts use.
Run:  python plot_grid_load.py
Needs lpc_10.py in the same folder. Nothing is recomputed in lpc_10; this only plots.
Outputs (PNG + PDF, one figure per file) in grid_load_out/.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG = {
    "lpc_module": "lpc_10",
    "output_dir": "grid_load_out",
    "winter_months": [11, 12, 1, 2],
    "summer_months": [6, 7, 8],
    "unit_label": "kW (average per 15 min)",   # the file's own unit; only the shape matters for lpc
    "dpi": 200,
}


def save(fig, out, stem, dpi):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{stem}.{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def shade_window(ax, hours):
    for h in hours:
        ax.axvspan(h, h + 1, color="0.85", zorder=0)


def main(cfg=CONFIG):
    sys.path.insert(0, os.getcwd())
    lpc = __import__(cfg["lpc_module"])
    lcfg = lpc.CONFIG
    s = lpc.load_grid_consumption(lcfg).dropna()
    hours, hourly = lpc.derive_constraint_window_hours(s, lcfg)
    pct = lcfg["grid_consumption"]["constraint_percentile"]
    thr = hourly.quantile(pct)
    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)
    u, dpi = cfg["unit_label"], cfg["dpi"]
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

    # 1. time series: daily mean with 7-day rolling mean
    fig, ax = plt.subplots(figsize=(9, 3.6))
    d = s.resample("D").mean()
    ax.plot(d.index, d.values, color="0.7", lw=0.6, label="daily mean")
    ax.plot(d.index, d.rolling(7, center=True, min_periods=4).mean(), color="k", lw=1.2, label="7-day mean")
    ax.set_ylabel(u)
    ax.set_title("Grid consumption over time")
    ax.legend(frameon=False)
    save(fig, out, "grid_load_timeseries", dpi)

    # 2. hour-of-day profile: all / winter / summer, with the constraint window
    fig, ax = plt.subplots(figsize=(7, 4))
    shade_window(ax, hours)
    for name, months, ls in (("all year", None, "-"), ("winter", cfg["winter_months"], "--"),
                             ("summer", cfg["summer_months"], ":")):
        x = s if months is None else s[s.index.month.isin(months)]
        p = x.groupby(x.index.hour).mean()
        ax.plot(p.index + 0.5, p.values, color="k", ls=ls, lw=1.6 if months is None else 1.2, label=name)
    ax.axhline(thr, color="0.4", lw=0.9)
    ax.text(0.2, thr, f" {pct:.0%} percentile of the all-year hourly profile", va="bottom", fontsize=8, color="0.3")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("hour of day"); ax.set_ylabel(u)
    ax.set_title(f"Daily profile; shaded = constrained hours {hours[0]}-{hours[-1] + 1}" if hours else "Daily profile")
    ax.legend(frameon=False)
    save(fig, out, "grid_load_daily_profile", dpi)

    # 3. month x hour heatmap
    fig, ax = plt.subplots(figsize=(7.5, 4))
    m = s.groupby([s.index.month, s.index.hour]).mean().unstack()
    im = ax.imshow(m.values, aspect="auto", cmap="Greys", origin="lower", extent=[0, 24, 0.5, 12.5])
    ax.set_yticks(range(1, 13)); ax.set_xticks(range(0, 25, 3))
    ax.grid(False)
    ax.set_xlabel("hour of day"); ax.set_ylabel("month")
    ax.set_title("Mean grid consumption by month and hour")
    fig.colorbar(im, ax=ax, label=u)
    save(fig, out, "grid_load_month_hour", dpi)

    # 4. load-duration curve
    fig, ax = plt.subplots(figsize=(6.5, 4))
    v = np.sort(s.values)[::-1]
    ax.plot(np.linspace(0, 100, len(v)), v, color="k", lw=1.3)
    ax.set_xlabel("share of time (%)"); ax.set_ylabel(u)
    ax.set_title("Load-duration curve")
    save(fig, out, "grid_load_duration_curve", dpi)

    pd.DataFrame({"hour": hourly.index, "mean_load": hourly.values,
                  "constrained": [h in hours for h in hourly.index]}).to_csv(
        os.path.join(out, "grid_load_hourly_profile.csv"), index=False, sep=";", decimal=",")
    print(f"[grid] {len(s)} intervals {s.index.min():%Y-%m-%d} to {s.index.max():%Y-%m-%d}; "
          f"constrained hours {hours}; outputs in {os.path.abspath(out)}")


if __name__ == "__main__":
    main()

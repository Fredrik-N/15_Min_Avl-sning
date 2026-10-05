"""
Compare the combined load of every customer in the site comparison file
with Öresundskraft's grid load.

Uses lpc_8.py (same folder) for everything it reads, so the settings are
the same as for the per-site analysis:
  - the customers are the rows of CONFIG["results_csv_path"]
    (site_comparison.csv); each one's meter file is found the same way as
    `lpc_8.py --rerun-all` finds it (stored path, else CONFIG["meter_folder"]);
  - the grid load is CONFIG["grid_consumption"] (hourly Netto_kWh held
    constant over each quarter-hour), and the constraint window is
    derived from it exactly as in lpc_8.

Questions answered:
  - How big are these customers together compared with the grid
    (energy share, share at the grid's peak)?
  - Do they peak when the grid peaks (coincidence, correlation,
    hour-of-day profiles, load in the constraint window)?
  - Which customer contributes most when the grid is constrained?

Everything is done for the whole period and again for each season
(winter / summer, CONFIG_CMP["seasons"]); by default each season gets its
own constraint window from the grid's load in that season.

Outputs (next to the comparison file, prefix CONFIG_CMP["output_prefix"]):
  <prefix>_summary.csv      one row per customer + a TOTAL row, per season
                            (column "season": all / winter / summer)
  <prefix>_timeseries.csv   15-min: grid, customers total, each customer
  <prefix>.png              four panels for the whole period (see plot())
  <prefix>_winter.png, <prefix>_summer.png   the same per season

Run:  python compare_customers_grid.py
"""
import copy
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lpc_8 as L

CONFIG_CMP = {
    # "common": only quarter-hours where EVERY customer has data -- the
    #           total is then a true total. Use when the files cover the
    #           same period.
    # "all":    every quarter-hour where at least one customer has data;
    #           the total then sums whoever reported (n_customers_reporting
    #           in the time series says how many).
    "period": "common",
    "grid_peak_top_share": 0.01,     # "grid peak" = the grid's top 1 % quarter-hours
    "output_prefix": "customers_vs_grid",
    "write_timeseries": True,
    "show_plot": True,
    # The same comparison for each season (on top of the whole period).
    # None = the seasons lpc_8 uses for its temperature correlation
    # (CONFIG["temp_corr_season_months"]: winter 11,12,1,2; summer 5,6,7,8).
    "seasons": None,
    # "own":  each season gets its own constraint window, derived from the
    #         grid's load in that season only (the grid's peak hours move
    #         between winter and summer).
    # "year": every season is scored against the whole-period window.
    "season_constraint_window": "own",
}


def load_customers(cfg):
    """{site_id: 15-min kW series} for every site in the comparison file
    whose meter file can be found; prints the ones that cannot."""
    df = L.read_comparison(cfg)
    if df is None:
        raise FileNotFoundError(f"No comparison file at {cfg.get('results_csv_path')!r}. "
                                f"Run lpc_8.py on your sites first.")
    series, missing = {}, []
    for _, row in df.iterrows():
        sid = str(row[L.SITE_KEY])
        path = L.find_meter_file(row, cfg)
        if path is None:
            missing.append(sid)
            continue
        c = copy.deepcopy(cfg)
        c["csv_path"] = path
        print(f"[customers] loading {sid}")
        series[sid] = L.load_series(c)
    if missing:
        print(f"[customers] meter file not found, left out: {missing} "
              f"(set CONFIG['meter_folder'] in lpc_8.py)")
    if not series:
        raise ValueError("No customer meter files could be loaded.")
    return series


def combine(series, grid, cmp):
    """One frame on the 15-min grid: each customer, their total, the grid."""
    sites = pd.concat(series, axis=1)
    n_rep = sites.notna().sum(axis=1)
    if cmp["period"] == "common":
        keep = n_rep == sites.shape[1]
    else:
        keep = n_rep > 0
    sites = sites[keep]
    frame = sites.copy()
    frame["customers_total_kw"] = sites.sum(axis=1, min_count=1)
    frame["n_customers_reporting"] = n_rep[keep]
    frame["grid_kw"] = grid.reindex(frame.index)
    frame = frame[frame["grid_kw"].notna()]
    if frame.empty:
        raise ValueError(
            "Customers and grid share no quarter-hours. Check that the grid file covers "
            "the same dates" + (", or use period='all'" if cmp["period"] == "common" else "")
            + " (and that both use the same time zone convention).")
    return frame


def _corr(a, b):
    ok = a.notna() & b.notna()
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 and a[ok].std() > 0 and b[ok].std() > 0 else np.nan


def summarise(frame, sites, constraint_hours, cmp, cfg):
    """Per-customer rows plus a TOTAL row."""
    h = pd.Timedelta(cfg["resample_freq"]).total_seconds() / 3600.0
    grid = frame["grid_kw"]
    in_win = frame.index.hour.isin(constraint_hours)
    top = grid >= grid.quantile(1 - cmp["grid_peak_top_share"])
    t_peak = grid.idxmax()
    grid_e = grid.sum() * h
    total = frame["customers_total_kw"]
    rows = []
    for name in sites + ["customers_total_kw"]:
        x = frame[name]
        e = x.sum() * h
        hourly = pd.DataFrame({"x": x, "g": grid}).resample("h").mean()
        rows.append({
            L.SITE_KEY: "TOTAL (all customers)" if name == "customers_total_kw" else name,
            "period_start": str(frame.index.min()), "period_end": str(frame.index.max()),
            "quarter_hours": int(x.notna().sum()),
            "energy_kwh": round(e, 0),
            "share_of_customers_energy": round(e / (total.sum() * h), 4),
            "share_of_grid_energy": round(e / grid_e, 5),
            "avg_kw": round(x.mean(), 2),
            "peak_kw": round(x.max(), 2),
            "kw_at_grid_peak": round(float(x.loc[t_peak]), 2) if pd.notna(x.loc[t_peak]) else np.nan,
            "share_of_grid_at_grid_peak": round(float(x.loc[t_peak] / grid.loc[t_peak]), 5),
            # how much of its own peak the customer draws when the grid is at its top 1 %
            "avg_kw_in_grid_top_hours": round(x[top].mean(), 2),
            "coincidence_with_grid_peak": round(x[top].mean() / x.max(), 3) if x.max() > 0 else np.nan,
            "avg_kw_in_constraint_window": round(x[in_win].mean(), 2),
            "avg_kw_outside_constraint_window": round(x[~in_win].mean(), 2),
            "window_avg_over_overall_avg": round(x[in_win].mean() / x.mean(), 3) if x.mean() > 0 else np.nan,
            "share_of_customers_in_constraint_window": round(x[in_win].mean() / total[in_win].mean(), 4),
            "share_of_grid_in_constraint_window": round(x[in_win].mean() / grid[in_win].mean(), 5),
            "corr_with_grid_15min": round(_corr(x, grid), 3),
            "corr_with_grid_hourly": round(_corr(hourly["x"], hourly["g"]), 3),
        })
    out = pd.DataFrame(rows)
    order = out.iloc[:-1].sort_values("avg_kw_in_constraint_window", ascending=False)
    return pd.concat([out.iloc[[-1]], order], ignore_index=True)


def _shade(ax, hours, alpha):
    """One band per run of consecutive constraint hours."""
    hs = sorted(hours)
    start = prev = None
    for h in hs + [None]:
        if start is None:
            start = prev = h
        elif h is not None and h == prev + 1:
            prev = h
        else:
            ax.axvspan(start - 0.5, prev + 0.5, color="tab:red", alpha=alpha, linewidth=0)
            start = prev = h


def plot(frame, sites, constraint_hours, path, show, title=""):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    if title:
        fig.suptitle(title, fontsize=14)
    grid, total = frame["grid_kw"], frame["customers_total_kw"]

    # 1. daily mean, two axes
    ax = axes[0, 0]
    # days with data only, side by side: a season that spans the new year
    # (Nov-Feb) is drawn without an empty spring/summer in the middle
    d = frame[["grid_kw", "customers_total_kw"]].resample("D").mean().dropna(how="all")
    x = np.arange(len(d))
    ax.plot(x, d["grid_kw"].to_numpy(), color="black", label="Grid")
    ax.set_ylabel("Grid, daily mean (kW)")
    ax2 = ax.twinx()
    ax2.plot(x, d["customers_total_kw"].to_numpy(), color="tab:blue", label="Customers total")
    ax2.set_ylabel("Customers total, daily mean (kW)", color="tab:blue")
    ticks = np.unique(np.linspace(0, len(d) - 1, min(len(d), 7)).astype(int))
    ax.set_xticks(ticks)
    ax.set_xticklabels([d.index[i].strftime("%Y-%m-%d") for i in ticks], rotation=30)
    jumps = np.flatnonzero(np.diff(d.index.to_numpy()).astype("timedelta64[D]").astype(int) > 1)
    for j in jumps:                      # mark where days are skipped
        ax.axvline(j + 0.5, color="grey", linestyle=":", linewidth=1)
    ax.set_title("Daily mean load (dotted line = days skipped)" if len(jumps) else "Daily mean load")
    ax.grid(True, alpha=0.3)

    # 2. hour-of-day shape, each relative to its own mean, weekdays
    ax = axes[0, 1]
    wd = frame[frame.index.dayofweek < 5]
    for col, lab, colr in (("grid_kw", "Grid", "black"), ("customers_total_kw", "Customers total", "tab:blue")):
        prof = wd[col].groupby(wd.index.hour).mean()
        ax.plot(prof.index, prof / wd[col].mean(), color=colr, linewidth=2, label=lab)
    _shade(ax, constraint_hours, 0.08)
    ax.axhline(1, color="grey", linewidth=0.8)
    ax.set_title("Weekday shape (relative to own mean); red = constraint window")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Load / mean")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. who makes up the total, by hour of day
    ax = axes[1, 0]
    byh = frame[sites].groupby(frame.index.hour).mean()
    ax.stackplot(byh.index, byh.T.values, labels=[s[:30] for s in sites], alpha=0.85)
    _shade(ax, constraint_hours, 0.06)
    ax.set_title("Customers' mean load by hour of day (stacked)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("kW")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # 4. hourly scatter
    ax = axes[1, 1]
    hourly = frame[["grid_kw", "customers_total_kw"]].resample("h").mean().dropna()
    inw = hourly.index.hour.isin(constraint_hours)
    ax.scatter(hourly["grid_kw"][~inw], hourly["customers_total_kw"][~inw], s=4, alpha=0.3,
               color="grey", label="other hours")
    ax.scatter(hourly["grid_kw"][inw], hourly["customers_total_kw"][inw], s=4, alpha=0.5,
               color="tab:red", label="constraint window")
    ax.set_title(f"Hourly: customers vs grid (r = {_corr(hourly['customers_total_kw'], hourly['grid_kw']):+.2f})")
    ax.set_xlabel("Grid (kW)")
    ax.set_ylabel("Customers total (kW)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


def main(cfg=L.CONFIG, cmp=CONFIG_CMP):
    gpath = cfg.get("grid_consumption", {}).get("csv_path")
    if not gpath or not os.path.exists(gpath):
        raise FileNotFoundError(f"Grid file {gpath!r} not found -- set CONFIG['grid_consumption']['csv_path'] in lpc_8.py.")
    series = load_customers(cfg)
    grid = L.load_grid_consumption(cfg)
    frame = combine(series, grid, cmp)
    sites = list(series)
    year_hours, _ = L.derive_constraint_window_hours(grid, cfg)

    folder = os.path.dirname(os.path.abspath(cfg["results_csv_path"]))
    base = os.path.join(folder, cmp["output_prefix"])
    sep, dec = cfg.get("results_csv_sep", ";"), cfg.get("results_csv_decimal", ",")

    seasons = cmp.get("seasons") or cfg.get("temp_corr_season_months") or {}
    periods = [("all", None)] + list(seasons.items())
    summaries, overview, files = [], [], []
    for name, months in periods:
        f = frame if months is None else frame[frame.index.month.isin(months)]
        if f.empty:
            print(f"\n[customers vs grid] {name}: no shared data in months {months} -- skipped")
            continue
        if months is not None and cmp.get("season_constraint_window", "own") == "own":
            print(f"\n[{name}]", end="")
            hours, _ = L.derive_constraint_window_hours(grid[grid.index.month.isin(months)], cfg)
        else:
            hours = year_hours
        sm = summarise(f, sites, hours, cmp, cfg)
        sm.insert(0, "season", name)
        sm.insert(1, "constraint_hours", ",".join(str(h) for h in hours))
        summaries.append(sm)
        png = f"{base}.png" if months is None else f"{base}_{name}.png"
        label = "whole period" if months is None else f"{name} (months {', '.join(str(m) for m in months)})"
        plot(f, sites, hours, png, cmp["show_plot"], title=f"Customers vs grid -- {label}")
        files.append(png)
        t = sm.iloc[0]
        overview.append({"season": name, "constraint hours": f"{min(hours)}-{max(hours)}" if hours else "-",
                         "share of grid energy": f"{t['share_of_grid_energy']:.2%}",
                         "share at grid peak": f"{t['share_of_grid_at_grid_peak']:.2%}",
                         "share in window": f"{t['share_of_grid_in_constraint_window']:.2%}",
                         "window avg / avg": f"{t['window_avg_over_overall_avg']:.2f}",
                         "coincidence": f"{t['coincidence_with_grid_peak']:.2f}",
                         "corr (hourly)": f"{t['corr_with_grid_hourly']:+.2f}"})

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(f"{base}_summary.csv", sep=sep, decimal=dec, index=False, encoding="utf-8-sig")
    if cmp["write_timeseries"]:
        frame.to_csv(f"{base}_timeseries.csv", sep=sep, decimal=dec, encoding="utf-8-sig",
                     index_label="timestamp", float_format="%.3f")

    t = summaries[0].iloc[0]
    print(f"\n[customers vs grid] {len(sites)} customer(s), {t['period_start']} .. {t['period_end']} "
          f"({cmp['period']} period, {t['quarter_hours']} quarter-hours)")
    print("\n  all customers together, per season:")
    print(pd.DataFrame(overview).to_string(index=False))
    cols = [L.SITE_KEY, "avg_kw_in_constraint_window", "share_of_customers_in_constraint_window",
            "window_avg_over_overall_avg", "coincidence_with_grid_peak", "corr_with_grid_hourly"]
    for sm in summaries:
        print(f"\n  per customer, {sm['season'].iloc[0]} (constraint hours {sm['constraint_hours'].iloc[0]}), "
              f"largest in the window first:")
        print(sm.iloc[1:][cols].to_string(index=False))
    print(f"\n  wrote {base}_summary.csv" + (f", {base}_timeseries.csv" if cmp["write_timeseries"] else "")
          + ", " + ", ".join(files))
    return summary, frame


if __name__ == "__main__":
    main()

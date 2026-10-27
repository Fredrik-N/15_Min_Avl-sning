"""
resolution_ladder.py -- how much does the reducible-power estimate change
when 1 s data are averaged to coarser resolutions? (script 5)

For every site with 1 s data:
  reference  synthetic sets: the true compressor power (ground-truth columns),
             hourly mean;  real sites: P_1s from unilm_fhmm (compressor_hourly.csv
             + compressor_list.csv, the same formula as reducible_power.py).
  ladder     the 1 s active power averaged to 1 s, 1, 5, 15 and 60 min (clock
             aligned). At each step the 15-min method of reducible_power.py is
             applied to that series: L_base = P5 of the samples in the season
             (exporting samples left out), estimate = f * max(0, hourly mean -
             L_base). The hourly mean itself does not depend on the resolution;
             the base does -- that is the resolution effect measured here.
Measures (per site-hour, then summarised):
  ladder_errors.csv     error = estimate - reference: mean, MAE, RMSE, relative
                        MAE, by resolution x f x site x month x season
  ladder_confusion.csv  LFM class (stand-alone firm >= 100 kW, pool candidate
                        >= 10 kW, else not relevant; firm = P10 over the hours)
                        estimate vs reference, plus Spearman rank agreement of
                        the sites' mean reducible power (needs >= 3 sites)
  ladder_check_vs_dso.csv  the 1 s data averaged to 15 min against the DSO's
                        15-min meter series for the same site and period
                        (mean difference, correlation) -- read with lpc_11
  ladder_error_vs_resolution.png/.pdf
Site codes only.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG = {
    "sites": [
        # one dict per 1 s site, e.g.
        # {"site": "S1", "path": "synthetic.csv", "time_col": "Time", "time_unit": "s",
        #  "p_col": "Aggregate_kW", "kind": "synthetic",
        #  "truth_cols": ["gt_Z1_power_kW", "gt_Z2_power_kW", "gt_Z3_power_kW"]},
        # {"site": "B", "path": "B_1s.csv", "time_col": "time", "time_unit": None, "p_col": "P_kW",
        #  "kind": "real", "unilm_dir": "results_B", "dso_meter": "735... B förbrukning.xlsx"},
    ],
    "time_zone": "Europe/Stockholm",   # 1 s timestamps: epoch seconds are UTC; text without zone = local
    "resolutions_s": [1, 60, 300, 900, 3600],
    "f_values": [0.3, 0.5, 0.7, 1.0],
    "base_percentile": 5,
    "seasons": {"winter": [11, 12, 1, 2], "summer": [6, 7, 8], "shoulder": [3, 4, 5, 9, 10]},
    "hour_min_observed": 0.9,
    "class_standalone_kw": 100.0,
    "class_pool_kw": 10.0,
    "firm_percentile": 10,
    "lpc_module": "lpc_11",            # only for the DSO check
    "output_dir": "resolution_ladder_out",
    "csv_sep": ";",
    "csv_decimal": ",",
}


def read_1s(s, cfg):
    path = s["path"]
    df = pd.read_pickle(path) if path.endswith(".pkl") else \
        pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path, sep=None, engine="python")
    t = df[s["time_col"]]
    if s.get("time_unit"):
        idx = pd.to_datetime(t, unit=s["time_unit"], utc=True).dt.tz_convert(cfg["time_zone"]).dt.tz_localize(None)
    else:
        idx = pd.to_datetime(t)
        if getattr(idx.dt, "tz", None) is not None:
            idx = idx.dt.tz_convert(cfg["time_zone"]).dt.tz_localize(None)
    p = pd.Series(pd.to_numeric(df[s["p_col"]], errors="coerce").to_numpy(), index=pd.DatetimeIndex(idx))
    truth = None
    if s.get("truth_cols"):
        truth = df[s["truth_cols"]].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        truth = pd.Series(truth.to_numpy(), index=p.index)
    p = p[~p.index.duplicated()].sort_index()
    return p, (truth[~truth.index.duplicated()].sort_index() if truth is not None else None)


def hourly(x, cfg, expected_per_hour):
    g = x.resample("1h")
    m, n = g.mean(), g.count()
    return m[n >= cfg["hour_min_observed"] * expected_per_hour]


def season_of(idx, cfg):
    som = {m: k for k, ms in cfg["seasons"].items() for m in ms}
    return pd.Series([som.get(m) for m in idx.month], index=idx)


def ladder_estimates(p, cfg):
    """{resolution_s: hourly estimate at f = 1} (scale by f later)."""
    out = {}
    for r in cfg["resolutions_s"]:
        x = p if r == 1 else p.resample(f"{r}s").mean()
        x = x.dropna()
        h = hourly(p, cfg, 3600)                         # hourly mean from the full data (resolution-invariant)
        seas_x = season_of(x.index, cfg)
        base = {}
        for name in cfg["seasons"]:
            v = x[(seas_x == name).to_numpy()]
            v = v[v >= 0]
            base[name] = float(v.quantile(cfg["base_percentile"] / 100)) if len(v) else np.nan
        b = season_of(h.index, cfg).map(base).astype(float)
        out[r] = ((h - b).clip(lower=0), b)
    return out


def reference_unilm(folder):
    hh = pd.read_csv(os.path.join(folder, "compressor_hourly.csv"))
    lst = pd.read_csv(os.path.join(folder, "compressor_list.csv")).set_index("appliance")
    hh = hh.join(lst[["controllable", "rated_kW", "units"]], on="compressor_id", rsuffix="_l")
    c = pd.to_numeric(hh["controllable_l"], errors="coerce").fillna(0)
    hh["p"] = c * pd.to_numeric(hh["rated_kW_l"], errors="coerce").fillna(0) * \
        pd.to_numeric(hh["units_l"], errors="coerce").fillna(1) * hh["running_share"]
    g = hh[c > 0].groupby("hour_start_local")
    p = g["p"].sum(min_count=1)[g["running_share"].apply(lambda s: s.notna().all())]
    return pd.Series(p.to_numpy(), index=pd.to_datetime(p.index.str.slice(0, 19)))


def lfm_class(firm, cfg):
    if pd.isna(firm):
        return "n/a"
    return "stand-alone" if firm >= cfg["class_standalone_kw"] else \
        "pool candidate" if firm >= cfg["class_pool_kw"] else "not relevant"


def dso_check(s, p, cfg):
    sys.path.insert(0, os.getcwd())
    L = __import__(cfg["lpc_module"])
    net, _ = L.load_net_series(dict(L.CONFIG, csv_path=s["dso_meter"], show_plot=False))
    a = p.resample("15min").mean()
    j = pd.concat([a.rename("from_1s"), net.rename("dso")], axis=1).dropna()
    if len(j) < 4:
        return {"site": s["site"], "n_quarter_hours": len(j), "note": "no overlap between 1 s data and DSO series"}
    return {"site": s["site"], "n_quarter_hours": len(j), "period_start": str(j.index.min()),
            "period_end": str(j.index.max()), "mean_1s_kw": round(j["from_1s"].mean(), 2),
            "mean_dso_kw": round(j["dso"].mean(), 2),
            "mean_difference_kw": round((j["from_1s"] - j["dso"]).mean(), 2),
            "mae_kw": round((j["from_1s"] - j["dso"]).abs().mean(), 2),
            "correlation": round(j["from_1s"].corr(j["dso"]), 4),
            "best_shift_quarters": int(max(range(-8, 9), key=lambda k: j["from_1s"].corr(j["dso"].shift(k)))),
            "note": "a best shift other than 0 points to a clock / time-zone offset"}


def main(cfg=CONFIG):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    if not cfg["sites"]:
        raise SystemExit("CONFIG['sites'] is empty")
    err_rows, per_site, dso = [], [], []
    for s in cfg["sites"]:
        p, truth = read_1s(s, cfg)
        if s.get("kind", "real") == "synthetic":
            if truth is None:
                raise SystemExit(f"{s['site']}: synthetic set needs truth_cols")
            ref, ref_kind = hourly(truth, cfg, 3600), "ground truth (compressor power)"
        else:
            ref, ref_kind = reference_unilm(s["unilm_dir"]), "P_1s from unilm_fhmm"
        est = ladder_estimates(p, cfg)
        for r, (e1, base) in est.items():
            for f in cfg["f_values"]:
                e = f * e1
                j = pd.concat([e.rename("est"), ref.rename("ref")], axis=1).dropna()
                if not len(j):
                    continue
                d = j["est"] - j["ref"]
                for grp_name, key in (("all", pd.Series("all", index=j.index)),
                                      ("month", pd.Series(j.index.strftime("%Y-%m"), index=j.index)),
                                      ("season", season_of(j.index, cfg))):
                    for g, idx in key.groupby(key).groups.items():
                        dd, jj = d.loc[idx], j.loc[idx]
                        err_rows.append({"site": s["site"], "reference": ref_kind, "resolution_s": r, "f": f,
                                         "group_by": grp_name, "group": g, "n_hours": len(dd),
                                         "mean_error_kw": dd.mean(), "mae_kw": dd.abs().mean(),
                                         "rmse_kw": float(np.sqrt((dd ** 2).mean())),
                                         "relative_mae": dd.abs().mean() / jj["ref"].mean() if jj["ref"].mean() else np.nan,
                                         "base_load_kw": float(base.loc[idx].mean())})
                pf = cfg["firm_percentile"] / 100
                per_site.append({"site": s["site"], "resolution_s": r, "f": f,
                                 "mean_est_kw": j["est"].mean(), "mean_ref_kw": j["ref"].mean(),
                                 "firm_est_kw": j["est"].quantile(pf), "firm_ref_kw": j["ref"].quantile(pf)})
        if s.get("dso_meter"):
            try:
                dso.append(dso_check(s, p, cfg))
            except Exception as ex:
                dso.append({"site": s["site"], "note": f"{type(ex).__name__}: {ex}"})
    err = pd.DataFrame(err_rows)
    meta = {"source_resolution": "1 s averaged to resolution_s", "season_def": str(cfg["seasons"]),
            "block_length_h": 1, "percentile": f"base P{cfg['base_percentile']}; firm P{cfg['firm_percentile']}"}
    for k, v in meta.items():
        err[k] = v
    w = dict(sep=cfg["csv_sep"], decimal=cfg["csv_decimal"], index=False, encoding="utf-8-sig")
    err.round(4).to_csv(os.path.join(cfg["output_dir"], "ladder_errors.csv"), **w)

    ps = pd.DataFrame(per_site)
    conf, rho = [], []
    for (r, f), g in ps.groupby(["resolution_s", "f"]):
        ce = g["firm_est_kw"].map(lambda v: lfm_class(v, cfg))
        cr = g["firm_ref_kw"].map(lambda v: lfm_class(v, cfg))
        for a, b in pd.crosstab(cr, ce).stack().items():
            conf.append({"resolution_s": r, "f": f, "class_reference": a[0], "class_estimate": a[1], "n_sites": b})
        sp = g["mean_est_kw"].rank().corr(g["mean_ref_kw"].rank()) if len(g) >= 3 else np.nan
        rho.append({"resolution_s": r, "f": f, "n_sites": len(g), "spearman_mean_kw": sp,
                    "note": "" if len(g) >= 3 else "fewer than 3 sites: no rank agreement"})
    c = pd.DataFrame(conf)
    for k, v in meta.items():
        c[k] = v
    c.to_csv(os.path.join(cfg["output_dir"], "ladder_confusion.csv"), **w)
    pd.DataFrame(rho).round(4).to_csv(os.path.join(cfg["output_dir"], "ladder_rank_agreement.csv"), **w)
    ps.round(3).to_csv(os.path.join(cfg["output_dir"], "ladder_site_levels.csv"), **w)
    pd.DataFrame(dso or [{"note": "no site has dso_meter set"}]).to_csv(
        os.path.join(cfg["output_dir"], "ladder_check_vs_dso.csv"), **w)

    a = err[err["group_by"] == "all"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    styles = ["-", "--", ":", "-."]
    for i, f in enumerate(cfg["f_values"]):
        m = a[a["f"] == f].groupby("resolution_s")["mae_kw"].mean()
        ax.plot(m.index / 60, m.values, ls=styles[i % 4], marker="o", color="k", label=f"f = {f}")
    ax.set_xscale("log")
    ax.set_xticks([1 / 60, 1, 5, 15, 60])
    ax.set_xticklabels(["1 s", "1 min", "5 min", "15 min", "60 min"])
    ax.set_xlabel("resolution")
    ax.set_ylabel("mean absolute error per site-hour (kW)")
    ax.set_title("Reducible power: error against the reference vs resolution")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(cfg["output_dir"], f"ladder_error_vs_resolution.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(a.groupby(["resolution_s", "f"])[["mean_error_kw", "mae_kw", "relative_mae"]].mean().round(2).to_string())
    print(f"[ladder] outputs in {os.path.abspath(cfg['output_dir'])}")
    return err


if __name__ == "__main__":
    main()

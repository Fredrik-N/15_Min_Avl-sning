"""
alt_filter.py -- alternative screening filter (exploratory; task_alternative_filter.md).

Tests a filter that is easier to defend than the current one and shows which
sites it selects. It changes nothing in the current filter or in the case
group; the result is a COMPARISON (Results 8.2.2, Discussion, Appendix B).

What differs from the current filter
  1. load factor -> base-load share (M3)
  2. winter temperature correlation -> full-year change-point model (M6);
     temperature is supporting evidence only, never a reason to exclude
  3. size is reported against the 100 kW LFM minimum bid, not filtered
  4. coverage stays as a data-quality check

Metrics per site (hourly load = clock-hour mean of the 15-min values; an hour
needs CONFIG["hour_min_quarters"] values; GROSS load = net + generation where
lpc_11 finds a generation file, otherwise net load):
  M1 coverage            valid 15-min intervals / intervals in the year
  M2 P_base              5th percentile of hourly load, full year
  M3 base-load share     P_base / mean hourly load
  M4 weekend ratio       mean weekend load / mean weekday load
  M5 median above base   median of max(0, hourly load - P_base), full year and
                         per season (winter Nov-Feb, shoulder Mar-Apr + Sep-Oct,
                         summer May-Aug; own definition in CONFIG["seasons"],
                         NOT lpc_11's); every season uses the full-year P_base
  M6 change-point model  daily mean load against daily mean outdoor temperature,
                         CalTRACK-style: intercept only; heating only
                         a + b_h*max(0, T_bh - T); cooling only a + b_c*max(0, T - T_bc);
                         heating and cooling. Balance points on a 1 degC grid
                         (T_bh 0-15, T_bc 5-20, T_bh <= T_bc), OLS. A model is
                         admissible only if every slope in it is positive and
                         has p < 0.1; among admissible models the highest
                         adjusted R2 is selected (intercept only is always
                         admissible, adjusted R2 = 0). Class: cooling (cooling
                         slope kept), heating (only heating kept), both, flat.
M2-M5 are also computed on winter only (Nov-Feb; own winter P_base) as a
sensitivity check (columns sens_winter_*).

Filter: pass = F1 coverage >= 0.8, F2 base-load share >= x, F3 weekend ratio >= y.
x in {0.4, 0.5, 0.6} (default 0.5), y in {0.7, 0.8, 0.9} (default 0.8): 9
combinations. NaN metrics fail. Temperature and size are REPORTED, not filtered:
  temp_support = yes if the class is cooling or both, else neutral
  size_vs_lfm  = median above base (year) / 100 kW
Sites with PV and no generation data (pv_no_production = 1): the net load has
summer export, so the change-point model is fitted on Oct-Mar as well and the
summer cooling slope is flagged unreliable.
Do not tune thresholds after seeing the result; defaults and sweep are as given.

A site whose P_base is <= 0 kW is listed in alt_filter_zero_base_check.txt with
the counts that tell zeros in the raw data from PV export. Its metrics are
computed as they are (and flagged), not hidden.

Outputs (alt_filter_out/): alt_filter_metrics.csv, alt_filter_membership.csv,
alt_filter_membership_heatmap.png/.pdf, filter_comparison.csv/.tex,
alt_filter_header.txt (data source, period, thresholds, labels used, join/leave
and group sums), alt_filter_changepoint.pdf (one page per site),
alt_filter_zero_base_check.txt (when needed), label_crosstab.csv/.tex (when
site_labels.csv exists). Site codes only: from CONFIG["site_code_key"] of
lpc_11 or, without it, ids that already are codes ("Site H" -> H); otherwise the
script stops. Deterministic.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

CONFIG = {
    "lpc_module": "lpc_11",
    "sites": "all",                       # "all" | [site ids as in site_comparison.csv]
    "year": 2025,
    "seasons": {"winter": [11, 12, 1, 2], "shoulder": [3, 4, 9, 10], "summer": [5, 6, 7, 8]},
    "winter_months": [11, 12, 1, 2],
    "octmar_months": [10, 11, 12, 1, 2, 3],
    "weekend_days": [5, 6],
    "base_percentile": 5,
    "min_hours_per_day": 20,              # a day needs this many valid hours (load and temperature)
    "lfm_kw": 100.0,
    # --- change-point grid (task brief)
    "heating_balance_c": list(range(0, 16)),
    "cooling_balance_c": list(range(5, 21)),
    "slope_p_max": 0.1,
    "min_active_days": 3,                 # technical guard: a slope needs >= 3 days on its active side
    # --- filter
    "coverage_min": 0.8,
    "base_share_values": [0.4, 0.5, 0.6],
    "base_share_default": 0.5,
    "weekend_values": [0.7, 0.8, 0.9],
    "weekend_default": 0.8,
    # --- the current filter, for the comparison (written by lpc_11 / filter_sensitivity)
    "current_pass_csv": os.path.join("screening_out", "screening_pass.csv"),
    "current_freq_csv": os.path.join("filter_sensitivity_out", "selection_frequency.csv"),
    # --- optional labels: site,label with cold / not_cold / unknown (site code or site id)
    "labels_csv": "site_labels.csv",
    "output_dir": "alt_filter_out",
    "allow_real_names": False,
    "pass_color": "#3b7d4f",
    "fail_color": "0.88",
    "csv_sep": ";",
    "csv_decimal": ",",
}

CODED = re.compile(r"^\s*(?:Site\s+)?([A-Z]{1,2})(?:\s|$)")


# ------------------------------------------------------------------ codes
def make_coder(L, lcfg, cfg):
    key = L._code_map(lcfg)

    def code(x):
        s = str(x).strip()
        if s.lower() in key:
            return key[s.lower()]
        s = re.sub(r"^UNCODED:", "", s)
        if s.lower() in key:
            return key[s.lower()]
        m = CODED.match(s)
        if m:
            return m.group(1)
        if cfg["allow_real_names"]:
            return s
        raise SystemExit(f"[alt] '{s[:3]}...' has no anonymised code. Put every site in the private key file "
                         f"(lpc_11 CONFIG['site_code_key'], columns real_site_id;anonymised_id) or name the "
                         f"files 'Site A ...'. Real names are not written to thesis files.")
    return code


# ------------------------------------------------------------------ change-point model
def _ols(X, y):
    n, k = X.shape
    dof = n - k
    if dof <= 0:
        return None
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    rss = float(res @ res)
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 0:
        return None
    s2 = rss / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    p = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1 - rss / sst
    return {"beta": beta, "p": p, "adj_r2": 1 - (1 - r2) * (n - 1) / dof, "n": n}


def fit_changepoint(T, y, cfg):
    """T, y: daily temperature and load (arrays). Returns a dict (model, class, parameters)."""
    T, y = np.asarray(T, float), np.asarray(y, float)
    ok = np.isfinite(T) & np.isfinite(y)
    T, y = T[ok], y[ok]
    out = {"model": "flat", "class": "flat", "T_bh": np.nan, "T_bc": np.nan, "a": float(y.mean()) if len(y) else np.nan,
           "b_h": np.nan, "p_h": np.nan, "b_c": np.nan, "p_c": np.nan, "adj_r2": 0.0, "n_days": int(len(y))}
    if len(y) < 30:
        out["note"] = "fewer than 30 days"
        return out
    one = np.ones_like(T)
    pmax, mn = cfg["slope_p_max"], cfg["min_active_days"]
    best = None                                   # (adj_r2, order, dict); order prefers the simpler model on ties

    def consider(fit, model, cls, order, th, tc):
        nonlocal best
        if fit is None:
            return
        slopes = fit["beta"][1:]
        if len(slopes) and not (np.all(np.isfinite(fit["p"][1:])) and np.all(slopes > 0) and np.all(fit["p"][1:] < pmax)):
            return
        cand = {"model": model, "class": cls, "T_bh": th, "T_bc": tc, "a": float(fit["beta"][0]),
                "b_h": np.nan, "p_h": np.nan, "b_c": np.nan, "p_c": np.nan, "adj_r2": float(fit["adj_r2"]),
                "n_days": int(fit["n"])}
        i = 1
        if model in ("heating", "heating_cooling"):
            cand["b_h"], cand["p_h"] = float(fit["beta"][i]), float(fit["p"][i]); i += 1
        if model in ("cooling", "heating_cooling"):
            cand["b_c"], cand["p_c"] = float(fit["beta"][i]), float(fit["p"][i])
        key = (round(cand["adj_r2"], 12), -order)
        if best is None or key > best[0]:
            best = (key, cand)

    for th in cfg["heating_balance_c"]:
        h = np.maximum(0.0, th - T)
        if (h > 0).sum() >= mn:
            consider(_ols(np.column_stack([one, h]), y), "heating", "heating", 1, float(th), np.nan)
    for tc in cfg["cooling_balance_c"]:
        c = np.maximum(0.0, T - tc)
        if (c > 0).sum() >= mn:
            consider(_ols(np.column_stack([one, c]), y), "cooling", "cooling", 1, np.nan, float(tc))
    for th in cfg["heating_balance_c"]:
        h = np.maximum(0.0, th - T)
        if (h > 0).sum() < mn:
            continue
        for tc in cfg["cooling_balance_c"]:
            if th > tc:
                continue
            c = np.maximum(0.0, T - tc)
            if (c > 0).sum() >= mn:
                consider(_ols(np.column_stack([one, h, c]), y), "heating_cooling", "both", 2, float(th), float(tc))
    if best is not None and best[1]["adj_r2"] > 0:
        out.update(best[1])
    out["note"] = ""
    return out


def cp_predict(fit, T):
    T = np.asarray(T, float)
    y = np.full_like(T, fit["a"])
    if np.isfinite(fit["b_h"]):
        y = y + fit["b_h"] * np.maximum(0.0, fit["T_bh"] - T)
    if np.isfinite(fit["b_c"]):
        y = y + fit["b_c"] * np.maximum(0.0, T - fit["T_bc"])
    return y


# ------------------------------------------------------------------ per site
def daily_pairs(h, temp, cfg, months=None):
    th = temp.resample("1h").mean()
    d = pd.DataFrame({"load": h, "t": th.reindex(h.index)})
    if months is not None:
        d = d[d.index.month.isin(months)]
    g = d.groupby(d.index.normalize())
    n_ok = g.apply(lambda x: int(x.dropna().shape[0]))
    daily = pd.DataFrame({"load": g["load"].mean(), "t": g["t"].mean()})
    daily = daily[n_ok >= cfg["min_hours_per_day"]].dropna()
    return daily


def metrics_for_site(net, load, source, h, temp, cfg, interval_h):
    r = {}
    y0, y1 = pd.Timestamp(cfg["year"], 1, 1), pd.Timestamp(cfg["year"] + 1, 1, 1)
    n_exp = int(round((y1 - y0).total_seconds() / 3600 / interval_h))
    nv = net[(net.index >= y0) & (net.index < y1)].notna().sum()
    r["M1_coverage"] = nv / n_exp
    if h.empty:
        return r, None, None
    seas = {m: s for s, ms in cfg["seasons"].items() for m in ms}
    sh = pd.Series([seas.get(m) for m in h.index.month], index=h.index)
    mean_h = float(h.mean())
    base = float(h.quantile(cfg["base_percentile"] / 100))
    r.update({"mean_hourly_kw": mean_h, "n_hours": int(len(h)), "M2_p_base_kw": base,
              "M3_base_share": base / mean_h if mean_h > 0 else np.nan})
    we = np.isin(h.index.dayofweek, cfg["weekend_days"])
    r["M4_weekend_ratio"] = float(h[we].mean() / h[~we].mean()) if we.any() and (~we).any() and h[~we].mean() > 0 else np.nan
    above = (h - base).clip(lower=0)
    r["M5_median_above_base_year_kw"] = float(above.median())
    for s in cfg["seasons"]:
        a = above[sh == s]
        r[f"M5_median_above_base_{s}_kw"] = float(a.median()) if len(a) else np.nan
        r[f"hours_{s}"] = int(len(a))
    w = h[h.index.month.isin(cfg["winter_months"])]                      # winter-only sensitivity
    if len(w):
        wb = float(w.quantile(cfg["base_percentile"] / 100))
        wm = float(w.mean())
        wwe = np.isin(w.index.dayofweek, cfg["weekend_days"])
        r.update({"sens_winter_M2_p_base_kw": wb, "sens_winter_M3_base_share": wb / wm if wm > 0 else np.nan,
                  "sens_winter_M4_weekend_ratio": float(w[wwe].mean() / w[~wwe].mean())
                  if wwe.any() and (~wwe).any() and w[~wwe].mean() > 0 else np.nan,
                  "sens_winter_M5_median_above_winter_base_kw": float((w - wb).clip(lower=0).median())})
    daily = daily_pairs(h, temp, cfg)
    return r, daily, h


def cp_columns(fit, prefix):
    return {f"{prefix}model": fit["model"], f"{prefix}class": fit["class"], f"{prefix}T_bh_c": fit["T_bh"],
            f"{prefix}T_bc_c": fit["T_bc"], f"{prefix}b_heating_kw_per_c": fit["b_h"], f"{prefix}p_heating": fit["p_h"],
            f"{prefix}b_cooling_kw_per_c": fit["b_c"], f"{prefix}p_cooling": fit["p_c"],
            f"{prefix}adj_r2": fit["adj_r2"], f"{prefix}n_days": fit["n_days"]}


def zero_base_text(code, r, net, load, h, source, pv_flag, pv_no_prod):
    y = r
    z15 = int((load == 0).sum())
    neg15 = int((load < 0).sum())
    zh, nh = int((h == 0).sum()), int((h < 0).sum())
    longest = 0
    run = 0
    for v in (h == 0).to_numpy():
        run = run + 1 if v else 0
        longest = max(longest, run)
    lines = [f"Site {code}: P_base = {y['M2_p_base_kw']:.2f} kW (5th percentile of {len(h)} hourly values), "
             f"load series = {source}; PV flag {pv_flag}, pv_no_production {pv_no_prod}.",
             f"  hourly values == 0: {zh} ({zh / len(h):.1%}), longest run {longest} h; hourly values < 0: {nh} ({nh / len(h):.1%}); "
             f"15-min values == 0: {z15}, < 0: {neg15}; hourly min {h.min():.2f}, median {h.median():.2f} kW."]
    if zh / len(h) > 0.05 or longest >= 6:
        lines.append("  -> many exact zeros (or long zero runs): look for outages or missing values stored as 0 in the raw file.")
    if nh / len(h) > 0.05:
        lines.append("  -> negative hourly load: export from PV or other generation in the NET series"
                     + ("; no generation file, so gross load could not be formed." if not source == "gross" else "."))
    if zh / len(h) <= 0.05 and nh / len(h) <= 0.05:
        lines.append("  -> the 5th percentile is 0 with few zeros: check the raw data for a few long zero periods before using the metrics.")
    lines.append("  The metrics of this site are computed as they are and flagged in the outputs; check the raw file before using them.")
    return lines


# ------------------------------------------------------------------ plots
def plot_changepoint(pdf, code, daily, fit, source, pv_no_prod, fit_oct=None, daily_oct=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if pv_no_prod and daily_oct is not None:
        mo = daily.index.isin(daily_oct.index)
        ax.scatter(daily["t"][~mo], daily["load"][~mo], s=9, color="0.75", label="Apr-Sep (net load, PV)")
        ax.scatter(daily["t"][mo], daily["load"][mo], s=9, color="0.25", label="Oct-Mar")
    else:
        ax.scatter(daily["t"], daily["load"], s=9, color="0.35", label="daily mean")
    if len(daily):
        tt = np.linspace(daily["t"].min(), daily["t"].max(), 200)
        ax.plot(tt, cp_predict(fit, tt), color="k", lw=1.8, label=f"fit, all year ({fit['model']})")
        if fit_oct is not None and fit_oct["model"] != "flat":
            ax.plot(tt, cp_predict(fit_oct, tt), color="k", lw=1.2, ls="--", label=f"fit, Oct-Mar ({fit_oct['model']})")
        elif fit_oct is not None:
            ax.axhline(fit_oct["a"], color="k", lw=1.0, ls="--", label="fit, Oct-Mar (flat)")
        for k, lab in (("T_bh", "T_bh"), ("T_bc", "T_bc")):
            if np.isfinite(fit[k]):
                ax.axvline(fit[k], color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("daily mean outdoor temperature (deg C)")
    ax.set_ylabel(f"daily mean load, {source} (kW)")
    cap = f"Site {code}: class {fit['class']}, adj. R$^2$ = {fit['adj_r2']:.2f}, n = {fit['n_days']} days"
    if np.isfinite(fit["b_c"]):
        cap += f", cooling {fit['b_c']:.1f} kW/deg C above {fit['T_bc']:.0f} deg C"
    if np.isfinite(fit["b_h"]):
        cap += f", heating {fit['b_h']:.1f} kW/deg C below {fit['T_bh']:.0f} deg C"
    ax.set_title(cap, fontsize=8)
    if pv_no_prod:
        ax.text(0.01, 0.01, "PV without generation data: the summer cooling slope is unreliable", transform=ax.transAxes,
                fontsize=7, va="bottom")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def heatmap(mem, combos, cfg, out_dir):
    sites = list(mem.index)
    fig, ax = plt.subplots(figsize=(7.4, 0.32 * len(sites) + 1.9))
    for i, s in enumerate(sites):
        for j, c in enumerate(combos):
            v = bool(mem.loc[s, c["col"]])
            ax.add_patch(Rectangle((j, i), 1, 1, fc=cfg["pass_color"] if v else cfg["fail_color"], ec="w", lw=1.2))
            ax.text(j + 0.5, i + 0.5, "pass" if v else "fail", ha="center", va="center", fontsize=6,
                    color="w" if v else "0.35")
        ax.text(len(combos) + 0.3, i + 0.5, f"{mem.loc[s, 'selection_frequency']:.2f}", va="center", fontsize=8)
    dj = next(j for j, c in enumerate(combos) if c["default"])
    ax.add_patch(Rectangle((dj, 0), 1, len(sites), fill=False, ec="k", lw=1.6))
    ax.set_xlim(0, len(combos) + 1.1)
    ax.set_ylim(len(sites), -0.1)
    ax.set_xticks([j + 0.5 for j in range(len(combos))])
    ax.set_xticklabels([f"{c['share']:g} / {c['we']:g}" for c in combos], fontsize=7)
    ax.set_yticks([i + 0.5 for i in range(len(sites))])
    ax.set_yticklabels([f"Site {s}" for s in sites], fontsize=8)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("base-load share >= / weekend ratio >=  (coverage >= %.1f; black frame = default)" % cfg["coverage_min"], fontsize=8)
    ax.text(len(combos) + 0.3, -0.45, "selection\nfrequency", fontsize=7, va="bottom")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"alt_filter_membership_heatmap.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ tables
def tex_table(rows, header, colspec, caption, label):
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(str(x) for x in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def f0(x):
    return "--" if pd.isna(x) else f"{x:,.0f}".replace(",", r"\,")


def f1(x):
    return "--" if pd.isna(x) else f"{x:.1f}"


def f2(x):
    return "--" if pd.isna(x) else f"{x:.2f}"


# ------------------------------------------------------------------ main
def main(cfg=CONFIG):
    sys.path.insert(0, os.getcwd())
    L = __import__(cfg["lpc_module"])
    lcfg = L.CONFIG
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    code = make_coder(L, lcfg, cfg)
    sep, dec = cfg["csv_sep"], cfg["csv_decimal"]

    def wr(df, name):
        df.to_csv(os.path.join(out_dir, name), sep=sep, decimal=dec, index=False, encoding="utf-8-sig")

    df = L.read_comparison(lcfg)
    if df is None:
        raise SystemExit("[alt] no site_comparison.csv (run lpc_11 first)")
    if cfg["sites"] != "all":
        want = {str(s).strip().lower() for s in cfg["sites"]}
        df = df[df[L.SITE_KEY].astype(str).str.strip().str.lower().isin(want)]
    if not lcfg.get("weather_csv_path"):
        raise SystemExit("[alt] lpc_11 CONFIG['weather_csv_path'] is empty; the change-point model needs temperature")
    temp = L.load_smhi_temperature(lcfg)
    interval_h = pd.Timedelta(lcfg["resample_freq"]).total_seconds() / 3600
    seas = cfg["seasons"]
    combos = [{"share": s, "we": w, "col": f"share{s:g}_we{w:g}",
               "default": s == cfg["base_share_default"] and w == cfg["weekend_default"]}
              for s in cfg["base_share_values"] for w in cfg["weekend_values"]]

    rows, fits, zero_notes = [], {}, []
    for _, r0 in df.iterrows():
        sid = str(r0[L.SITE_KEY])
        c = code(sid)
        path = L.find_meter_file(r0, lcfg)
        if path is None:
            print(f"[alt] {c}: meter file not found -- skipped")
            continue
        cc = dict(lcfg, csv_path=path, show_plot=False)
        net, info = L.load_net_series(cc)
        gross = L.gross_series(net, cc)
        load, source = (gross, "gross") if gross is not None else (net, "net")
        load = load[(load.index.year == cfg["year"])]
        h = L.hourly_mean(load, cc)
        h = h[~h.index.duplicated()]
        pv_flag = str(r0.get("pv_flag")).strip().lower() in ("true", "1", "1.0")
        pv_no_prod = int(source == "net" and pv_flag)
        m, daily, h = metrics_for_site(net, load, source, h, temp, cfg, interval_h)
        row = {"site": c, "is_negative_control": r0.get("is_negative_control"), "load_series": source,
               "period": f"{cfg['year']}-01-01 to {cfg['year']}-12-31", "pv_flag": pv_flag,
               "pv_no_production": pv_no_prod}
        row.update(m)
        if h is None or h.empty:
            rows.append(row)
            continue
        fit = fit_changepoint(daily["t"], daily["load"], cfg)
        row.update(cp_columns(fit, "M6_"))
        fit_oct, d_oct = None, None
        if pv_no_prod:
            d_oct = daily[daily.index.month.isin(cfg["octmar_months"])]
            fit_oct = fit_changepoint(d_oct["t"], d_oct["load"], cfg)
            row.update(cp_columns(fit_oct, "M6_octmar_"))
        row["temp_support"] = "yes" if fit["class"] in ("cooling", "both") else "neutral"
        row["temp_support_note"] = ("summer cooling slope unreliable (net load with PV, no generation data); "
                                    f"Oct-Mar fit: {fit_oct['class']}" if pv_no_prod else "")
        row["size_vs_lfm"] = row["M5_median_above_base_year_kw"] / cfg["lfm_kw"]
        row["base_load_check"] = ""
        if row["M2_p_base_kw"] <= 0:
            row["base_load_check"] = "P_base <= 0 kW: see alt_filter_zero_base_check.txt"
            zero_notes += zero_base_text(c, row, net, load, h, source, pv_flag, pv_no_prod) + [""]
        fits[c] = (daily, fit, source, pv_no_prod, fit_oct, d_oct)
        rows.append(row)
    if not rows:
        raise SystemExit("[alt] no site could be analysed")
    M = pd.DataFrame(rows)
    if M["site"].duplicated().any():
        raise SystemExit("[alt] two sites got the same code: " + ", ".join(M.loc[M["site"].duplicated(), "site"]))

    # ---- filter and membership
    for cb in combos:
        M[cb["col"]] = ((M["M1_coverage"] >= cfg["coverage_min"]) & (M["M3_base_share"] >= cb["share"])
                        & (M["M4_weekend_ratio"] >= cb["we"])).fillna(False)
    cols = [cb["col"] for cb in combos]
    M["selection_frequency"] = M[cols].mean(axis=1)
    dcol = next(cb["col"] for cb in combos if cb["default"])
    M["alt_pass_default"] = M[dcol]

    # ---- labels
    labels = None
    if cfg.get("labels_csv") and os.path.exists(cfg["labels_csv"]):
        lab = pd.read_csv(cfg["labels_csv"], sep=None, engine="python", encoding="utf-8-sig")
        labels = {code(a): str(b).strip().lower() for a, b in zip(lab.iloc[:, 0], lab.iloc[:, 1])}
        M["label"] = M["site"].map(labels)
    else:
        M["label"] = np.nan

    # ---- current filter
    cur_pass, cur_sel, cur_freq = {}, {}, {}
    if os.path.exists(cfg["current_pass_csv"]):
        cp = pd.read_csv(cfg["current_pass_csv"], sep=sep, decimal=dec, encoding="utf-8-sig")
        for _, r in cp.iterrows():
            cur_pass[code(r["site"])] = str(r["filter_pass"]).lower() in ("true", "1", "1.0")
            cur_sel[code(r["site"])] = str(r.get("selected")).lower() in ("true", "1", "1.0")
    if os.path.exists(cfg["current_freq_csv"]):
        cf = pd.read_csv(cfg["current_freq_csv"], sep=sep, decimal=dec, encoding="utf-8-sig")
        for _, r in cf.iterrows():
            cur_freq[code(r["site_id"])] = float(r["selection_frequency"])
    M["current_pass"] = M["site"].map(cur_pass)
    M["current_selected"] = M["site"].map(cur_sel)
    M["current_frequency"] = M["site"].map(cur_freq)

    # ---- outputs: metrics, membership
    M = M.sort_values("M5_median_above_base_year_kw", ascending=False).reset_index(drop=True)
    wr(M, "alt_filter_metrics.csv")
    mem = M[["site"] + cols + ["selection_frequency"]].copy()
    for cb in combos:
        mem[f"thr_{cb['col']}"] = f"coverage>={cfg['coverage_min']:g}; base share>={cb['share']:g}; weekend>={cb['we']:g}"
    wr(mem[["site"] + cols + ["selection_frequency"]].assign(default_combination=dcol), "alt_filter_membership.csv")
    heatmap(M.set_index("site"), combos, cfg, out_dir)

    # ---- comparison
    comp = M[["site", "current_pass", "current_frequency", "alt_pass_default", "selection_frequency",
              "M5_median_above_base_winter_kw", "M5_median_above_base_year_kw", "size_vs_lfm", "temp_support",
              "label", "base_load_check"]].rename(columns={"selection_frequency": "alt_frequency",
                                                           "M5_median_above_base_winter_kw": "median_above_base_winter_kw",
                                                           "M5_median_above_base_year_kw": "median_above_base_year_kw"})
    comp["is_negative_control"] = M["is_negative_control"]
    wr(comp, "filter_comparison.csv")
    pf = lambda v: "--" if pd.isna(v) else ("pass" if v else "fail")
    trs = []
    for _, r in comp.iterrows():
        trs.append([f"Site {r['site']}" + ("$^{\\dagger}$" if r["base_load_check"] else ""),
                    pf(r["current_pass"]), f2(r["current_frequency"]), pf(r["alt_pass_default"]), f2(r["alt_frequency"]),
                    f1(r["median_above_base_winter_kw"]), f1(r["median_above_base_year_kw"]), r["temp_support"],
                    "--" if pd.isna(r["label"]) else str(r["label"]).replace("_", r"\_")])
    cap = (f"Current and alternative screening filter, sorted by median load above base load (year). Alternative: coverage "
           f"$\\geq$ {cfg['coverage_min']:g}, base-load share $\\geq$ {cfg['base_share_default']:g}, weekend ratio $\\geq$ "
           f"{cfg['weekend_default']:g}; frequency = share of the sweep that passes (current: 81 combinations, alternative: "
           f"{len(combos)}). Median above base is against the full-year base load ({'gross' if (M['load_series'] == 'gross').all() else 'gross or net, see metrics file'} load, {cfg['year']}). "
           f"Temperature support: cooling slope in the change-point model. $^{{\\dagger}}$\\,base load $\\leq$ 0 kW, check the raw data.")
    with open(os.path.join(out_dir, "filter_comparison.tex"), "w", encoding="utf-8") as fh:
        fh.write(tex_table(trs, ["Site", "Current", "Freq.", "Alt.", "Freq.", "Winter (kW)", "Year (kW)", "Temp.", "Label"],
                           "lccccrrll", cap, "tab:filter_comparison"))

    # ---- change-point pages
    with PdfPages(os.path.join(out_dir, "alt_filter_changepoint.pdf")) as pdf:
        for c in M["site"]:
            if c in fits:
                daily, fit, source, pvn, fit_oct, d_oct = fits[c]
                plot_changepoint(pdf, c, daily, fit, source, pvn, fit_oct, d_oct)

    # ---- zero-base check
    zpath = os.path.join(out_dir, "alt_filter_zero_base_check.txt")
    if zero_notes:
        with open(zpath, "w", encoding="utf-8") as fh:
            fh.write("\n".join(zero_notes))
    elif os.path.exists(zpath):
        os.remove(zpath)

    # ---- labels cross-tab
    lab_txt = "labels: not used (no site_labels.csv)"
    if labels is not None:
        known = M[M["label"].isin(["cold", "not_cold"])]
        unk = M[~M["label"].isin(["cold", "not_cold"])]
        ct_rows, tex_rows = [], []
        for name, col in (("current filter", "current_pass"), ("alternative filter (default)", "alt_pass_default")):
            sel = known[col].map(lambda v: v is True or v == True)
            a = int(((sel) & (known["label"] == "cold")).sum()); b = int(((sel) & (known["label"] == "not_cold")).sum())
            cc_ = int(((~sel) & (known["label"] == "cold")).sum()); d = int(((~sel) & (known["label"] == "not_cold")).sum())
            ct_rows.append({"filter": name, "selected_cold": a, "selected_not_cold": b, "not_selected_cold": cc_,
                            "not_selected_not_cold": d, "unknown_left_out": int(len(unk))})
            tex_rows += [[name, "selected", a, b], ["", "not selected", cc_, d]]
        wr(pd.DataFrame(ct_rows), "label_crosstab.csv")
        with open(os.path.join(out_dir, "label_crosstab.tex"), "w", encoding="utf-8") as fh:
            fh.write(tex_table(tex_rows, ["Filter", "", "cold", "not cold"], "llrr",
                               f"Selected and not selected sites by label ({len(unk)} sites labelled unknown or without label left out).",
                               "tab:filter_labels"))
        lab_txt = f"labels: used ({cfg['labels_csv']}); {len(unk)} site(s) unknown or unlabelled left out of the 2x2 tables"

    # ---- header and summary
    cur_set = set(M.loc[M["current_pass"] == True, "site"])
    alt_set = set(M.loc[M["alt_pass_default"] == True, "site"])
    join, leave = sorted(alt_set - cur_set), sorted(cur_set - alt_set)

    def grp(sites, col):
        return float(M.loc[M["site"].isin(sites), col].sum())
    big = lambda sites: int((M.loc[M["site"].isin(sites), "M5_median_above_base_year_kw"] >= cfg["lfm_kw"]).sum())
    srcs = M["load_series"].value_counts().to_dict()
    head = [
        "alt_filter.py -- header of this run (exploratory; the case group is unchanged)",
        f"data source: {srcs} (gross = net + generation; net where no generation file; pv_no_production = PV flagged and no gross: "
        f"{sorted(M.loc[M['pv_no_production'] == 1, 'site'])})",
        f"period: {cfg['year']}-01-01 to {cfg['year']}-12-31, 15-min active power, hourly = mean of the 15-min values; temperature: {lcfg['weather_csv_path']}",
        f"seasons (own definition): {cfg['seasons']}; change-point grid T_bh {cfg['heating_balance_c'][0]}-{cfg['heating_balance_c'][-1]}, "
        f"T_bc {cfg['cooling_balance_c'][0]}-{cfg['cooling_balance_c'][-1]} degC, slope p < {cfg['slope_p_max']}",
        f"thresholds: coverage >= {cfg['coverage_min']}; base-load share {cfg['base_share_values']} (default {cfg['base_share_default']}); "
        f"weekend ratio {cfg['weekend_values']} (default {cfg['weekend_default']}); {len(combos)} combinations; size reported against {cfg['lfm_kw']:g} kW",
        lab_txt,
        f"sites analysed: {len(M)}; current filter pass (defaults): {sorted(cur_set)}; alternative pass (defaults): {sorted(alt_set)}",
        f"join (alternative pass, current fail): {join or 'none'}; leave (current pass, alternative fail): {leave or 'none'}",
        f"summed median above base, current group: winter {grp(cur_set, 'M5_median_above_base_winter_kw'):.1f} kW, year "
        f"{grp(cur_set, 'M5_median_above_base_year_kw'):.1f} kW; alternative group: winter {grp(alt_set, 'M5_median_above_base_winter_kw'):.1f} kW, "
        f"year {grp(alt_set, 'M5_median_above_base_year_kw'):.1f} kW",
        f"selected sites with median above base (year) >= {cfg['lfm_kw']:g} kW: current {big(cur_set)} of {len(cur_set)}; alternative {big(alt_set)} of {len(alt_set)}",
        f"sites with base load <= 0 kW: {sorted(M.loc[M['base_load_check'] != '', 'site']) or 'none'}"
        + (" (see alt_filter_zero_base_check.txt)" if zero_notes else ""),
        "temperature support: " + ", ".join(f"{k} {v}" for k, v in M["temp_support"].value_counts().items()),
        "figure captions: alt_filter_membership_heatmap -- pass/fail of each site for the 9 threshold combinations "
        "(base-load share / weekend ratio), black frame = default, number = selection frequency; alt_filter_changepoint.pdf -- "
        "daily mean load against daily mean outdoor temperature with the selected change-point model, one page per site.",
        "note: the 'current' median above base in lpc_11 uses a seasonal P5 base; here every site uses the full-year P5 base (M2).",
    ]
    with open(os.path.join(out_dir, "alt_filter_header.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(head) + "\n")
    print("\n".join(head[6:10]))
    print(M[["site", "M1_coverage", "M2_p_base_kw", "M3_base_share", "M4_weekend_ratio", "M5_median_above_base_year_kw",
             "M6_class", "temp_support", "alt_pass_default", "selection_frequency", "current_pass"]].round(3).to_string(index=False))
    print(f"[alt] outputs in {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()

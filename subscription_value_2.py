"""
subscription_value.py -- value of the pool's flexibility towards the
subscription (script 9; task_subscription_value.md).

Question: in the hours that set the monthly peak of the area substation load,
how much can the pool cut, what is that worth to the DSO, and how much energy
does it take? Method: FlexAbility delrapport 3 (Plexigrid and Ellevio, 2025),
chapter on optimising the subscription: lower the level step by step, sum the
energy that must be cut, value the saving, compare.

Subscription facts (I3 and E.ON correspondence; config values):
  * the level is set PER MONTH, cost p_sub = 40 SEK/kW and month;
  * an overrun is charged only in the month it happens, on the excess:
    overrun_factor x excess x p_sub = 2 x 40 = 80 SEK per kW of excess;
  * withdrawal only, no time-dependent fee; measured value = highest hourly
    average power of the month.
Not known, so assumed and labelled (A1-A7, all in CONFIG and in the header file):
  A1 excess e above the monthly level before any flexibility (100/250/500/1000 kW)
  A2 level S_m = monthly peak P_m of the substation series - e
  A3 perfect foresight (the pool is called exactly in the hours above S_m);
     sensitivity: also call in hours above S_m - 5 % of peak (see below)
  A4 pool offer = firm level (P10), f = 0.5; sensitivity f 0.3 / 1.0, P5, P20
  A5 p_sub = 40, overrun factor 2; sensitivity p_sub 30 / 50
  A6 the flexibility price is NOT estimated: the result is a break-even price
     per delivered MWh, compared with reference lines (FlexAbility pp. 19-22)
  A7 rebound: off; if switched on, scenario shares in CONFIG["rebound"] are
     used (assumptions, they are not from script 6: there are no activations)

For each month m and excess e:
  1. P_m = max A_h; S_m = P_m - e.
  2. H = hours with A_h > S_m, required cut r_h = A_h - S_m; consecutive hours
     form an event (number, longest event in h).
  3. E_m(e) = sum r_h / 1000 (MWh): the FlexAbility "need" curve.
  4. Feasible if r_h <= offerable power in every hour of H. The offer in hour h
     is the pool level for the slice (month, hour, weekday/weekend) and block
     length min(L, 4) h, L = length of the event; the smallest available block
     >= that is used (a 3 h event uses the 4 h block, which is the minimum over
     more hours, so conservative). Events longer than 4 h are flagged "beyond 4 h".
     A slice missing in pool_levels counts as offering 0 kW (counted).
  5. If not feasible: d = largest delivered reduction <= e (step 10 kW) at which
     every hour is covered, i.e. the peak is lowered to P_m - d and every hour
     above that level is cut to it. d is found by bisection (feasibility is
     monotone in d). d = 0 when even the smallest step fails.
  6. value = p_sub x d (value_basis "level": the DSO can ask for a level d kW
     lower; default) or overrun_factor x p_sub x d ("overrun"). Both are also
     written as value_level_SEK and value_overrun_SEK.
  7. break-even = value / E_m(d) in SEK per MWh: the most the DSO
     could pay per delivered MWh and still break even. It is not a bid.
  8. Forecast-margin sensitivity (interpretation, stated in the header): the
     pool is called in every hour above S_m - margin (margin = 5 % of P_m) and
     cuts those hours to that lower level, so E_m(d) and the feasibility
     requirement both grow; the value is still counted on d.
Do not tune e or f to get a feasible result: infeasible months stay infeasible.

Where e >= P_m the level would be <= 0: the row is written with a note and no
numbers.

Outputs (subscription_value_out/):
  subscription_value_monthly.csv   every case x month x e (case "base" is the main result)
  subscription_value_curves.csv    base case, e = 50..1000 step 50, per month and "year"
  subscription_value_limits.csv    per month: largest reduction the pool covers (base)
  subscription_value_sensitivity.csv  annual summary per case and e
  subscription_value_rebound.csv   only if rebound is switched on
  subscription_value_table_e250.tex   LaTeX booktabs table (e = CONFIG["table_e_kw"])
  subscription_value_header.txt    series, f, percentile, block lengths, prices, assumptions, checks
  fig_need_energy.png/.pdf         Figure 1 (FlexAbility Figure 8 analogue)
  fig_breakeven.png/.pdf           Figure 2 (FlexAbility Figure 10 analogue)
Every row of the monthly file also records series (proxy or actual), f,
percentile, source, p_sub and overrun factor. No site data are read, only the
pool sum, so no site codes occur.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG = {
    # --- series A_h (hourly mean load of the area substation, withdrawal)
    "proxy_csv": os.path.join("screening_out", "substation_proxy_hourly.csv"),   # lpc_11 --export-proxy;
    "proxy_value_col": "load_kw",                                                # replace by the summed three-point series
    "series_is_proxy": True,                                                     # False once the actual series is used
    "series_label": "area substation load (PROXY)",
    "year": 2025,
    # --- pool offerable levels (offerable_flex.py)
    "pool_levels_csv": os.path.join("offerable_flex_out", "pool_levels.csv"),
    "f": 0.5,
    "level_col": "firm_P10_kw",
    "percentile_label": "P10",
    "source_label": "15 min",
    "blocks_h": [1, 2, 4],
    "max_block_h": 4,
    "weekend_days": [5, 6],
    # --- excess and prices
    "excess_list_kw": [100, 250, 500, 1000],              # A1
    "curve_e_kw": list(range(50, 1001, 50)),
    "table_e_kw": 250,
    "step_kw": 10,                                        # step of the delivered reduction d
    "p_sub": 40.0,                                        # SEK per kW and month (I3's calculation assumption)
    "overrun_factor": 2.0,
    "value_basis": "level",                               # "level": value = p_sub x d (a level d kW lower)
                                                          # "overrun": value = overrun_factor x p_sub x d
    # --- one-at-a-time sensitivities (A3-A5); set a list to [] to drop it
    "sensitivity": {
        "f": [0.3, 1.0],
        "level_col": {"P5": "sens_P5_P5_kw", "P20": "sens_P20_P20_kw"},
        "p_sub": [30.0, 50.0],
        "forecast_margin_share": [0.05],                  # share of the monthly peak
    },
    # --- reference prices, SEK/MWh (FlexAbility pp. 19-22)
    "reference_prices": {
        "availability_low": 200.0, "availability_high": 1000.0,
        "activation_average_2024_25": 3300.0, "pessimistic": 10000.0,
    },
    # --- A7 rebound (off). Shares of the energy cut in an event that returns in
    # the 1st, 2nd, ... hour after it. ASSUMPTIONS, not measured.
    "rebound": {
        "enabled": False,
        "scenarios": {
            "no control (all back in hour 1)": [1.0],
            "staggered restart over 2 h": [0.5, 0.5],
            "staggered restart over 3 h": [1 / 3, 1 / 3, 1 / 3],
            "pre-cooling (no rebound)": [0.0],
        },
    },
    "output_dir": "subscription_value_out",
    "csv_sep": ";",
    "csv_decimal": ",",
}

def value_factor(cfg):
    """SEK per kW of reduction = value_factor x p_sub."""
    return 1.0 if cfg["value_basis"] == "level" else cfg["overrun_factor"]


VALUE_TXT = {"level": "lower subscription level", "overrun": "avoided overrun"}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------- input
def read_series(cfg):
    s = pd.read_csv(cfg["proxy_csv"], sep=cfg["csv_sep"], decimal=cfg["csv_decimal"], encoding="utf-8-sig")
    t = pd.to_datetime(s.iloc[:, 0].astype(str).str.slice(0, 19))
    a = pd.Series(pd.to_numeric(s[cfg["proxy_value_col"]], errors="coerce").to_numpy(), index=t).dropna()
    a = a[~a.index.duplicated()].sort_index()
    a = a[a.index.year == cfg["year"]]
    if a.empty:
        raise SystemExit(f"[sub] no hours of {cfg['year']} in {cfg['proxy_csv']}")
    neg = a < 0
    clipped = neg.groupby(a.index.month).sum().reindex(range(1, 13), fill_value=0).astype(int)
    return a.clip(lower=0.0), clipped


class Month:
    """Hourly data of one month, as arrays."""

    def __init__(self, m, a, cfg):
        x = a[a.index.month == m]
        self.month = m
        self.n = len(x)
        self.th = (x.index.values.astype("datetime64[h]").astype("int64") if self.n else np.array([], dtype="int64"))
        self.vals = x.to_numpy(float)
        self.hour = x.index.hour.to_numpy()
        self.dt = np.where(np.isin(x.index.dayofweek, cfg["weekend_days"]), "weekend", "weekday")
        self.peak = float(self.vals.max()) if self.n else np.nan
        self.i_peak = int(self.vals.argmax()) if self.n else None


def read_pool(cfg):
    p = pd.read_csv(cfg["pool_levels_csv"], sep=cfg["csv_sep"], decimal=cfg["csv_decimal"], encoding="utf-8-sig")
    return p


def offer_dict(pool, f, col):
    q = pool[np.isclose(pool["f"].astype(float), f)]
    if q.empty or col not in q:
        return None
    d = {}
    for b, m, h, dt, v in zip(q["block_h"], q["month"], q["hour_of_day"], q["day_type"], q[col]):
        if pd.notna(v):
            d[(int(b), int(m), int(h), str(dt))] = float(v)
    return d, int(q["n_sites"].iloc[0]) if "n_sites" in q else None


# ---------------------------------------------------------------- engine
def _events(md, level):
    idx = np.flatnonzero(md.vals > level)
    if idx.size == 0:
        return idx, []
    cut = np.flatnonzero(np.diff(md.th[idx]) != 1) + 1
    return idx, np.split(np.arange(idx.size), cut)


def need_mwh(md, level):
    v = md.vals[md.vals > level]
    return float((v - level).sum() / 1000.0)


def _block(L, cfg):
    want = min(L, cfg["max_block_h"])
    ok = [b for b in sorted(cfg["blocks_h"]) if b >= want]
    return ok[0] if ok else max(cfg["blocks_h"])


def feasible_at(md, x, margin, offer, cfg):
    """(ok, n_missing_slices) when the peak is lowered by x (and cut to margin below)."""
    level = md.peak - x - margin
    idx, groups = _events(md, level)
    ok, miss = True, 0
    for g in groups:
        b = _block(len(g), cfg)
        for k in g:
            i = idx[k]
            o = offer.get((b, md.month, int(md.hour[i]), str(md.dt[i])))
            if o is None:
                miss += 1
                o = 0.0
            if md.vals[i] - level > o + 1e-9:
                ok = False
    return ok, miss


def largest_reduction(md, e, margin, offer, cfg):
    """Largest d <= e on the step grid at which every hour is covered (0 if none)."""
    step = cfg["step_kw"]
    xs = sorted({0.0, float(e)} | {float(k * step) for k in range(1, int(e // step) + 1)})
    lo, hi, best = 0, len(xs) - 1, -1
    while lo <= hi:                                  # feasibility is monotone in x
        mid = (lo + hi) // 2
        if feasible_at(md, xs[mid], margin, offer, cfg)[0]:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return xs[best] if best >= 0 else 0.0


def evaluate(md, e, case, offer, cfg):
    row = {"case": case["name"], "month": md.month, "e_kW": e, "P_m_kW": md.peak, "S_m_kW": md.peak - e,
           "hours_in_data": md.n}
    if md.n == 0 or e >= md.peak:
        row["note"] = "no data" if md.n == 0 else "e >= monthly peak: level would be <= 0, not evaluated"
        return row
    margin = case["margin_share"] * md.peak
    S = md.peak - e
    idx, groups = _events(md, S)
    lens = [len(g) for g in groups]
    idx_c, _ = _events(md, S - margin)
    row.update({"hours_above": int(idx.size), "hours_called": int(idx_c.size), "events": len(groups),
                "longest_event_h": max(lens) if lens else 0,
                "events_beyond_4h": int(sum(1 for L in lens if L > cfg["max_block_h"])),
                "E_MWh": need_mwh(md, S)})
    ok, miss = feasible_at(md, e, margin, offer, cfg)
    d = e if ok else largest_reduction(md, e, margin, offer, cfg)
    e_del = need_mwh(md, md.peak - d - margin) if d > 0 else 0.0
    v_over = cfg["overrun_factor"] * case["p_sub"] * d
    v_level = case["p_sub"] * d
    v_main = value_factor(cfg) * case["p_sub"] * d
    row.update({"feasible": "y" if ok else "n", "d_kW": d, "E_delivered_MWh": e_del,
                "value_SEK": v_main, "value_overrun_SEK": v_over, "value_level_SEK": v_level,
                "breakeven_SEK_per_MWh": (v_main / e_del) if d > 0 and e_del > 0 else np.nan,
                "pool_firm_at_peak_hour_kW": offer.get((1, md.month, int(md.hour[md.i_peak]), str(md.dt[md.i_peak]))),
                "n_missing_slices": miss, "note": ""})
    return row


def rebound_rows(md, a_all_by_hour, e, d, case, cfg):
    """A7: load in the hours after each event, with the returning energy added."""
    out = []
    if d <= 0 or e >= md.peak:
        return out
    margin = case["margin_share"] * md.peak
    level = md.peak - d - margin
    S = md.peak - e
    idx, groups = _events(md, level)
    for name, shares in cfg["rebound"]["scenarios"].items():
        worst, n_exc = -np.inf, 0
        for g in groups:
            C = float((md.vals[idx[g]] - level).sum())           # kWh cut in the event
            end = md.th[idx[g[-1]]]
            peak_post = -np.inf
            for j, s in enumerate(shares, start=1):
                base = a_all_by_hour.get(int(end) + j)
                if base is not None:
                    peak_post = max(peak_post, base + s * C)
            if np.isfinite(peak_post):
                worst = max(worst, peak_post)
                n_exc += int(peak_post > S)
        out.append({"case": case["name"], "month": md.month, "e_kW": e, "d_kW": d, "scenario": name,
                    "max_post_event_load_kW": worst if np.isfinite(worst) else np.nan, "S_m_kW": S,
                    "events": len(groups), "events_post_load_above_S": n_exc,
                    "exceeds_S_m": "y" if n_exc else "n"})
    return out


# ---------------------------------------------------------------- cases
def build_cases(cfg, pool):
    base = {"name": "base", "f": cfg["f"], "col": cfg["level_col"], "pct": cfg["percentile_label"],
            "p_sub": cfg["p_sub"], "margin_share": 0.0}
    cases = [base]
    s = cfg["sensitivity"]
    for f in s.get("f", []):
        cases.append(dict(base, name=f"f={f:g}", f=f))
    for pct, col in (s.get("level_col") or {}).items():
        cases.append(dict(base, name=f"percentile={pct}", col=col, pct=pct))
    for p in s.get("p_sub", []):
        cases.append(dict(base, name=f"p_sub={p:g}", p_sub=p))
    for m in s.get("forecast_margin_share", []):
        cases.append(dict(base, name=f"forecast margin {m:.0%} of peak", margin_share=m))
    keep, offers = [], {}
    for c in cases:
        o = offer_dict(pool, c["f"], c["col"])
        if o is None:
            print(f"[sub] case '{c['name']}' skipped: f={c['f']} / column {c['col']} not in pool levels")
            continue
        offers[c["name"]], c["n_sites"] = o
        keep.append(c)
    if not keep or keep[0]["name"] != "base":
        raise SystemExit("[sub] the base case (f / level_col of CONFIG) is not in pool_levels.csv")
    return keep, offers


# ---------------------------------------------------------------- output helpers
INT_COLS = ["month", "hours_above", "hours_called", "events", "longest_event_h", "events_beyond_4h",
            "n_missing_slices", "n_pool_sites", "hours_in_data", "hours_clipped", "months_valid", "months_feasible",
            "months_evaluated"]


def write(df, cfg, name):
    df = df.copy()
    for c in INT_COLS:
        if c in df and not (c == "month" and df[c].dtype == object):
            df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
    path = os.path.join(cfg["output_dir"], name)
    df.to_csv(path, sep=cfg["csv_sep"], decimal=cfg["csv_decimal"], index=False, encoding="utf-8-sig")
    return path


def save(fig, cfg, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(cfg["output_dir"], f"{stem}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def month_styles():
    cmap = plt.get_cmap("viridis")
    ls = ["-", "--", "-.", ":"]
    return {m: (cmap((m - 1) / 11.0), ls[(m - 1) % 4]) for m in range(1, 13)}


def fig_need(cfg, curves, limits, mds, case_base):
    st = month_styles()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for m in range(1, 13):
        c = curves[(curves["month"] == m)].dropna(subset=["E_MWh"])
        if c.empty:
            continue
        ax.plot(c["e_kW"], c["E_MWh"], color=st[m][0], ls=st[m][1], lw=1.4, label=MONTHS[m - 1])
        L = limits[limits["month"] == m].iloc[0]
        if L["d_max_kW"] > 0:
            ax.plot([L["d_max_kW"]], [L["E_at_d_max_MWh"]], marker="o", ms=6, mfc="w", mec="k", color=st[m][0], zorder=5)
    ax.plot([], [], marker="o", ls="", mfc="w", mec="k",
            label=f"largest reduction the pool covers\n({case_base['pct']}, f = {case_base['f']:g}, {cfg['source_label']})")
    ax.set_xlabel("reduction of the monthly peak (kW)")
    ax.set_ylabel("energy to cut, E (MWh per month)")
    ax.set_title(f"Energy needed to lower the monthly peak, {cfg['year']} ({cfg['series_label']})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, cfg, "fig_need_energy")


def fig_breakeven(cfg, mds, limits, case_base):
    st = month_styles()
    p, k = case_base["p_sub"], value_factor(cfg)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    step = cfg["step_kw"]
    for m, md in mds.items():
        if md.n == 0:
            continue
        xmax = min(max(cfg["curve_e_kw"]), int((md.peak - 1) // step) * step)
        xs = np.arange(step, xmax + 1, step, dtype=float)
        if xs.size == 0:
            continue
        be = np.array([k * p * x / need_mwh(md, md.peak - x) for x in xs])
        dmax = float(limits[limits["month"] == m]["d_max_kW"].iloc[0])
        ok = xs <= dmax
        ax.plot(xs[ok], be[ok], color=st[m][0], ls="-", lw=1.5, label=MONTHS[m - 1])
        if (~ok).any():
            seg = np.r_[max(ok.sum() - 1, 0):len(xs)] if ok.any() else np.arange(len(xs))
            ax.plot(xs[seg], be[seg], color=st[m][0], ls=":", lw=1.1)
    r = cfg["reference_prices"]
    ax.axhspan(r["availability_low"], r["availability_high"], color="0.85", zorder=0)
    ax.text(ax.get_xlim()[1], np.sqrt(r["availability_low"] * r["availability_high"]),
            "availability price 200-1,000 ", ha="right", va="center", fontsize=7)
    for key, lab in (("activation_average_2024_25", "activation average 2024/25: 3,300"),
                     ("pessimistic", "pessimistic: 10,000")):
        ax.axhline(r[key], color="k", lw=0.9, ls="--")
        ax.text(ax.get_xlim()[1], r[key], lab + " ", ha="right", va="bottom", fontsize=7)
    ax.plot([], [], color="k", ls="-", label="covered by the pool")
    ax.plot([], [], color="k", ls=":", label="beyond the pool's firm level")
    ax.set_yscale("log")
    ax.set_xlabel("reduction of the monthly peak (kW)")
    ax.set_ylabel("break-even price (SEK per delivered MWh)")
    ax.set_title(f"Most the DSO could pay per MWh, {cfg['year']} ({cfg['series_label']}; "
                 + (f"{p:g} SEK/kW, lower level)" if cfg["value_basis"] == "level" else f"{k:g} x {p:g} SEK/kW, avoided overrun)"), fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, cfg, "fig_breakeven")


def fmt(x, nd=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:,.{nd}f}".replace(",", r"\,")


def latex_table(cfg, monthly, case_base):
    e = cfg["table_e_kw"]
    t = monthly[(monthly["case"] == "base") & (monthly["e_kW"] == e)].sort_values("month")
    lines = [r"\begin{table}[htbp]", r"\centering",
             rf"\caption{{Peak reduction towards the subscription at an excess of {e} kW above the monthly level, "
             rf"{cfg['year']}. Series: {cfg['series_label']}. Pool offer: firm level ({case_base['pct']}), "
             rf"f = {case_base['f']:g}, {cfg['source_label']} data. $S_m$ is the level, $d$ the largest reduction the "
             rf"pool covers in every hour above it, and the break-even price is the value ("
             rf"{VALUE_TXT[cfg['value_basis']]}) divided by the "
             rf"energy cut. $^{{*}}$\,the pool cannot cover the full excess. Value: {value_factor(cfg):g} "
             rf"$\times$ {case_base['p_sub']:g}~SEK per kW and month.}}",
             r"\label{tab:subscription_value}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Month & $S_m$ (kW) & Hours above & $E$ (MWh) & $d$ (kW) & Value (SEK) & Break-even (SEK/MWh) \\",
             r"\midrule"]
    tot = {"h": 0, "E": 0.0, "ov": 0.0, "Ed": 0.0}
    for _, r in t.iterrows():
        if pd.isna(r.get("hours_above")):
            lines.append(f"{MONTHS[int(r['month']) - 1]} & \\multicolumn{{6}}{{l}}{{not evaluated (excess $\\geq$ peak)}} \\\\")
            continue
        star = r"$^{*}$" if r["feasible"] == "n" else ""
        lines.append(f"{MONTHS[int(r['month']) - 1]} & {fmt(r['S_m_kW'])} & {int(r['hours_above'])} & {fmt(r['E_MWh'], 1)} & "
                     f"{fmt(r['d_kW'])}{star} & {fmt(r['value_SEK'])} & {fmt(r['breakeven_SEK_per_MWh'])} \\\\")
        tot["h"] += int(r["hours_above"]); tot["E"] += r["E_MWh"]; tot["ov"] += r["value_SEK"]
        tot["Ed"] += r["E_delivered_MWh"]
    be = tot["ov"] / tot["Ed"] if tot["Ed"] > 0 else np.nan
    lines += [r"\midrule", f"Year & -- & {tot['h']} & {fmt(tot['E'], 1)} & -- & {fmt(tot['ov'])} & {fmt(be)} \\\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(cfg["output_dir"], "subscription_value_table_e%d.tex" % e)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------- main
def main(cfg=CONFIG):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    a, clipped = read_series(cfg)
    pool = read_pool(cfg)
    cases, offers = build_cases(cfg, pool)
    mds = {m: Month(m, a, cfg) for m in range(1, 13)}
    e_all = sorted(set(cfg["excess_list_kw"]) | set(cfg["curve_e_kw"]))
    proxy_flag = "proxy" if cfg["series_is_proxy"] else "actual"

    rows, rb_rows = [], []
    a_by_hour = dict(zip(a.index.values.astype("datetime64[h]").astype("int64"), a.to_numpy(float)))
    for c in cases:
        off = offers[c["name"]]
        for m, md in mds.items():
            for e in e_all:
                r = evaluate(md, e, c, off, cfg)
                r.update({"f": c["f"], "percentile": c["pct"], "source": cfg["source_label"],
                          "series": cfg["series_label"], "proxy_flag": proxy_flag, "p_sub": c["p_sub"],
                          "overrun_factor": cfg["overrun_factor"], "forecast_margin_share": c["margin_share"],
                          "block_lengths_h": "/".join(str(b) for b in cfg["blocks_h"]), "n_pool_sites": c.get("n_sites"),
                          "hours_clipped": int(clipped.get(m, 0))})
                rows.append(r)
                if cfg["rebound"]["enabled"] and c["name"] == "base" and r.get("d_kW", 0) > 0:
                    rb_rows += rebound_rows(md, a_by_hour, e, r["d_kW"], c, cfg)
    cols = ["case", "month", "e_kW", "S_m_kW", "P_m_kW", "hours_above", "hours_called", "events", "longest_event_h",
            "events_beyond_4h", "E_MWh", "feasible", "d_kW", "E_delivered_MWh", "value_SEK", "value_overrun_SEK", "value_level_SEK",
            "breakeven_SEK_per_MWh", "pool_firm_at_peak_hour_kW", "n_missing_slices", "f", "percentile", "source",
            "proxy_flag", "series", "p_sub", "overrun_factor", "forecast_margin_share", "block_lengths_h",
            "n_pool_sites", "hours_in_data", "hours_clipped", "note"]
    allm = pd.DataFrame(rows).reindex(columns=cols)
    monthly = allm[allm["e_kW"].isin(cfg["excess_list_kw"]) | (allm["e_kW"] == cfg["table_e_kw"])]
    write(monthly.round(3), cfg, "subscription_value_monthly.csv")

    base = cases[0]
    b = allm[allm["case"] == "base"]

    # curves: base case, per month and for the year
    cur = b[b["e_kW"].isin(cfg["curve_e_kw"])][["month", "e_kW", "E_MWh", "d_kW", "E_delivered_MWh",
                                                  "value_SEK", "value_overrun_SEK", "value_level_SEK", "feasible"]].copy()
    yr = cur.groupby("e_kW").agg(E_MWh=("E_MWh", "sum"), d_kW=("d_kW", "sum"), E_delivered_MWh=("E_delivered_MWh", "sum"),
                                 value_SEK=("value_SEK", "sum"),
                                 value_overrun_SEK=("value_overrun_SEK", "sum"), value_level_SEK=("value_level_SEK", "sum"),
                                 months_valid=("E_MWh", "count"),
                                 months_feasible=("feasible", lambda s: int((s == "y").sum()))).reset_index()
    yr.insert(0, "month", "year")
    yr["breakeven_SEK_per_MWh"] = np.where(yr["E_delivered_MWh"] > 0, yr["value_SEK"] / yr["E_delivered_MWh"], np.nan)
    curves = pd.concat([cur.assign(month=cur["month"].astype(int)), yr.drop(columns=[])], ignore_index=True)
    for k, v in (("f", base["f"]), ("percentile", base["pct"]), ("source", cfg["source_label"]), ("series", cfg["series_label"]),
                 ("proxy_flag", proxy_flag), ("p_sub", base["p_sub"]), ("overrun_factor", cfg["overrun_factor"])):
        curves[k] = v
    write(curves.round(3), cfg, "subscription_value_curves.csv")

    # limits: largest reduction the pool covers per month (base)
    lim = []
    off = offers["base"]
    for m, md in mds.items():
        if md.n == 0:
            lim.append({"month": m, "P_m_kW": np.nan, "d_max_kW": 0.0, "E_at_d_max_MWh": 0.0, "breakeven_at_d_max": np.nan})
            continue
        cap = min(max(cfg["curve_e_kw"]), int((md.peak - 1) // cfg["step_kw"]) * cfg["step_kw"])
        d = largest_reduction(md, cap, 0.0, off, cfg) if cap > 0 else 0.0
        E = need_mwh(md, md.peak - d) if d > 0 else 0.0
        lim.append({"month": m, "P_m_kW": md.peak, "d_max_kW": d, "E_at_d_max_MWh": E,
                    "breakeven_at_d_max": value_factor(cfg) * base["p_sub"] * d / E if E > 0 else np.nan})
    limits = pd.DataFrame(lim)
    write(limits.round(3), cfg, "subscription_value_limits.csv")

    # sensitivity: annual summary per case and e
    g = allm[allm["e_kW"].isin(cfg["excess_list_kw"])].groupby(["case", "e_kW"])
    sens = g.agg(months_evaluated=("E_MWh", "count"), months_feasible=("feasible", lambda s: int((s == "y").sum())),
                 E_need_MWh=("E_MWh", "sum"), d_sum_kW=("d_kW", "sum"), E_delivered_MWh=("E_delivered_MWh", "sum"),
                 value_SEK=("value_SEK", "sum"),
                 value_overrun_SEK=("value_overrun_SEK", "sum"), value_level_SEK=("value_level_SEK", "sum")).reset_index()
    sens["breakeven_SEK_per_MWh"] = np.where(sens["E_delivered_MWh"] > 0, sens["value_SEK"] / sens["E_delivered_MWh"], np.nan)
    write(sens.round(3), cfg, "subscription_value_sensitivity.csv")

    if rb_rows:
        write(pd.DataFrame(rb_rows).round(3), cfg, "subscription_value_rebound.csv")
    elif cfg["rebound"]["enabled"]:
        print("[sub] rebound on, but no month with a delivered reduction > 0")

    tex = latex_table(cfg, monthly, base)
    fig_need(cfg, curves[curves["month"] != "year"].assign(month=lambda d: d["month"].astype(int)), limits, mds, base)
    fig_breakeven(cfg, mds, limits, base)

    # checks
    checks = []
    for (case, m), q in allm.dropna(subset=["E_MWh"]).groupby(["case", "month"]):
        q = q.sort_values("e_kW")
        if (np.diff(q["E_MWh"].to_numpy()) < -1e-9).any():
            checks.append(f"FAIL E_m(e) decreasing: case {case}, month {m}")
    v = allm.dropna(subset=["d_kW"])
    if (v["d_kW"] > v["e_kW"] + 1e-9).any():
        checks.append("FAIL d > e in some row")
    if ((v["d_kW"] == 0) & ((v["value_overrun_SEK"] != 0) | (v["value_level_SEK"] != 0))).any():
        checks.append("FAIL value > 0 with d = 0")
    ann = curves[curves["month"] == "year"].set_index("e_kW")
    mon = curves[curves["month"] != "year"].groupby("e_kW")["value_SEK"].sum()
    if not np.allclose(ann["value_SEK"].to_numpy(float), mon.reindex(ann.index).to_numpy(float), equal_nan=True):
        checks.append("FAIL annual value is not the sum of the months")
    if not checks:
        checks = ["all checks passed: E_m(e) non-decreasing in e; d <= e; value = 0 when d = 0; annual sum = sum of months"]

    head = [
        "subscription_value.py -- header of this run",
        f"series: {cfg['series_label']} ({proxy_flag}); file {cfg['proxy_csv']}; year {cfg['year']}",
        f"hours of the series: {len(a)}; negative values (feed-in) clipped to 0: {int(clipped.sum())} hours "
        f"(per month: {', '.join(f'{MONTHS[m-1]} {int(n)}' for m, n in clipped.items())})",
        f"pool: {cfg['pool_levels_csv']}; source {cfg['source_label']}; f = {base['f']:g}; level {base['pct']} "
        f"({base['col']}); block lengths {cfg['blocks_h']} h; sites in the pool: {base.get('n_sites')}",
        f"prices: p_sub = {cfg['p_sub']:g} SEK/kW and month; overrun factor {cfg['overrun_factor']:g} "
        f"(= {cfg['overrun_factor'] * cfg['p_sub']:g} SEK per kW of excess); level per month; withdrawal only",
        f"excess e: {cfg['excess_list_kw']} kW (A1, same in every month); level S_m = P_m - e (A2); perfect foresight (A3)",
        "forecast-margin sensitivity: the pool is called in hours above S_m - margin and cuts them to that level "
        "(interpretation of 'also call in'); margin = share of the monthly peak",
        f"value basis: {cfg['value_basis']} ({VALUE_TXT[cfg['value_basis']]}: {value_factor(cfg):g} x p_sub per kW of reduction); "
        "flexibility price: not estimated (A6); break-even price = value / energy cut; reference prices "
        f"{cfg['reference_prices']}",
        f"rebound (A7): {'ON with assumed shares ' + str(cfg['rebound']['scenarios']) if cfg['rebound']['enabled'] else 'off'}",
        "offer lookup: block length min(L, 4) h, smallest available block >= that; missing slice = 0 kW offered",
        f"months without data: {[m for m, md in mds.items() if md.n == 0]}",
        "checks: " + "; ".join(checks),
    ]
    with open(os.path.join(cfg["output_dir"], "subscription_value_header.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(head) + "\n")

    t = monthly[(monthly["case"] == "base") & (monthly["e_kW"] == cfg["table_e_kw"])]
    print(t[["month", "S_m_kW", "hours_above", "E_MWh", "feasible", "d_kW", "value_SEK",
             "breakeven_SEK_per_MWh"]].round(1).to_string(index=False))
    print("\n".join(checks))
    print(f"[sub] outputs in {os.path.abspath(cfg['output_dir'])} (series is {proxy_flag}; table {os.path.basename(tex)})")


if __name__ == "__main__":
    main()

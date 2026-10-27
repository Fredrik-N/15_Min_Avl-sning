"""
thesis_tables.py -- every table and figure of the screening and 15-minute
chapters, written as LaTeX (booktabs) you can paste, plus PNG/PDF figures.
Replaces attrition_tables.py. It reads the output of the other scripts, so
nothing is typed in by hand and nothing is recomputed:

  lpc_11.py            site_comparison.csv, screening_out/attrition.csv
  filter_sensitivity.py  filter_sensitivity_out/ (selection frequency, winter coverage)
  robustness_value.py  (called here to redraw robustness_vs_value with the new size measure)
  reducible_power.py   reducible_power_out/reducible_power.csv
  offerable_flex.py    offerable_flex_out/offer_classification.csv

Run from the folder that holds them:   python thesis_tables.py

Site codes: lpc_11's CONFIG["site_code_key"] (the private key). Every site id
is turned into its code here; the script REFUSES to write a table that would
hold a real name (set allow_real_names for a private test run).

SCREENING AND ROBUSTNESS (chapter section 6.2)
  tab_attrition_sites.tex      per site: identified, matched, usable, passed, 1-s data, selected, reason
  tab_attrition_counts.tex     sites remaining after each step;  attrition_chain.tex ("18 -> 18 -> ...")
  attrition_sentence.tex       the method-chapter sentence with the counts
  tab_data_quality.tex         coverage (year, winter), PV, production data
  tab_screening_metrics.tex    winter metrics per site, failing values in bold, threshold row
  tab_negative_controls.tex    do the controls fail as they should (+ negative_controls_sentence.tex)
  selection_frequency.png/pdf  bars over the 81 threshold combinations;  selection_frequency_text.tex
  robustness_vs_value.png/pdf  (made by robustness_value.py with the same codes)
  winter_profiles.png/pdf      winter weekday / weekend profile: a cold store next to a failed site
  tab_final_group.tex          the selected sites: why, median above base by season, 1-s data
  final_group_share_sentence.tex   share of the segment's load above base the final group covers
15-MINUTE RESOURCE (section 6.3.1)
  tab_p15_by_season.tex        P_15 (median kW) per site, season and f
  tab_levels_by_season.tex     typical (P50) and firm (P10) per site and season
  tab_classification_sites.tex stand-alone / pool candidate / not relevant per site and season
  (the month x hour heatmaps of typical and firm level: figures.py)
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG = {
    "lpc_module": "lpc_11",
    "output_dir": "thesis_tables_out",
    "allow_real_names": False,

    # ---- inputs (None = the usual place) ----
    "attrition_csv": None,                       # lpc_11: <screening_out_dir>/attrition.csv
    "sensitivity_dir": "filter_sensitivity_out",
    "reducible_csv": os.path.join("reducible_power_out", "reducible_power.csv"),
    "offer_dir": "offerable_flex_out",
    "csv_sep": ";",                              # of the sensitivity / reducible / offer files
    "csv_decimal": ",",

    # ---- the choices for the 15-minute tables ----
    "f": 0.5,                                    # f shown in the levels / classification tables
    "block_h": 1,
    "f_values": [0.3, 0.5, 0.7, 1.0],            # columns of tab_p15_by_season

    # ---- by hand ----
    "reason_override": {},                       # {code: "reason text"} for the attrition table
    "step_override": {},                         # {code: {"identified":1,"matched":0,...}}
    "reason_selected_override": {},              # {code: "why selected"} for the final-group table
    "profile_sites": {"cold_store": None, "failed": None},   # codes; None = picked automatically
    "robust_min": 0.9,                           # selection frequency >= this: "robust"
    "segment": "passing",                        # the share sentence: "passing" (sites that passed the filter)
                                                 # or "usable" (all sites with data, controls excluded)
    "run_robustness_chart": True,
    "table_labels_prefix": "tab:res_",
    "filter_labels": {
        "coverage_fraction": "coverage",
        "load_factor_winter": "load factor",
        "weekend_to_weekday_ratio_winter": "weekend ratio",
        "temp_corr_winter_best_lag_value": "temperature correlation",
    },
}

YES, NO = r"\checkmark", r"--"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def truthy(x):
    if isinstance(x, pd.Series):
        return x.astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "yes", "y", "x"])
    return str(x).strip().lower() in ("1", "1.0", "true", "yes", "y", "x")


def esc(s):
    """LaTeX-safe text."""
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}"), ("<", r"$<$"), (">", r"$>$")]:
        s = s.replace(a, b)
    return s


def num(x, d=2, na="n/a"):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return na
    return na if pd.isna(v) else f"{v:.{d}f}"


def pct(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if pd.isna(v) else f"{100 * v:.0f}\\,\\%"


def tick(b):
    return YES if b else NO


def table(cfg, name, caption, label, colspec, header_rows, body, small="small", notes=None):
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\{small}", rf"\caption{{{caption}}}",
             rf"\label{{{cfg['table_labels_prefix']}{label}}}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    lines += header_rows + [r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines += [r"\par\vspace{2pt}{\footnotesize " + notes + "}"]
    lines += [r"\end{table}"]
    write(cfg, name, "\n".join(lines))


def write(cfg, name, text):
    with open(os.path.join(cfg["output_dir"], name), "w", encoding="utf-8") as fh:
        fh.write(text.rstrip("\n") + "\n")
    print(f"  wrote {name}")


def rd(path, cfg):
    return pd.read_csv(path, sep=cfg["csv_sep"], decimal=cfg["csv_decimal"], encoding="utf-8-sig")


def save_fig(fig, cfg, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(cfg["output_dir"], f"{stem}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png/.pdf")


def fail_text(raw, cfg):
    if pd.isna(raw) or str(raw).strip() == "":
        return ""
    txt = str(raw)
    found = [lab for col, lab in cfg["filter_labels"].items() if col in txt]
    return ("fails " + ", ".join(found)) if found else txt


# ----------------------------------------------------------------------
# loading and coding
# ----------------------------------------------------------------------
def load_all(cfg):
    sys.path.insert(0, os.getcwd())
    L = __import__(cfg["lpc_module"])
    lc = L.CONFIG
    key = L._code_map(lc)
    codes = set(key.values())
    if not key and not cfg["allow_real_names"]:
        sys.exit("lpc_11 CONFIG['site_code_key'] is not set: the tables would hold real site names. "
                 "Set it to your private key file (or allow_real_names = True for a private test).")

    def to_code(x):
        x = str(x).strip()
        if x.startswith("UNCODED:"):
            x = x[len("UNCODED:"):]
        if x in codes:
            return x
        c = key.get(x.lower())
        if c is not None:
            return c
        return x if cfg["allow_real_names"] else f"UNCODED:{x}"
    comp = L.read_comparison(lc)
    if comp is None:
        sys.exit("No site_comparison.csv: run lpc_11.py --rerun-all first.")
    comp = comp.copy()
    comp["code"] = comp[L.SITE_KEY].map(to_code)
    if "is_negative_control" not in comp:
        comp["is_negative_control"] = False
    comp["is_negative_control"] = truthy(comp["is_negative_control"])
    for c in ("filter_pass", "selected", "pv_flag"):
        if c in comp:
            comp[c] = truthy(comp[c])
    bad = comp["code"].str.startswith("UNCODED:")
    if bad.any():
        sys.exit(f"{int(bad.sum())} site(s) in site_comparison.csv have no code in the key file: add them to it.")
    return L, lc, comp, to_code


def check_coded(cfg, series, where):
    """Stop (without printing real names) if a site has no code."""
    bad = series.astype(str).str.startswith("UNCODED:")
    if bad.any():
        sys.exit(f"{int(bad.sum())} site(s) in {where} have no code in the key file (site_code_key): "
                 f"add them to it (sites identified but never run included), then run again.")


# ----------------------------------------------------------------------
# 1. attrition
# ----------------------------------------------------------------------
def attrition(cfg, L, lc, comp, to_code):
    path = cfg["attrition_csv"] or os.path.join(lc.get("screening_out_dir") or "screening_out", "attrition.csv")
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run `python lpc_11.py --rank-only` first.")
    a = pd.read_csv(path, sep=lc.get("results_csv_sep", ";"), encoding="utf-8-sig")
    a = a[a["site"].astype(str) != "TOTAL"].copy()
    a["code"] = a["site"].map(to_code)
    check_coded(cfg, a["code"], "attrition.csv")
    ctrl = set(comp.loc[comp["is_negative_control"], "code"])
    a = a[~a["code"].isin(ctrl)]                       # negative controls are not candidates
    cols = {"identified": "s_ident", "matched": "s_match", "usable_data": "s_usable", "passed_filter": "s_pass",
            "one_second_data_granted": "s_1s", "selected": "s_sel"}
    for k, v in cols.items():
        a[v] = truthy(a[k])
    for code, ov in cfg["step_override"].items():
        for k, v in ov.items():
            a.loc[a["code"] == code, cols.get(k, k)] = bool(v)
    c = comp.set_index("code")

    def reason(r):
        if r["code"] in cfg["reason_override"]:
            return cfg["reason_override"][r["code"]]
        if not r["s_match"]:
            return "not matched to a customer / no meter file"
        if not r["s_usable"]:
            return "no usable meter data"
        row = c.loc[r["code"]] if r["code"] in c.index else None
        if row is None:
            return ""
        note = str(row.get("selection_note") or "")
        if "EXCEPTION" in note and r["s_sel"]:
            return "kept as contrasting case; " + fail_text(row.get("filter_fail_reasons"), cfg)
        if not r["s_pass"]:
            return fail_text(row.get("filter_fail_reasons"), cfg) or "failed the filter"
        if "deselected by hand" in note:
            return "passed; left out by hand (" + note.split("deselected by hand:")[-1].strip(" ;") + ")"
        if not r["s_sel"]:
            return "trimmed (small contribution)" if "trim" in note.lower() else "passed, not selected"
        if "selected by hand" in note:
            return "passed; " + note.split("selected by hand:")[-1].strip(" ;")
        return "passed the filter"
    a["reason_out"] = a.apply(reason, axis=1)
    a = a.sort_values("code")
    rows = [f"{esc(r.code)} & {tick(r.s_ident)} & {tick(r.s_match)} & {tick(r.s_usable)} & {tick(r.s_pass)} & "
            f"{tick(r.s_1s)} & {tick(r.s_sel)} & {esc(r.reason_out)} \\\\" for r in a.itertuples()]
    table(cfg, "tab_attrition_sites.tex",
          r"Attrition of candidate sites in the 15-minute screening. \checkmark{} = the site remains after the "
          r"step. The filter is applied to winter data (November--February) with the thresholds in "
          r"Table~\ref{tab:m_thresholds}.",
          "attrition_sites", r"l c c c c c c p{0.34\textwidth}",
          [r"Site & Identified & Matched & Usable & Passed & 1-s data & Selected & Reason \\"], rows, "footnotesize")
    n = {k: int(a[k].sum()) for k in cols.values()}
    labels = [("Identified as candidates", "s_ident"), ("Matched to a customer", "s_match"),
              ("Usable meter data", "s_usable"), ("Passed the filter", "s_pass"),
              ("1-second data granted", "s_1s"), ("Selected as case sites", "s_sel")]
    table(cfg, "tab_attrition_counts.tex", "Number of sites remaining after each step of the screening.",
          "attrition_counts", "lr", [r"Step & Sites \\"], [f"{lab} & {n[k]} \\\\" for lab, k in labels])
    chain = " $\\rightarrow$ ".join(str(n[k]) for _, k in labels)
    write(cfg, "attrition_chain.tex", chain)
    n_exc = int((a["s_sel"] & ~a["s_pass"]).sum())
    exc_txt = (f", including {n_exc} that failed the filter but was kept as a contrasting case" if n_exc == 1 else
               f", including {n_exc} kept as contrasting cases" if n_exc > 1 else "")
    sent = (f"Of the {n['s_ident']} candidates identified, {n['s_match']} were matched to a customer, "
            f"{n['s_usable']} had usable data, {n['s_pass']} passed the filter, and {n['s_sel']} were selected "
            f"as case sites{exc_txt}. {n['s_1s']} of the sites provided 1-second data.")
    write(cfg, "attrition_sentence.tex", sent)
    print("  chain (table order):", " -> ".join(str(n[k]) for _, k in labels))
    return a, n


# ----------------------------------------------------------------------
# 2. data quality and screening metrics
# ----------------------------------------------------------------------
def winter_coverage(cfg, to_code):
    p = os.path.join(cfg["sensitivity_dir"], "filter_sensitivity_metrics.csv")
    if not os.path.exists(p):
        return {}
    m = rd(p, cfg)
    if "coverage_winter" not in m:
        return {}
    return dict(zip(m["site_id"].map(to_code), pd.to_numeric(m["coverage_winter"], errors="coerce")))


def data_quality(cfg, comp, to_code, attr):
    wc = winter_coverage(cfg, to_code)
    use = set(attr.loc[attr["s_usable"], "code"])
    rows = []
    for r in comp[comp["code"].isin(use) & ~comp["is_negative_control"]].sort_values("code").itertuples():
        has_prod = isinstance(getattr(r, "production_files", None), str) and r.production_files.strip() != ""
        gross = truthy(getattr(r, "gross_available", False))
        pv = bool(r.pv_flag)
        if gross:
            prod = "yes: gross load"
        elif has_prod:
            prod = "export file: net load"
        else:
            prod = "none: net load only" if pv else "--"
        rows.append(f"{esc(r.code)} & {pct(r.coverage_fraction)} & {pct(wc.get(r.code))} & "
                    f"{'yes' if pv else 'no'} & {prod} \\\\")
    table(cfg, "tab_data_quality.tex",
          "Data quality per site: share of 15-minute intervals with valid data over the whole period and over "
          "the winter months, whether solar production behind the meter was detected, and whether production "
          "data were available (gross load) or only the net exchange.",
          "data_quality", "lcccl", [r"Site & Coverage, year & Coverage, winter & PV & Production data \\"], rows)


def screening_metrics(cfg, lc, comp):
    sel = lc["selection"]
    flt = {k: v for k, v in sel["filters"].items() if v.get("enabled", True)}
    dec = sel.get("compare_decimals")
    cols = [("coverage_fraction", "Coverage"), ("load_factor_winter", "Load factor"),
            ("weekend_to_weekday_ratio_winter", "Weekend ratio"), ("temp_corr_winter_best_lag_value", "Temp. corr.")]

    def passes(col, v):
        s = flt.get(col)
        if s is None:
            return True
        try:
            v = float(v)
        except (TypeError, ValueError):
            return not sel.get("missing_value_fails", True)
        if pd.isna(v):
            return not sel.get("missing_value_fails", True)
        if dec is not None:
            v = round(v, dec)
        return (s.get("min") is None or v >= s["min"]) and (s.get("max") is None or v <= s["max"])

    def cell(col, v):
        t = num(v, 2)
        return t if passes(col, v) else rf"\textbf{{{t}}}"
    thr = []
    for col, _ in cols:
        s = flt.get(col)
        thr.append("--" if s is None else
                   " ".join(([rf"$\geq {s['min']:g}$"] if s.get("min") is not None else [])
                            + ([rf"$\leq {s['max']:g}$"] if s.get("max") is not None else [])))
    head = [r"Site & " + " & ".join(c for _, c in cols) + r" & Base load (kW) & Median above base (kW) & Result \\",
            r"Threshold & " + " & ".join(thr) + r" & & & \\"]

    def row(r):
        note = str(getattr(r, "selection_note", "") or "")
        res = ("kept (contrasting case)" if "EXCEPTION" in note else
               "pass" if r.filter_pass else "fail")
        cells = " & ".join(cell(col, getattr(r, col, np.nan)) for col, _ in cols)
        return (f"{esc(r.code)} & {cells} & {num(getattr(r, 'base_load_winter_kw', np.nan), 1)} & "
                f"{num(getattr(r, 'median_above_base_winter_kw', np.nan), 1)} & {res} \\\\")
    real = comp[~comp["is_negative_control"]].sort_values(["filter_pass", "code"], ascending=[False, True])
    body = [row(r) for r in real.itertuples()]
    ctl = comp[comp["is_negative_control"]].sort_values("code")
    if len(ctl):
        body += [r"\midrule", r"\multicolumn{8}{l}{\textit{Negative controls}} \\"] + [row(r) for r in ctl.itertuples()]
    table(cfg, "tab_screening_metrics.tex",
          "Screening metrics per site, all on winter data (November--February). Values that fail their threshold "
          "are in bold. Base load is the 5th percentile of the winter load; median above base is the median over "
          "winter hours of the hourly load above that base.",
          "screening_metrics", "lccccrrl", head, body)

    # negative controls
    if not len(ctl):
        print("  no negative controls in site_comparison.csv (SELECTION['negative_controls']): "
              "tab_negative_controls.tex not written")
        return
    rows, nfail = [], 0
    for r in ctl.itertuples():
        failed = [lab for col, lab in cfg["filter_labels"].items() if col in flt and not passes(col, getattr(r, col, np.nan))]
        ok = len(failed) > 0
        nfail += ok
        rows.append(f"{esc(r.code)} & {esc(', '.join(failed)) if failed else '--'} & "
                    f"{'fails, as it should' if ok else 'PASSES the filter'} \\\\")
    table(cfg, "tab_negative_controls.tex",
          "Negative controls (sites that are not cold storage) run through the same winter filter.",
          "negative_controls", "lll", [r"Control & Filters failed & Outcome \\"], rows)
    write(cfg, "negative_controls_sentence.tex",
          f"{nfail} of the {len(ctl)} negative controls failed the filter" +
          ("." if nfail == len(ctl) else f"; {len(ctl) - nfail} passed, so the filter does not separate them from cold storage."))


# ----------------------------------------------------------------------
# 3. selection frequency, robustness chart, profiles
# ----------------------------------------------------------------------
def selection_frequency(cfg, comp, to_code):
    p = os.path.join(cfg["sensitivity_dir"], "filter_factorial_selection_frequency.csv")
    if not os.path.exists(p):
        print(f"  {p} not found: run filter_sensitivity.py -- selection frequency figure skipped")
        return
    f = rd(p, cfg)
    f["code"] = f["site_id"].map(to_code)
    f["selection_frequency"] = pd.to_numeric(f["selection_frequency"], errors="coerce")
    st = comp.set_index("code")
    f["selected"] = f["code"].map(st["selected"]).fillna(False).astype(bool)
    f["exception"] = f["code"].map(lambda c: "EXCEPTION" in str(st["selection_note"].get(c, "")))
    f = f[~f["code"].isin(set(comp.loc[comp["is_negative_control"], "code"]))]    # controls are not candidates
    f = f.sort_values(["selection_frequency", "code"], ascending=[False, True])
    n_runs = int(f["n_runs"].iloc[0]) if "n_runs" in f else 81
    fig, ax = plt.subplots(figsize=(max(5.5, 0.38 * len(f) + 2), 3.8))
    for i, r in enumerate(f.itertuples()):
        ax.bar(i, r.selection_frequency, color="0.15" if r.selected and not r.exception else
               "white" if r.exception else "0.78", edgecolor="0.1", hatch="//" if r.exception else None, lw=0.8)
    ax.set_xticks(range(len(f)))
    ax.set_xticklabels(f["code"], rotation=90, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"selection frequency (share of {n_runs} threshold combinations)")
    ax.axhline(cfg["robust_min"], color="k", ls="--", lw=0.8)
    ax.text(len(f) - 0.5, cfg["robust_min"] + 0.01, f"robust: {cfg['robust_min']:.0%}", ha="right", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="0.15", ec="0.1", label="selected"),
                       Patch(fc="white", ec="0.1", hatch="//", label="kept by hand (fails the filter)"),
                       Patch(fc="0.78", ec="0.1", label="not selected")], frameon=False, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, cfg, "selection_frequency")
    ok = f[~f["exception"]]
    robust = ok[ok["selection_frequency"] >= cfg["robust_min"]]["code"].tolist()
    border = ok[(ok["selection_frequency"] > 0) & (ok["selection_frequency"] < cfg["robust_min"])]
    never = ok[ok["selection_frequency"] == 0]["code"].tolist()
    txt = (f"Of the {len(ok)} sites, {len(robust)} passed in at least {cfg['robust_min']:.0%} of the {n_runs} "
           f"threshold combinations ({', '.join(robust) if robust else 'none'}); {len(border)} passed in some but "
           f"not nearly all ({', '.join(f'{c} {v:.0%}' for c, v in zip(border['code'], border['selection_frequency'])) or 'none'}); "
           f"{len(never)} never passed.")
    write(cfg, "selection_frequency_text.tex", txt)
    print("  " + txt)


def robustness_chart(cfg, lc):
    if not cfg["run_robustness_chart"]:
        return
    try:
        import robustness_value as R
        c = dict(R.CONFIG, code_key=lc.get("site_code_key"), anonymise=True,
                 sensitivity_dir=cfg["sensitivity_dir"])
        if not c["code_key"]:
            c["code_key"] = None
        R.main(c, False)
        for ext in ("png", "pdf"):
            src = os.path.join(c["output_dir"], f"robustness_vs_value.{ext}")
            if os.path.exists(src):
                import shutil
                shutil.copy(src, os.path.join(cfg["output_dir"], f"robustness_vs_value.{ext}"))
        print("  copied robustness_vs_value.png/.pdf from robustness_value.py")
    except SystemExit as e:
        print(f"  robustness_vs_value skipped: {e}")
    except Exception as e:
        print(f"  robustness_vs_value skipped: {type(e).__name__}: {e}")


def winter_profiles(cfg, L, lc, comp):
    ps = cfg["profile_sites"]
    ok = comp[~comp["is_negative_control"]]
    cold, bad = ps.get("cold_store"), ps.get("failed")
    if cold is None:
        c = ok[ok["selected"] & ok["filter_pass"]].sort_values("median_above_base_year_kw", ascending=False)
        cold = c["code"].iloc[0] if len(c) else None
    if bad is None:
        b = ok[~ok["filter_pass"]].copy()          # prefer a site that was not kept by hand
        b["n_fail"] = b["filter_fail_reasons"].fillna("").astype(str).str.count(";") + (b["filter_fail_reasons"].fillna("") != "")
        b = b.sort_values(["selected", "n_fail"], ascending=[True, False])
        bad = b["code"].iloc[0] if len(b) else None
    if cold is None or bad is None:
        print("  winter_profiles skipped: need one passing and one failing site (or set profile_sites)")
        return
    months = lc.get("shape_months") or [11, 12, 1, 2]
    we_days = lc.get("weekend_days", [5, 6])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, code, kind in zip(axs, (cold, bad), ("passes the filter", "fails the filter")):
        r = comp[comp["code"] == code].iloc[0]
        path = L.find_meter_file(r, lc)
        if path is None:
            ax.set_title(f"Site {code}: meter file not found")
            continue
        c = dict(lc, csv_path=path, show_plot=False)
        net, _ = L.load_net_series(c)
        g = L.gross_series(net, c)
        x = (g if g is not None else net).dropna()
        w = x[x.index.month.isin(months)]
        tod = w.index.hour + w.index.minute / 60
        we = np.isin(w.index.dayofweek, we_days)
        for mask, ls, lab in ((~we, "-", "weekday"), (we, "--", "weekend")):
            s = w[mask]
            t = pd.Series(s.to_numpy()).groupby(np.asarray(tod)[mask]).agg(["mean", lambda v: v.quantile(.25),
                                                                            lambda v: v.quantile(.75)])
            t.columns = ["mean", "q25", "q75"]
            ax.plot(t.index, t["mean"], color="k", ls=ls, lw=1.6, label=f"{lab} mean")
            if lab == "weekday":
                ax.fill_between(t.index, t["q25"], t["q75"], color="0.8", label="weekday P25--P75")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlabel("hour of day")
        ax.set_ylabel("kW" + (" (gross)" if g is not None else " (net exchange)"))
        ax.set_title(f"Site {code}: {kind}", fontsize=10)
        ax.grid(alpha=0.3)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Typical winter load profile (November--February)", fontsize=11)
    save_fig(fig, cfg, "winter_profiles")


# ----------------------------------------------------------------------
# 4. final group
# ----------------------------------------------------------------------
def final_group(cfg, comp, attr):
    ok = comp[~comp["is_negative_control"]]
    fin = ok[ok["selected"]].sort_values("median_above_base_year_kw", ascending=False)
    one_s = dict(zip(attr["code"], attr["s_1s"]))

    def why(r):
        if r.code in cfg["reason_selected_override"]:
            return cfg["reason_selected_override"][r.code]
        note = str(r.selection_note or "")
        m = re.search(r"EXCEPTION[^:]*:\s*(.*?)(?:\s\[|;|$)", note)
        h = re.search(r"selected by hand:\s*(.*?)(?:;|$)", note)
        if "EXCEPTION" in note:
            return "fails the filter; kept by hand" + (f": {m.group(1)}" if m and m.group(1) else "")
        base = "passed the filter"
        return base + (f"; selected by hand: {h.group(1)}" if h and h.group(1) else "")
    rows = []
    for r in fin.itertuples():
        v = lambda s: num(getattr(r, f"median_above_base_{s}_kw", np.nan), 1)
        rows.append(f"{esc(r.code)} & {esc(why(r))} & {v('winter')} & {v('shoulder')} & {v('summer')} & "
                    f"{v('year')} & {'yes' if one_s.get(r.code) else 'no'} \\\\")
    table(cfg, "tab_final_group.tex",
          "The selected sites: why each was selected, its median load above base per season (kW) and whether "
          "1-second data were granted.",
          "final_group", r"l p{0.30\textwidth} rrrr c",
          [r"Site & Selected because & Winter & Shoulder & Summer & Year & 1-s data \\"], rows)
    # share of the segment's load above base
    val = "median_above_base_year_kw"
    seg = ok[ok["filter_pass"]] if cfg["segment"] == "passing" else ok[ok[val].notna()]
    seg_sum = pd.to_numeric(seg[val], errors="coerce").sum()
    inn = seg[seg["selected"]]
    inn_sum = pd.to_numeric(inn[val], errors="coerce").sum()
    outside = fin[~fin["code"].isin(seg["code"])]
    segname = "the sites that passed the filter" if cfg["segment"] == "passing" else "all sites with usable data"
    s = (f"The final group of {len(fin)} sites accounts for {100 * inn_sum / seg_sum:.0f}\\,\\% of the median load above "
         f"base of {segname} ({inn_sum:.0f} of {seg_sum:.0f}\\,kW, year)" if seg_sum > 0 else "n/a")
    if len(outside) and seg_sum > 0:
        s += (f"; {len(outside)} site{'s' if len(outside) > 1 else ''} kept by hand ("
              f"{', '.join(outside['code'])}) {'are' if len(outside) > 1 else 'is'} not part of that share")
    write(cfg, "final_group_share_sentence.tex", s + ".")
    print("  " + s)


# ----------------------------------------------------------------------
# 5. the 15-minute resource
# ----------------------------------------------------------------------
SEASONS = ["winter", "shoulder", "summer"]


def p15_tables(cfg, to_code):
    if os.path.exists(cfg["reducible_csv"]):
        d = rd(cfg["reducible_csv"], cfg)
        d = d[d["source"] == "15 min"].copy()
        d["code"] = d["site"].map(to_code)
        check_coded(cfg, d["code"], "reducible_power.csv")
        d["f"] = d["f"].astype(float)
        m = d.groupby(["season", "code", "f"])["kW"].median().unstack("f")
        body = []
        for season in SEASONS:
            if season not in m.index.get_level_values(0):
                continue
            body.append(rf"\multicolumn{{{1 + len(cfg['f_values'])}}}{{l}}{{\textit{{{season.capitalize()}}}}} \\")
            for code in sorted(m.loc[season].index):
                body.append(esc(code) + " & " + " & ".join(
                    num(m.loc[(season, code)].get(f, np.nan), 1) for f in cfg["f_values"]) + r" \\")
            body.append(r"\addlinespace")
        table(cfg, "tab_p15_by_season.tex",
              r"Reducible power $P_{15}$ per site and season (median over hours, kW), for each share $f$ of the "
              r"load above base that can be moved.",
              "p15_by_season", "l" + "r" * len(cfg["f_values"]),
              [r"Site & " + " & ".join(rf"$f={f:g}$" for f in cfg["f_values"]) + r" \\"], body[:-1])
    else:
        print(f"  {cfg['reducible_csv']} not found: tab_p15_by_season skipped")
    p = os.path.join(cfg["offer_dir"], "offer_classification.csv")
    if not os.path.exists(p):
        print(f"  {p} not found: levels and classification tables skipped")
        return
    c = rd(p, cfg)
    c["code"] = c["site"].map(to_code)
    check_coded(cfg, c["code"], "offer_classification.csv")
    c = c[(c["block_h"] == cfg["block_h"]) & (c["f"].isna() | np.isclose(c["f"].astype(float).fillna(-1), cfg["f"]))]
    seasons = SEASONS + ["year"]
    sites = sorted(c["code"].unique())
    rows, rows_c = [], []
    ab = {"stand-alone": "stand-alone", "pool candidate": "pool candidate", "not relevant": "not relevant", "n/a": "--"}
    for s in sites:
        cells, cc = [], []
        for se in seasons:
            r = c[(c["code"] == s) & (c["season"] == se)]
            if len(r):
                r = r.iloc[0]
                cells += [num(r["typical_P50_kw"], 1), num(r["firm_P10_kw"], 1)]
                cc.append(ab.get(r["class"], "--") if isinstance(r["class"], str) else "--")
            else:
                cells += ["--", "--"]
                cc.append("--")
        rows.append(esc(s) + " & " + " & ".join(cells) + r" \\")
        rows_c.append(esc(s) + " & " + " & ".join(cc) + r" \\")
    head = [r" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{se.capitalize()}}}" for se in seasons) + r" \\",
            "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(seasons))),
            r"Site & " + " & ".join(["Typical", "Firm"] * len(seasons)) + r" \\"]
    table(cfg, "tab_levels_by_season.tex",
          rf"Typical (P50) and firm (P10) offerable power per site and season (kW), {cfg['block_h']}-hour blocks; "
          rf"15-minute sites at $f={cfg['f']:g}$.",
          "levels_by_season", "l" + "rr" * len(seasons), head, rows)
    table(cfg, "tab_classification_sites.tex",
          rf"Classification of each site by season ({cfg['block_h']}-hour blocks, firm level P10; 15-minute sites at "
          rf"$f={cfg['f']:g}$): stand-alone ($\geq 100$\,kW), pool candidate, not relevant.",
          "classification_sites", "l" + "l" * len(seasons),
          [r"Site & " + " & ".join(se.capitalize() for se in seasons) + r" \\"], rows_c)


# ----------------------------------------------------------------------
def main(cfg=CONFIG):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    L, lc, comp, to_code = load_all(cfg)
    print("[screening tables]")
    attr, n = attrition(cfg, L, lc, comp, to_code)
    data_quality(cfg, comp, to_code, attr)
    screening_metrics(cfg, lc, comp)
    print("[figures]")
    selection_frequency(cfg, comp, to_code)
    robustness_chart(cfg, lc)
    winter_profiles(cfg, L, lc, comp)
    print("[final group]")
    final_group(cfg, comp, attr)
    print("[15-minute resource]")
    p15_tables(cfg, to_code)
    print(f"[tables] outputs in {os.path.abspath(cfg['output_dir'])}")


if __name__ == "__main__":
    main()

"""
attrition_tables.py -- LaTeX tables for the 15-minute screening results (section 6.2.1).

Writes three tables from the screening output, so every tick in the thesis
comes from the data and not from hand-editing:

  tab_attrition_sites.tex   one row per site: identified, matched, usable data,
                            passed filter, 1-s data granted, selected, reason
  tab_attrition_counts.tex  number of sites remaining after each step
  tab_data_quality.tex      coverage (whole span and winter), PV, production data
  attrition_sentence.tex    the sentence for the method chapter, with the counts

Inputs (run from the folder that holds them):

  site_comparison.csv  written by lpc_10.py. Used columns:
                       site_id, coverage_fraction, filter_pass,
                       filter_fail_reasons, selected, selection_note,
                       pv_flag, production_files
                       (coverage_winter is taken from
                        filter_sensitivity_out/filter_sensitivity_metrics.csv
                        if that file exists)

  site_steps.csv       filled in by hand -- the steps the meter data cannot know.
                       Columns:
                         site_id            as in site_comparison.csv
                         code               anonymised code used in the thesis (A, B, ...)
                         identified         1/0
                         matched            1/0  (matched to an Öresundskraft customer)
                         one_second         1/0  (1-second data granted)
                         reason             short English text for the reason column
                                            (leave empty to use the filter reasons)
                       A template is written on the first run if the file is missing.

Nothing is sent anywhere. The site_id -> code key stays in site_steps.csv,
which must not go into the thesis.
"""

import os
import sys
import pandas as pd

CONFIG = {
    "comparison_csv": "site_comparison.csv",
    "metrics_csv": os.path.join("filter_sensitivity_out", "filter_sensitivity_metrics.csv"),
    "steps_csv": "site_steps.csv",
    "csv_sep": ",",          # set to ";" and decimal "," if lpc_10 writes Swedish CSV
    "csv_decimal": ".",
    "output_dir": "attrition_out",
    # Readable names for the filter columns, used in the reason column
    "filter_labels": {
        "coverage_fraction": "coverage",
        "load_factor_winter": "load factor",
        "weekend_to_weekday_ratio_winter": "weekend ratio",
        "temp_corr_winter_best_lag_value": "temperature correlation",
    },
}

YES, NO = r"\checkmark", r"--"


def read(path, cfg):
    return pd.read_csv(path, sep=cfg["csv_sep"], decimal=cfg["csv_decimal"], encoding="utf-8-sig")


def truthy(s):
    return s.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y", "x"])


def norm(s):
    return s.astype(str).str.strip().str.lower()


def write_template(comp, cfg):
    t = pd.DataFrame({
        "site_id": comp["site_id"],
        "code": [chr(ord("A") + i) if i < 26 else f"S{i+1}" for i in range(len(comp))],
        "identified": 1,
        "matched": 1,
        "one_second": 0,
        "reason": "",
    })
    t.to_csv(cfg["steps_csv"], index=False, encoding="utf-8-sig")
    sys.exit(f"Wrote a template {cfg['steps_csv']!r}. Fill in code, matched, one_second and "
             f"reason, then run again. Sites that were identified but never matched must be added "
             f"as extra rows by hand (they are not in site_comparison.csv).")


def fail_text(raw, cfg):
    """Turn lpc_10's filter_fail_reasons into a short English list."""
    if pd.isna(raw) or str(raw).strip() == "":
        return ""
    txt = str(raw)
    found = [lab for col, lab in cfg["filter_labels"].items() if col in txt]
    return "fails " + ", ".join(found) if found else txt


def main(cfg=CONFIG):
    if not os.path.exists(cfg["comparison_csv"]):
        sys.exit(f"No {cfg['comparison_csv']!r} here. Run from the folder with lpc_10.py's output.")
    comp = read(cfg["comparison_csv"], cfg)
    if not os.path.exists(cfg["steps_csv"]):
        write_template(comp, cfg)
    steps = pd.read_csv(cfg["steps_csv"], encoding="utf-8-sig", dtype={"code": str})

    comp["_k"] = norm(comp["site_id"])
    steps["_k"] = norm(steps["site_id"])
    df = steps.merge(comp, on="_k", how="left", suffixes=("", "_c"))

    if os.path.exists(cfg["metrics_csv"]):
        m = read(cfg["metrics_csv"], cfg)
        if "coverage_winter" in m.columns:
            m["_k"] = norm(m["site_id"])
            df = df.merge(m[["_k", "coverage_winter"]], on="_k", how="left")

    # ---- steps --------------------------------------------------------
    df["s_ident"] = truthy(df["identified"])
    df["s_match"] = df["s_ident"] & truthy(df["matched"])
    df["s_usable"] = df["s_match"] & df["coverage_fraction"].notna()
    df["s_pass"] = df["s_usable"] & truthy(df["filter_pass"].fillna(False))
    df["s_1s"] = df["s_usable"] & truthy(df["one_second"])
    df["s_sel"] = df["s_usable"] & truthy(df["selected"].fillna(False))
    df["s_exception"] = df["s_sel"] & ~df["s_pass"]

    # ---- reason column --------------------------------------------------
    def reason(r):
        if isinstance(r["reason"], str) and r["reason"].strip():
            return r["reason"].strip()
        if not r["s_match"]:
            return "not matched to a customer"
        if not r["s_usable"]:
            return "no usable meter data"
        if r["s_exception"]:
            return "contrasting case; " + fail_text(r["filter_fail_reasons"], cfg)
        if not r["s_pass"]:
            return fail_text(r["filter_fail_reasons"], cfg)
        if not r["s_sel"]:
            note = str(r.get("selection_note", "") or "")
            return "trimmed (small contribution)" if "trim" in note.lower() else "passed, not selected"
        return "passed filter"

    df["reason_out"] = df.apply(reason, axis=1)
    df = df.sort_values("code")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    tick = lambda b: YES if b else NO

    # ---- table 1: per site ----------------------------------------------
    rows = [f"{r.code} & {tick(r.s_ident)} & {tick(r.s_match)} & {tick(r.s_usable)} & "
            f"{tick(r.s_pass)} & {tick(r.s_1s)} & {tick(r.s_sel)} & {r.reason_out} \\\\"
            for r in df.itertuples()]
    t1 = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Attrition of candidate sites in the 15-minute screening. "
        r"\checkmark{} = the site remains after the step. The filter is applied to winter data "
        r"(November--February) with the thresholds in Table~\ref{tab:m_thresholds}.}",
        r"\label{tab:res_attrition_sites}",
        r"\begin{tabularx}{\textwidth}{l c c c c c c >{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"Site & Ident. & Matched & Usable & Passed & 1-s data & Selected & Reason \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
    ])

    # ---- table 2: counts -------------------------------------------------
    n = {k: int(df[k].sum()) for k in ["s_ident", "s_match", "s_usable", "s_pass", "s_1s", "s_sel"]}
    n_exc = int(df["s_exception"].sum())
    labels = [("Identified as candidates", "s_ident"),
              ("Matched to a customer", "s_match"),
              ("Usable meter data", "s_usable"),
              ("Passed the filter", "s_pass"),
              ("Selected as case sites", "s_sel"),
              ("1-second data granted", "s_1s")]
    rows2 = [f"{lab} & {n[k]} \\\\" for lab, k in labels]
    t2 = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Number of sites remaining after each step of the screening.}",
        r"\label{tab:res_attrition_counts}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Step & Sites \\",
        r"\midrule",
        *rows2,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    # note: selected may include exceptions that did not pass the filter
    sentence = (f"Of the {n['s_ident']} candidates identified, {n['s_match']} were matched to a customer, "
                f"{n['s_usable']} had usable data, {n['s_pass']} passed the filter, and {n['s_sel']} were "
                f"selected as case sites"
                + (f", including {n_exc} that failed the filter but was kept as a contrasting case"
                   if n_exc == 1 else f", including {n_exc} kept as contrasting cases" if n_exc > 1 else "")
                + f". {n['s_1s']} of the sites provided 1-second data.")

    # ---- table 3: data quality --------------------------------------------
    def pct(x):
        return "--" if pd.isna(x) else f"{100 * float(x):.0f}\\,\\%"

    def prod(r):
        has = isinstance(r.production_files, str) and r.production_files.strip() != ""
        pv = truthy(pd.Series([r.pv_flag]))[0] if not pd.isna(r.pv_flag) else False
        if has:
            return "yes (gross load reconstructed)"
        return "no (net load only)" if pv else "--"

    has_w = "coverage_winter" in df.columns
    rows3 = []
    for r in df[df["s_usable"]].itertuples():
        pv = truthy(pd.Series([r.pv_flag]))[0] if not pd.isna(r.pv_flag) else False
        cov_w = pct(r.coverage_winter) if has_w else "--"
        rows3.append(f"{r.code} & {pct(r.coverage_fraction)} & {cov_w} & {'yes' if pv else 'no'} & {prod(r)} \\\\")
    t3 = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Data quality per site: share of 15-minute intervals with valid data over the "
        r"whole period and over the winter months, whether solar production behind the meter was "
        r"detected, and whether production data were available to reconstruct the gross load.}",
        r"\label{tab:res_data_quality}",
        r"\begin{tabular}{lcccl}",
        r"\toprule",
        r"Site & Coverage, year & Coverage, winter & PV & Production data \\",
        r"\midrule",
        *rows3,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    for name, txt in [("tab_attrition_sites.tex", t1), ("tab_attrition_counts.tex", t2),
                      ("tab_data_quality.tex", t3), ("attrition_sentence.tex", sentence)]:
        with open(os.path.join(cfg["output_dir"], name), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")

    print(" -> ".join(str(n[k]) for _, k in labels[:5]), f"(1-s data: {n['s_1s']})")
    print(sentence)
    if n_exc:
        print("Exceptions (selected although failing the filter):",
              ", ".join(df.loc[df["s_exception"], "code"]))
    missing = df[df["s_match"] & df["coverage_fraction"].isna()]
    if len(missing):
        print("Matched but not in site_comparison.csv (counted as no usable data):",
              ", ".join(missing["code"]))
    print(f"Tables written to {cfg['output_dir']}/")


if __name__ == "__main__":
    main()

"""
Sensitivity test of the 15-min screening filters.

Works only on the per-site metrics already in site_comparison.csv (written
by lpc_10.py). The metrics are read ONCE, saved to
filter_sensitivity_metrics.csv, and every set of thresholds is applied to
that one table -- nothing is recomputed between runs. No randomness.

The BASELINE is read from lpc_10.py's SELECTION (thresholds, upper
bounds, decimals, rank_by, trim rule, exceptions), so the two scripts
cannot disagree. Only the TESTED values are set here (CONFIG["filters"]).

Step 0  lpc_10's own selection code is run on site_comparison.csv and
        compared with (a) the filter_pass / selected columns stored in the
        file and (b) this script's re-implementation. Any difference is
        printed and the script STOPS.
A       One-at-a-time sweep (one threshold moved, the others at baseline).
B       Full grid, every combination (3 x 3 x 3 x 3 = 81): selection
        frequency per site.
C       Margins: value, threshold, margin per site and filter; binding
        filter (failing sites) / tightest filter (passing sites).
D       Tail trim under several parameterised rules.
E       The exception site(s): values, margins, how many grid runs they
        would pass, cumulative share with and without them. An exception is
        NEVER counted as passing a filter.

Run from the folder with site_comparison.csv:
    python filter_sensitivity.py
    python filter_sensitivity.py --skip-baseline-check   (only to look at
        the outputs after Step 0 has told you why it differs)
Outputs go to CONFIG["output_dir"].
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

# ======================================================================
# CONFIG -- everything that can be changed is here
# ======================================================================
CONFIG = {
    "comparison_csv": "site_comparison.csv",
    "csv_sep": ";",
    "csv_decimal": ",",
    "output_dir": "filter_sensitivity_out",
    "site_key": "site_id",
    # The screening script whose SELECTION is the baseline (in this folder).
    "lpc_module": "lpc_10",
    # True: baseline, upper bounds, decimals, rank_by, trim and exceptions
    # are taken from lpc_10's SELECTION (the values below are overwritten).
    "baseline_from_lpc": True,

    # The four filters and the thresholds TESTED for each ("values"; the
    # baseline must be one of them). "baseline" and "fixed_max" (an upper
    # bound that is not varied) come from lpc_10 when baseline_from_lpc.
    "filters": {
        "coverage_fraction": {"label": "Coverage", "baseline": 0.8,
                              "values": [0.7, 0.8, 0.9], "fixed_max": None},
        "load_factor_winter": {"label": "Load factor (winter)", "baseline": 0.4,
                               "values": [0.3, 0.4, 0.5], "fixed_max": None},
        "weekend_to_weekday_ratio_winter": {"label": "Weekend/weekday (winter)", "baseline": 0.8,
                                            "values": [0.7, 0.8, 0.9], "fixed_max": None},
        "temp_corr_winter_best_lag_value": {"label": "Temp. correlation (winter, best lag)",
                                            "baseline": 0.15, "values": [0.10, 0.15, 0.20],
                                            "fixed_max": None},
    },
    # Values are rounded to this many decimals before the comparison, exactly
    # as lpc_10 does (SELECTION["compare_decimals"]). None = exact.
    "compare_decimals": 2,
    # A missing value fails the filter (as lpc_10's missing_value_fails).
    "missing_value_fails": True,

    # What the cumulative share is computed on (lpc_10's rank_by).
    "share_by": "window_above_base_winter_kw",

    # Baseline tail trim: the rule that reproduces the current final group.
    #   {"rule": "n_smallest", "n": 3}          the n smallest contributors
    #   {"rule": "tail_share", "share": 0.075}  the smallest sites that together
    #                                           hold at most this share
    #   {"rule": "min_share", "share": 0.02}    every site under this share
    "baseline_trim": {"rule": "n_smallest", "n": 3},
    # Trim rules tested in D (the baseline rule is added automatically).
    "trim_rules": [
        {"rule": "tail_share", "share": 0.05},
        {"rule": "tail_share", "share": 0.075},
        {"rule": "tail_share", "share": 0.10},
        {"rule": "min_share", "share": 0.01},
        {"rule": "min_share", "share": 0.02},
        {"rule": "min_share", "share": 0.03},
    ],
    # Is the trim's denominator (total kW of the passing sites) computed
    # "without_exception" or "with_exception"? Both are reported in D and E.
    "trim_denominator": "without_exception",

    # Sites added by hand although they fail the filters: {site_id: why}.
    # {} = take the sites whose selection_note in site_comparison.csv says
    # "passed by hand" or "selected by hand" (lpc_10's force_pass / force_select).
    "exception_sites": {},

    # The current final group that Step 0 must reproduce:
    #   None            the "selected" column of site_comparison.csv
    #   [ids]           an explicit list of site ids
    #   "path.csv"      the site_id column of a file, e.g.
    #                   "portfolio_out/portfolio_15min.csv"
    "reference_final_group": None,

    # Winter months for the winter coverage column (fixed, never varied).
    "winter_months": [11, 12, 1, 2],
    "interval_minutes": 15,
    # Report a site when whole-span and winter-only coverage differ by more
    # than this.
    "coverage_diff_report": 0.05,
}


# ======================================================================
# reading
# ======================================================================
def _truthy(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def _norm(x):
    return str(x).strip().lower()


def load_lpc(cfg):
    import importlib
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        return importlib.import_module(cfg["lpc_module"])
    except ImportError as e:
        sys.exit(f"Cannot import {cfg['lpc_module']}.py ({e}). Put filter_sensitivity.py in the folder with it.")


def sync_with_lpc(cfg, L):
    """Take the baseline from lpc_10's SELECTION. Stops on settings the
    sensitivity test cannot reproduce."""
    sel = L.CONFIG.get("selection") or L.SELECTION
    lf = {c: sp for c, sp in sel.get("filters", {}).items() if sp.get("enabled", True)}
    extra = set(lf) - set(cfg["filters"])
    missing = set(cfg["filters"]) - set(lf)
    if extra or missing:
        sys.exit(f"lpc_10's SELECTION filters differ from the four tested here: extra {sorted(extra)}, "
                 f"switched off/missing {sorted(missing)}. Make them the same four.")
    for c, sp in lf.items():
        if sp.get("min") is None:
            sys.exit(f"lpc_10 filter {c} has no 'min' -- the test varies the minimum.")
        cfg["filters"][c]["baseline"] = sp["min"]
        cfg["filters"][c]["fixed_max"] = sp.get("max")
        vals = cfg["filters"][c]["values"]
        if not any(abs(v - sp["min"]) < 1e-9 for v in vals):
            sys.exit(f"lpc_10's baseline for {c} is {sp['min']}, which is not among the tested values {vals}. "
                     f"Set CONFIG['filters']['{c}']['values'] to include it.")
    hand = {k: sel.get(k) for k in ("force_pass", "force_fail", "force_select", "force_deselect") if sel.get(k)}
    if hand:
        sys.exit(f"lpc_10's SELECTION has sites chosen by hand: {hand}. The test cannot vary hand choices. "
                 f"Put the large site in SELECTION['exceptions'] and let SELECTION['trim'] drop the small "
                 f"ones; empty force_pass/force_fail/force_select/force_deselect; run `python "
                 f"{cfg['lpc_module']}.py --rank-only`, then rerun this.")
    if sel.get("select_mode", "cumulative_share") != "all_passing":
        sys.exit(f"lpc_10's select_mode is {sel.get('select_mode')!r}; the base case is 'all_passing' + trim. "
                 f"Set SELECTION['select_mode'] = 'all_passing'.")
    cfg["compare_decimals"] = sel.get("compare_decimals")
    cfg["missing_value_fails"] = sel.get("missing_value_fails", True)
    cfg["share_by"] = sel.get("rank_by", cfg["share_by"])
    cfg["baseline_trim"] = sel.get("trim") or {"rule": "n_smallest", "n": 0}
    cfg["exception_sites"] = dict(sel.get("exceptions") or {})
    print("[baseline from lpc_10] " + "; ".join(
        f"{c} >= {s['baseline']}" + (f" and <= {s['fixed_max']}" if s.get("fixed_max") is not None else "")
        for c, s in cfg["filters"].items())
        + f"; decimals {cfg['compare_decimals']}; trim {cfg['baseline_trim']}; exceptions "
        + (", ".join(cfg["exception_sites"]) or "none"))
    return sel


def read_metrics(cfg):
    path = cfg["comparison_csv"]
    if not os.path.exists(path):
        sys.exit(f"No {path!r} here. Run from the folder with site_comparison.csv "
                 f"(or set CONFIG['comparison_csv']).")
    df = pd.read_csv(path, sep=cfg["csv_sep"], decimal=cfg["csv_decimal"], encoding="utf-8-sig")
    key = cfg["site_key"]
    df[key] = df[key].astype(str).str.strip()
    if df[key].duplicated().any():
        sys.exit(f"Duplicate site ids in {path}: {df[key][df[key].duplicated()].tolist()}")
    return df


def winter_coverage(df, cfg):
    """Share of the winter (Nov-Feb) quarter-hours inside the site's data
    span (data_start..data_end) that have a value. lpc_10's
    coverage_fraction is the same share over the WHOLE span."""
    out, span_w = [], []
    step = pd.Timedelta(minutes=cfg["interval_minutes"])
    for _, r in df.iterrows():
        try:
            a = pd.Timestamp(str(r.get("data_start"))[:19])
            b = pd.Timestamp(str(r.get("data_end"))[:19])
            idx = pd.date_range(a, b, freq=step)
            n_exp = int(idx.month.isin(cfg["winter_months"]).sum())
            have = pd.to_numeric(r.get("winter_quarter_hours"), errors="coerce")
            out.append(float(have) / n_exp if n_exp and pd.notna(have) else np.nan)
            span_w.append(n_exp * step.total_seconds() / 86400)
        except Exception:
            out.append(np.nan)
            span_w.append(np.nan)
    return pd.Series(out, index=df.index), pd.Series(span_w, index=df.index)


def na_reason(r, col, df_cols):
    if col not in df_cols:
        return "column not in site_comparison.csv (row written by an older lpc -- run lpc_10.py --rerun-all)"
    wq = pd.to_numeric(r.get("winter_quarter_hours"), errors="coerce")
    if col != "coverage_fraction" and (pd.isna(wq) or wq == 0):
        return "no winter (Nov-Feb) data in the meter file" if wq == 0 else \
            "no winter metrics in this row (run lpc_10.py --rerun-all)"
    if col.startswith("temp_corr"):
        n = pd.to_numeric(r.get("temp_corr_n_samples"), errors="coerce")
        return ("no overlap with the weather file" if pd.isna(n) or n == 0 else
                f"too few overlapping winter samples for a correlation (all-year n = {int(n)})")
    if col.startswith("weekend"):
        return "no weekday or no weekend winter data (or weekday mean <= 0)"
    if col.startswith("load_factor"):
        return "winter peak <= 0"
    return "not computed"


def build_table(cfg):
    """The one metrics table every run uses."""
    df = read_metrics(cfg)
    key = cfg["site_key"]
    cols = list(cfg["filters"])
    keep = [key] + [c for c in cols if c in df.columns] + \
        [c for c in [cfg["share_by"], "filter_pass", "selected", "selection_note", "filter_fail_reasons",
                     "data_start", "data_end", "winter_quarter_hours", "temp_corr_n_samples", "pv_flag"]
         if c in df.columns]
    t = df[keep].copy()
    for c in cols:
        t[c] = pd.to_numeric(t[c], errors="coerce") if c in t.columns else np.nan
    t[cfg["share_by"]] = pd.to_numeric(t.get(cfg["share_by"]), errors="coerce")
    t["coverage_winter"], t["winter_days_in_span"] = winter_coverage(df, cfg)
    for c in cols:
        t[c + "_na_reason"] = [na_reason(r, c, df.columns) if pd.isna(r[c]) else "" for _, r in t.iterrows()]
    # exception sites
    exc = {_norm(k): v for k, v in (cfg["exception_sites"] or {}).items()}
    source = "CONFIG['exception_sites']"
    if not exc and not cfg.get("baseline_from_lpc") and "selection_note" in t.columns:
        note = t["selection_note"].fillna("").astype(str)
        m = note.str.contains(r"passed by hand|(?<!de)selected by hand", case=False, regex=True)
        exc = {_norm(s): n for s, n in zip(t.loc[m, key], note[m])}
        source = "selection_note in site_comparison.csv (force_pass / force_select)"
    elif cfg.get("baseline_from_lpc"):
        source = "lpc_10 SELECTION['exceptions']"
    unknown = set(exc) - set(t[key].map(_norm))
    if unknown:
        sys.exit(f"Exception site(s) not in {cfg['comparison_csv']}: {sorted(unknown)}")
    t["is_exception"] = t[key].map(_norm).isin(exc)
    t["exception_why"] = t[key].map(lambda s: exc.get(_norm(s), ""))
    return t, source


# ======================================================================
# filters, share, trim
# ======================================================================
def compared(v, cfg):
    d = cfg["compare_decimals"]
    return v.round(d) if d is not None else v


def apply_filters(t, thr, cfg):
    """thr: {col: threshold}. Returns a boolean Series 'passes all four'
    (exceptions are evaluated too; callers decide what to do with them)."""
    ok = pd.Series(True, index=t.index)
    for c, spec in cfg["filters"].items():
        v = compared(t[c], cfg)
        bad = v < thr[c]
        if spec.get("fixed_max") is not None:
            bad |= v > spec["fixed_max"]
        if cfg["missing_value_fails"]:
            bad |= v.isna()
        ok &= ~bad.fillna(False)
    return ok


def shares(t, members, cfg):
    """Each member's share of the summed share_by kW of the members
    (negative values count as 0, as in lpc_10)."""
    v = t.loc[members, cfg["share_by"]].clip(lower=0).fillna(0)
    tot = v.sum()
    return v / tot if tot > 0 else v * np.nan


def trim(t, members, rule, cfg, denominator_members=None):
    """Sites dropped from `members` by `rule`. The share of each site is
    taken over `denominator_members` (default: the members)."""
    den = members if denominator_members is None else denominator_members
    sh = shares(t, den, cfg).reindex(members)
    order = sh.sort_values(kind="mergesort")           # ascending, stable
    if rule["rule"] == "n_smallest":
        drop = list(order.index[:int(rule["n"])])
    elif rule["rule"] == "tail_share":
        cum = order.cumsum()
        drop = list(cum.index[cum <= rule["share"] + 1e-12])
    elif rule["rule"] == "min_share":
        drop = list(order.index[order < rule["share"]])
    else:
        raise ValueError(f"unknown trim rule {rule}")
    return drop, float(sh.reindex(drop).sum()) if drop else 0.0


def rule_name(rule):
    if rule["rule"] == "n_smallest":
        return f"{rule['n']} smallest"
    if rule["rule"] == "tail_share":
        return f"last {rule['share']:.1%} of cumulative share"
    return f"each under {rule['share']:.0%}"


def final_group(t, passing_idx, cfg):
    """passing (non-exception) sites - baseline trim + exceptions."""
    exc = list(t.index[t["is_exception"]])
    den = passing_idx if cfg["trim_denominator"] == "without_exception" else list(passing_idx) + exc
    drop, sh = trim(t, list(passing_idx), cfg["baseline_trim"], cfg, den)
    keep = [i for i in passing_idx if i not in drop]
    return keep + exc, drop, sh


# ======================================================================
# step 0
# ======================================================================
def reference_group(t, cfg):
    key = cfg["site_key"]
    ref = cfg["reference_final_group"]
    if ref is None:
        if "selected" not in t.columns:
            sys.exit("site_comparison.csv has no 'selected' column; set CONFIG['reference_final_group'].")
        return set(t.loc[_truthy(t["selected"]), key].map(_norm)), "the 'selected' column of site_comparison.csv"
    if isinstance(ref, str):
        r = pd.read_csv(ref, sep=cfg["csv_sep"], decimal=cfg["csv_decimal"], encoding="utf-8-sig")
        return set(r[key].astype(str).map(_norm)), ref
    return {_norm(x) for x in ref}, "CONFIG['reference_final_group']"


def baseline_check(t, cfg, L=None):
    key = cfg["site_key"]
    base = {c: s["baseline"] for c, s in cfg["filters"].items()}
    ok = apply_filters(t, base, cfg)
    passing = t.index[ok & ~t["is_exception"]]
    grp, drop, dsh = final_group(t, list(passing), cfg)
    mine_pass = set(t.loc[passing, key])
    mine_f = set(t.loc[grp, key])
    problems = []
    raw = read_metrics(cfg)
    stored_pass = set(raw.loc[_truthy(raw["filter_pass"]), key]) if "filter_pass" in raw else None
    stored_sel = set(raw.loc[_truthy(raw["selected"]), key]) if "selected" in raw else None
    ex_names = set(t.loc[t["is_exception"], key])
    if L is not None:
        # (b) lpc_10's own code on the same table
        lc = dict(L.CONFIG)
        out = L.select_sites(raw.copy(), lc)
        lpc_pass = set(out.loc[out["filter_pass"].astype(bool), key]) - ex_names
        lpc_sel = set(out.loc[out["selected"].astype(bool), key])
        for s in sorted(lpc_pass ^ mine_pass):
            r = t.set_index(key).loc[s]
            problems.append(f"filter (re-implementation vs lpc_10 code): {s} passes in lpc_10 = {s in lpc_pass}, "
                            f"here = {s in mine_pass} ({', '.join(f'{c} {r[c]}' for c in cfg['filters'])})")
        for s in sorted(lpc_sel ^ mine_f):
            problems.append(f"final group (re-implementation vs lpc_10 code): {s} in lpc_10 = {s in lpc_sel}, "
                            f"here = {s in mine_f}")
        # (a) is the stored file up to date with SELECTION?
        stale = []
        if stored_pass is not None and (stored_pass - ex_names) != lpc_pass:
            stale.append("filter_pass")
        if stored_sel is not None and stored_sel != lpc_sel:
            stale.append("selected")
        if stale:
            problems.append(f"site_comparison.csv is out of date with lpc_10's SELECTION (columns {', '.join(stale)} "
                            f"differ: stored selection {sorted(stored_sel or [])} vs now {sorted(lpc_sel)}). Run "
                            f"`python {cfg['lpc_module']}.py --rank-only` and rerun this.")
        src = f"{cfg['lpc_module']}.py's selection of site_comparison.csv"
    else:
        ref, src = reference_group(t, cfg)
        for s in sorted(ref - {x.lower() for x in mine_f}):
            problems.append(f"final group: {s} is in {src} but not reproduced here")
        for s in sorted({x.lower() for x in mine_f} - ref):
            problems.append(f"final group: {s} is reproduced here but not in {src}")
    return {"passing": list(passing), "final": grp, "dropped": drop, "dropped_share": dsh,
            "problems": problems, "reference_source": src}


# ======================================================================
# A, B, C, D, E
# ======================================================================
def run_sets(cfg):
    cols = list(cfg["filters"])
    base = {c: cfg["filters"][c]["baseline"] for c in cols}
    for c in cols:
        if base[c] not in cfg["filters"][c]["values"]:
            sys.exit(f"The baseline {base[c]} of {c} is not among its values {cfg['filters'][c]['values']}")
    oat = [("baseline", "-", None, dict(base))]
    for c in cols:
        for v in cfg["filters"][c]["values"]:
            if v != base[c]:
                d = dict(base)
                d[c] = v
                oat.append((f"oat_{c}_{v}", c, v, d))
    grid = []
    for combo in itertools.product(*[cfg["filters"][c]["values"] for c in cols]):
        d = dict(zip(cols, combo))
        grid.append(("grid_" + "_".join(f"{x:g}" for x in combo), "grid", None, d))
    return oat, grid


def evaluate(t, runs, kind, base_final, cfg):
    key = cfg["site_key"]
    rows, member = [], {}
    for rid, param, val, thr in runs:
        ok = apply_filters(t, thr, cfg)
        passing = list(t.index[ok & ~t["is_exception"]])
        grp, drop, dsh = final_group(t, passing, cfg)
        exc_would = [t.loc[i, key] for i in t.index[ok & t["is_exception"]]]
        fset = set(t.loc[grp, key])
        rows.append({
            "run_id": rid, "kind": kind, "parameter_changed": param, "value": val,
            **{f"thr_{c}": thr[c] for c in cfg["filters"]},
            "n_pass": len(passing), "sites_pass": "|".join(t.loc[passing, key]),
            "trim_dropped": "|".join(t.loc[drop, key]), "trim_dropped_share": round(dsh, 4),
            "n_final_incl_exception": len(grp), "final_group": "|".join(sorted(fset)),
            "final_group_changed": fset != base_final,
            "final_added": "|".join(sorted(fset - base_final)), "final_removed": "|".join(sorted(base_final - fset)),
            "exception_would_pass": "|".join(exc_would),
        })
        member[rid] = np.where(t["is_exception"], np.where(ok, "exception (would pass)", "exception (fails)"),
                               np.where(ok, "pass", "fail"))
    return pd.DataFrame(rows), pd.DataFrame(member, index=t[key])


def margins(t, cfg):
    key = cfg["site_key"]
    rows = []
    for i, r in t.iterrows():
        rec = {key: r[key], "is_exception": r["is_exception"]}
        rel = {}
        for c, s in cfg["filters"].items():
            v = r[c]
            vc = compared(pd.Series([v]), cfg).iloc[0]
            thr = s["baseline"]
            if pd.isna(v):
                m = np.nan
                rec[f"{c}_value"] = "N/A"
                rec[f"{c}_na_reason"] = r[c + "_na_reason"]
            else:
                m = vc - thr
                if s.get("fixed_max") is not None:
                    m = min(m, s["fixed_max"] - vc)
                rec[f"{c}_value"] = round(float(v), 4)
                rec[f"{c}_na_reason"] = ""
                rel[c] = m / thr if thr else m
            rec[f"{c}_value_compared"] = vc
            rec[f"{c}_threshold"] = thr
            rec[f"{c}_margin"] = None if pd.isna(m) else round(float(m), 4)
            rec[f"{c}_rel_margin"] = None if pd.isna(m) else round(float(rel[c]), 4)
        missing = [c for c in cfg["filters"] if pd.isna(r[c])]
        fails = {c: x for c, x in rel.items() if x < 0}
        if missing and cfg["missing_value_fails"]:
            rec["baseline_pass"] = False
        else:
            rec["baseline_pass"] = not fails
        if fails:
            b = min(fails, key=fails.get)
            rec["binding_filter"] = b
            rec["binding_rel_margin"] = round(fails[b], 4)
        elif missing:
            rec["binding_filter"] = "N/A: " + ", ".join(missing)
            rec["binding_rel_margin"] = None
        else:
            b = min(rel, key=rel.get) if rel else ""
            rec["binding_filter"] = ""
            rec["tightest_filter"] = b
            rec["tightest_rel_margin"] = round(rel[b], 4) if b else None
        failed_txt = [f"{c} by {abs(x):.1%}" for c, x in sorted(fails.items(), key=lambda z: z[1])]
        failed_txt += [f"{c} N/A ({r[c + '_na_reason']})" for c in missing] if cfg["missing_value_fails"] else []
        rec["filters_failed"] = "; ".join(failed_txt)
        rec["coverage_fraction_span"] = r["coverage_fraction"]
        rec["coverage_winter"] = None if pd.isna(r["coverage_winter"]) else round(float(r["coverage_winter"]), 4)
        rows.append(rec)
    return pd.DataFrame(rows)


def trim_table(t, passing, cfg):
    key = cfg["site_key"]
    exc = list(t.index[t["is_exception"]])
    rules = [cfg["baseline_trim"]] + [r for r in cfg["trim_rules"] if r != cfg["baseline_trim"]]
    rows = []
    base_drop = {}
    for den_name, den in [("without_exception", list(passing)), ("with_exception", list(passing) + exc)]:
        for j, rule in enumerate(rules):
            drop, sh = trim(t, list(passing), rule, cfg, den)
            names = sorted(t.loc[drop, key])
            if j == 0:
                base_drop[den_name] = set(names)
            rows.append({"rule": rule_name(rule) + (" (baseline rule)" if j == 0 else ""),
                         "denominator": den_name, "n_dropped": len(names),
                         "dropped_sites": "|".join(names), "dropped_share": round(sh, 4),
                         "same_as_baseline_rule": set(names) == base_drop[den_name]})
    return pd.DataFrame(rows)


def share_table(t, passing, cfg):
    key = cfg["site_key"]
    exc = list(t.index[t["is_exception"]])
    a = shares(t, list(passing), cfg)
    b = shares(t, list(passing) + exc, cfg)
    out = pd.DataFrame({key: t.loc[list(passing) + exc, key],
                        "is_exception": t.loc[list(passing) + exc, "is_exception"],
                        cfg["share_by"]: t.loc[list(passing) + exc, cfg["share_by"]],
                        "share_without_exception": a.reindex(list(passing) + exc).round(4),
                        "share_with_exception": b.round(4)})
    out = out.sort_values(cfg["share_by"], ascending=False, kind="mergesort")
    out["cum_share_without_exception"] = out["share_without_exception"].fillna(0).cumsum().round(4)
    out.loc[out["is_exception"], "cum_share_without_exception"] = np.nan
    out["cum_share_with_exception"] = out["share_with_exception"].cumsum().round(4)
    return out


# ======================================================================
# plots
# ======================================================================
def plots(t, oat_df, oat_mem, freq, marg, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    out = cfg["output_dir"]
    key = cfg["site_key"]
    # 1. membership heatmap (one-at-a-time runs) + grid frequency
    order = freq.sort_values(["selection_frequency", key], ascending=[False, True])[key].tolist()
    m = oat_mem.loc[order]
    code = m.replace({"fail": 0, "pass": 1, "exception (fails)": 2, "exception (would pass)": 3}).astype(int)
    labels = [c.replace("oat_", "").replace("coverage_fraction", "cov").replace("load_factor_winter", "LF")
              .replace("weekend_to_weekday_ratio_winter", "WE").replace("temp_corr_winter_best_lag_value", "TC")
              for c in m.columns]
    h = max(3, 0.28 * len(order) + 1.5)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, h), gridspec_kw={"width_ratios": [3, 1]}, sharey=True)
    cmap = ListedColormap(["#e8e8e8", "#2b6cb0", "#f6d6a8", "#d97706"])
    ax.imshow(code.values, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_title("Filter pass per one-at-a-time run\n(blue = pass, grey = fail, orange = exception site)", fontsize=9)
    f = freq.set_index(key).loc[order, "selection_frequency"]
    ax2.barh(range(len(order)), f.values, color=["#d97706" if e else "#2b6cb0"
                                                for e in freq.set_index(key).loc[order, "is_exception"]])
    ax2.set_xlim(0, 1)
    ax2.set_title(f"Pass share over the {cfg['_n_grid']} grid runs", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "membership_heatmap.png"), dpi=130)
    plt.close(fig)
    # 2. group size against each threshold
    cols = list(cfg["filters"])
    fig, axs = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.2), sharey=True)
    base = oat_df[oat_df["run_id"] == "baseline"].iloc[0]
    for ax, c in zip(np.atleast_1d(axs), cols):
        pts = [(cfg["filters"][c]["baseline"], base["n_pass"], base["n_final_incl_exception"])]
        for _, r in oat_df[oat_df["parameter_changed"] == c].iterrows():
            pts.append((r["value"], r["n_pass"], r["n_final_incl_exception"]))
        pts.sort()
        x = [p[0] for p in pts]
        ax.plot(x, [p[1] for p in pts], "o-", color="#2b6cb0", label="pass the filters")
        ax.plot(x, [p[2] for p in pts], "s--", color="#555", label="final group (trim, + exception)")
        ax.axvline(cfg["filters"][c]["baseline"], color="#d97706", lw=1, ls=":")
        ax.set_xticks(x)
        ax.set_title(cfg["filters"][c]["label"], fontsize=9)
        ax.grid(alpha=0.3)
    np.atleast_1d(axs)[0].set_ylabel("number of sites")
    np.atleast_1d(axs)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "group_size_vs_threshold.png"), dpi=130)
    plt.close(fig)
    # 3. margins per filter
    fig, axs = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), max(3, 0.22 * len(t) + 1.5)))
    for ax, c in zip(np.atleast_1d(axs), cols):
        d = marg[[key, "is_exception", f"{c}_margin"]].copy()
        d["m"] = pd.to_numeric(d[f"{c}_margin"], errors="coerce")
        d = d.sort_values("m", na_position="first", kind="mergesort")
        y = np.arange(len(d))
        col = ["#d97706" if e else ("#2b6cb0" if mm >= 0 else "#c53030") for e, mm in zip(d["is_exception"], d["m"])]
        ax.barh(y, d["m"].fillna(0), color=col)
        for yy, mm in zip(y, d["m"]):
            if pd.isna(mm):
                ax.text(0, yy, " N/A", va="center", fontsize=6, color="#555")
        lo, hi = min(cfg["filters"][c]["values"]), max(cfg["filters"][c]["values"])
        b = cfg["filters"][c]["baseline"]
        ax.axvspan(lo - b, hi - b, color="#999", alpha=0.15, lw=0)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(d[key], fontsize=6)
        ax.set_title(f"{cfg['filters'][c]['label']}\nvalue - {b} (grey: tested range)", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "margins_per_filter.png"), dpi=130)
    plt.close(fig)


# ======================================================================
# main
# ======================================================================
def write(df, name, cfg, index=False):
    p = os.path.join(cfg["output_dir"], name)
    df.to_csv(p, sep=cfg["csv_sep"], decimal=cfg["csv_decimal"], index=index, encoding="utf-8-sig")
    return p


def main(cfg=CONFIG, skip_check=False):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    key = cfg["site_key"]
    L = None
    if cfg.get("baseline_from_lpc"):
        L = load_lpc(cfg)
        sync_with_lpc(cfg, L)
    t, exc_source = build_table(cfg)
    write(t, "filter_sensitivity_metrics.csv", cfg)
    exc_names = t.loc[t["is_exception"], key].tolist()
    print(f"[sensitivity] {len(t)} sites from {cfg['comparison_csv']}; exception site(s): "
          f"{exc_names or 'none'} (from {exc_source})")

    # ---- step 0
    chk = baseline_check(t, cfg, L)
    if chk["problems"]:
        print("\n[step 0] the baseline is NOT reproduced:")
        for p in chk["problems"]:
            print("   - " + p)
        print("\n  If the file is out of date: python lpc_10.py --rank-only. Any other line is a mismatch "
              "between lpc_10 and this script -- send it to me.")
        with open(os.path.join(cfg["output_dir"], "step0_baseline_check.txt"), "w", encoding="utf-8") as fh:
            fh.write("NOT reproduced\n" + "\n".join(chk["problems"]) + "\n")
        if not skip_check:
            sys.exit("[step 0] stopped. Fix the cause above (or run with --skip-baseline-check to look anyway).")
    else:
        print(f"[step 0] baseline reproduced: {len(chk['passing'])} sites pass, trim drops "
              f"{len(chk['dropped'])} ({chk['dropped_share']:.1%}), final group {len(chk['final'])} "
              f"incl. exception -- matches {chk['reference_source']}")
        with open(os.path.join(cfg["output_dir"], "step0_baseline_check.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"reproduced (reference: {chk['reference_source']})\n")
    base_final = set(t.loc[chk["final"], key])

    # ---- A and B
    oat, grid = run_sets(cfg)
    cfg["_n_grid"] = len(grid)
    oat_df, oat_mem = evaluate(t, oat, "one_at_a_time", base_final, cfg)
    grid_df, grid_mem = evaluate(t, grid, "grid", base_final, cfg)
    write(pd.concat([oat_df, grid_df], ignore_index=True), "filter_sensitivity_runs.csv", cfg)
    write(pd.concat([oat_mem, grid_mem], axis=1), "filter_sensitivity_membership.csv", cfg, index=True)
    gm = grid_mem
    freq = pd.DataFrame({key: gm.index,
                         "is_exception": t["is_exception"].values,
                         "runs_passed": (gm.isin(["pass", "exception (would pass)"])).sum(axis=1).values,
                         "n_runs": gm.shape[1]})
    freq["selection_frequency"] = (freq["runs_passed"] / freq["n_runs"]).round(4)
    freq["note"] = np.where(freq["is_exception"], "EXCEPTION: never counted as passing; frequency is "
                                                  "how often it WOULD pass", "")
    freq["in_final_group_runs"] = [int(grid_df["final_group"].str.split("|").map(lambda s, k=k: k in s).sum())
                                   for k in freq[key]]
    freq = freq.sort_values(["selection_frequency", key], ascending=[False, True], kind="mergesort")
    write(freq, "filter_factorial_selection_frequency.csv", cfg)

    # ---- C
    marg = margins(t, cfg)
    write(marg, "filter_margins.csv", cfg)

    # ---- D and E
    passing = chk["passing"]
    tr = trim_table(t, passing, cfg)
    write(tr, "trim_sensitivity.csv", cfg)
    sh = share_table(t, passing, cfg)
    write(sh, "cumulative_share_with_without_exception.csv", cfg)

    plots(t, oat_df, oat_mem, freq, marg, cfg)

    # ---- note (facts only)
    oat_runs = oat_mem.columns
    always = [s for s in oat_mem.index if (oat_mem.loc[s] == "pass").all()]
    short = {c: cfg["filters"][c]["label"] for c in cfg["filters"]}
    drop_out, come_in = {}, {}
    for s in oat_mem.index:
        if t.set_index(key).loc[s, "is_exception"]:
            continue
        fl = sorted({short[oat_df.loc[oat_df["run_id"] == r, "parameter_changed"].iloc[0]]
                     for r in oat_runs if oat_mem.loc[s, r] != oat_mem.loc[s, "baseline"]})
        if fl:
            (drop_out if oat_mem.loc[s, "baseline"] == "pass" else come_in)[s] = fl
    swing = {}
    for c in cfg["filters"]:
        sub = oat_df[oat_df["parameter_changed"] == c]
        changes = [len((set(r["sites_pass"].split("|")) ^ set(oat_df.iloc[0]["sites_pass"].split("|"))) - {""})
                   for _, r in sub.iterrows()]
        swing[c] = sum(changes)
    top = max(swing, key=swing.get)
    fin_changed_oat = oat_df[oat_df["final_group_changed"]]
    fin_changed_grid = int(grid_df["final_group_changed"].sum())
    trim_same = tr.groupby("denominator")["same_as_baseline_rule"].all().to_dict()
    exc_lines = []
    for _, r in marg[marg["is_exception"]].iterrows():
        exc_lines.append(f"{r[key]}: fails {r.get('filters_failed') or 'none'}; would pass "
                         f"{int(freq.set_index(key).loc[r[key], 'runs_passed'])} of {len(grid)} grid runs")
    covdiff = t[(t["coverage_fraction"] - t["coverage_winter"]).abs() > cfg["coverage_diff_report"]][key].tolist()
    nas = {c: t.loc[t[c].isna(), key].tolist() for c in cfg["filters"] if t[c].isna().any()}
    lines = [
        "# Filter sensitivity -- note",
        "",
        f"1. Baseline reproduced: {'YES' if not chk['problems'] else 'NO (' + str(len(chk['problems'])) + ' differences, see step0_baseline_check.txt)'}"
        f"; {len(chk['passing'])} pass, trim drops {len(chk['dropped'])} ({chk['dropped_share']:.1%}), final group {len(chk['final'])} incl. exception.",
        f"2. Pass in every one-at-a-time run: {len(always)} of {len(chk['passing'])} baseline sites "
        f"({', '.join(always) or 'none'}). Marginal -- pass at baseline but drop out in some run: "
        + ("; ".join(f"{s} ({', '.join(f)})" for s, f in drop_out.items()) or "none")
        + ". Fail at baseline but come in in some run: "
        + ("; ".join(f"{s} ({', '.join(f)})" for s, f in come_in.items()) or "none") + ".",
        f"3. Sites changing status per filter, summed over its two non-baseline values: "
        + ", ".join(f"{cfg['filters'][c]['label']} {n}" for c, n in swing.items()) + f" -- most: {cfg['filters'][top]['label']}.",
        f"4. Final group (baseline trim rule '{rule_name(cfg['baseline_trim'])}', + exception) changes in "
        f"{len(fin_changed_oat)} of {len(oat_df) - 1} one-at-a-time runs and {fin_changed_grid} of {len(grid)} grid runs"
        + (": " + "; ".join(f"{r['run_id']} (+{r['final_added'] or '-'} / -{r['final_removed'] or '-'})"
                            for _, r in fin_changed_oat.iterrows()) if len(fin_changed_oat) else "") + ".",
        f"5. Trim rules giving the same dropped sites as the baseline rule: "
        + "; ".join(f"{d}: {'all' if v else 'not all'}" for d, v in trim_same.items()) + " (see trim_sensitivity.csv).",
        "6. Exception: " + ("; ".join(exc_lines) if exc_lines else "none marked") + ".",
        f"7. Coverage: coverage_fraction is over each site's whole data span; winter-only coverage added. "
        f"Differ by > {cfg['coverage_diff_report']:.0%}: {', '.join(covdiff) or 'none'}.",
        "8. Missing values (N/A, fail the filter): " + ("; ".join(f"{c}: {', '.join(v)}" for c, v in nas.items()) or "none") + ".",
    ]
    with open(os.path.join(cfg["output_dir"], "filter_sensitivity_note.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[2:]))
    print(f"\n[sensitivity] outputs in {os.path.abspath(cfg['output_dir'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-baseline-check", action="store_true")
    ap.add_argument("--comparison", help="path to site_comparison.csv")
    a = ap.parse_args()
    if a.comparison:
        CONFIG["comparison_csv"] = a.comparison
    main(CONFIG, a.skip_baseline_check)

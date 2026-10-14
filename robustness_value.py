"""
Robustness against value: site table, disagreements and the two-axis chart.

Extends the filter-sensitivity module (filter_sensitivity.py). Nothing is
recomputed: it reads the files that run wrote (CONFIG["sensitivity_dir"]):
    filter_factorial_selection_frequency.csv   selection frequency, 81 runs
    filter_margins.csv                         margins, binding filter
    filter_sensitivity_metrics.csv             winter metrics, coverage, PV, exception
    filter_sensitivity_runs.csv                the baseline run: final group, trimmed
    cumulative_share_with_without_exception.csv
plus site_comparison.csv (PV flag, export share and, if present, the old
composite rank with its six factor scores).

The composite rank and its six scores come from lpc_8's FIRST ranking.
lpc_9 / lpc_10 delete those columns whenever they rewrite site_comparison.csv.
If your file no longer has them, point CONFIG["composite_csv"] at an older copy
of site_comparison.csv that does (columns rank_score and score_*). Without
them the composite columns are N/A, Part 2 and the second chart are skipped,
and the rest still runs.

Run from the folder with site_comparison.csv:
    python robustness_value.py
    python robustness_value.py --rerun-sensitivity    (runs filter_sensitivity.py first)
Outputs go to CONFIG["output_dir"]; the private key to CONFIG["private_dir"].

Every site id in the outputs is anonymised (anonymise.py's list: Site A ...,
"Other site n" for the rest). If the folder is already an anonymised copy, the
ids stay as they are. The key real name -> code is written ONLY to private_dir.
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

# ======================================================================
# CONFIG
# ======================================================================
CONFIG = {
    "sensitivity_dir": "filter_sensitivity_out",
    "comparison_csv": "site_comparison.csv",
    "composite_csv": None,           # an older site_comparison.csv with rank_score / score_* (None = comparison_csv)
    "output_dir": "robustness_value_out",
    "private_dir": "robustness_value_private",     # the key. Not for sharing.
    "csv_sep": ";",
    "csv_decimal": ",",
    "site_key": "site_id",
    "anonymise": True,

    # ---- chart 1 -------------------------------------------------------
    "x_threshold": 0.8,              # selection frequency separating robust / fragile
    "y_threshold": "median",         # kW separating large / small: "median" or a number.
                                     # The median is taken over all screened sites; the same
                                     # line without the exception site is shown dotted.
    "y_scale": "linear",             # "linear" or "log" for robustness_vs_value.*
    "also_log_version": True,        # also write robustness_vs_value_log.* when the sites span a wide range
    "log_if_ratio_over": 10,         # ... max / median above this

    # ---- Part 2: disagreements ------------------------------------------
    # high frequency = selection frequency >= this quantile of the screened sites (ties included);
    # low frequency = <= (1 - quantile). The composite bottom/top half is by rank.
    "high_freq_quantile": 2 / 3,
    "composite_half": 0.5,
    # classification (see the note the script writes)
    "unreliable_factors": ["peak_frac", "window_peak", "summer"],    # substrings of score_* names
    "pv_distorted_factors": ["load_factor", "summer", "temp_corr"],  # factors solar distorts
    "min_factor_gap": 0.15,          # the lowering factor must sit this far (score points) below the median site
    "stretch_spread": 0.05,          # raw values of a factor whose IQR / median is under this are "stretched" by the percentile scoring
    "size_percentile": 0.5,          # kw_above_base_window below this quantile = "small"
}

STATUS_ORDER = ["final", "trimmed", "exception", "excluded"]
QUADRANT_NAMES = {
    ("robust", "large"): "robust and large", ("robust", "small"): "robust and small",
    ("fragile", "large"): "fragile and large", ("fragile", "small"): "fragile and small",
}


# ======================================================================
# reading
# ======================================================================
def read_csv(path, cfg, required=True):
    if not os.path.exists(path):
        if required:
            sys.exit(f"Missing {path!r}. Run filter_sensitivity.py first (or "
                     f"python robustness_value.py --rerun-sensitivity).")
        return None
    return pd.read_csv(path, sep=cfg["csv_sep"], decimal=cfg["csv_decimal"], encoding="utf-8-sig")


def truthy(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def load_inputs(cfg):
    key = cfg["site_key"]
    d = cfg["sensitivity_dir"]
    freq = read_csv(os.path.join(d, "filter_factorial_selection_frequency.csv"), cfg)
    marg = read_csv(os.path.join(d, "filter_margins.csv"), cfg)
    met = read_csv(os.path.join(d, "filter_sensitivity_metrics.csv"), cfg)
    runs = read_csv(os.path.join(d, "filter_sensitivity_runs.csv"), cfg)
    cum = read_csv(os.path.join(d, "cumulative_share_with_without_exception.csv"), cfg, required=False)
    comp = read_csv(cfg["comparison_csv"], cfg)
    for t in (freq, marg, met, comp):
        t[key] = t[key].astype(str).str.strip()
    if cum is not None:
        cum[key] = cum[key].astype(str).str.strip()
    return freq, marg, met, runs, cum, comp


def composite_source(cfg, comp):
    """(table with composite_rank and the six scores, how it was found / why not)."""
    key = cfg["site_key"]
    src = comp
    where = cfg["comparison_csv"]
    if cfg.get("composite_csv"):
        src = read_csv(cfg["composite_csv"], cfg)
        src[key] = src[key].astype(str).str.strip()
        where = cfg["composite_csv"]
    scores = [c for c in src.columns if c.startswith("score_")]
    if "rank_score" not in src.columns and not scores:
        return None, (f"no rank_score / score_* columns in {where} (lpc_9 / lpc_10 delete them when they "
                      f"rewrite the file; set CONFIG['composite_csv'] to an older site_comparison.csv that has them)")
    out = src[[key]].copy()
    for c in scores:
        out[c] = pd.to_numeric(src[c], errors="coerce")
    if "rank_score" in src.columns:
        sc = pd.to_numeric(src["rank_score"], errors="coerce")
        how = f"rank_score in {where}"
    else:
        sc = out[scores].mean(axis=1)
        how = f"mean of the {len(scores)} score_* columns in {where} (equal weights)"
    out["composite_score"] = sc
    out["composite_rank"] = sc.rank(ascending=False, method="min")
    for c in src.columns:                                       # raw metrics, for the "stretched" test
        if c not in out.columns and not c.startswith("score_") and c != "rank_score":
            out["raw__" + c] = pd.to_numeric(src[c], errors="coerce")
    return out, how


# ======================================================================
# anonymisation
# ======================================================================
def build_anonymiser(site_ids, cfg):
    if not cfg["anonymise"]:
        return {s: s for s in site_ids}, None
    try:
        sys.path.insert(0, os.getcwd())
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import anonymise as A
    except ImportError:
        sys.exit("anonymise.py is not in this folder. Put it next to this script, or set "
                 "CONFIG['anonymise'] = False for a private run.")
    an = A.Anonymiser()
    an.learn(site_ids)
    mapping = {s: an.text(s) for s in site_ids}
    codes = list(mapping.values())
    dup = {c for c in codes if codes.count(c) > 1}
    if dup:
        sys.exit(f"Two sites got the same code {sorted(dup)} -- check the site ids against anonymise.py's list.")
    return mapping, an


# ======================================================================
# Part 1
# ======================================================================
def build_table(cfg, freq, marg, met, runs, cum, comp, composite):
    key = cfg["site_key"]
    share_by = None
    for c in met.columns:
        if c == "window_above_base_winter_kw":
            share_by = c
    if share_by is None:
        sys.exit("filter_sensitivity_metrics.csv has no window_above_base_winter_kw column.")
    t = pd.DataFrame({key: met[key]})
    f = freq.set_index(key)
    t["selection_frequency"] = t[key].map(f["selection_frequency"])
    t["kw_above_base_window"] = pd.to_numeric(met[share_by], errors="coerce").values
    t["kw_na_reason"] = np.where(t["kw_above_base_window"].isna(),
                                 "window_above_base_winter_kw missing (no winter data or no grid window)", "")
    base = runs[runs["run_id"] == "baseline"].iloc[0]
    final = set(str(base["final_group"]).split("|")) if isinstance(base["final_group"], str) else set()
    trimmed = set(str(base["trim_dropped"]).split("|")) if isinstance(base["trim_dropped"], str) else set()
    exc = set(met.loc[truthy(met["is_exception"]), key])
    passing = set(str(base["sites_pass"]).split("|")) if isinstance(base["sites_pass"], str) else set()
    def status(s):
        if s in exc:
            return "exception"
        if s in trimmed:
            return "trimmed"
        if s in final:
            return "final"
        if s in passing:
            return "passing (not final)"
        return "excluded"
    t["status"] = t[key].map(status)
    t["is_exception"] = t[key].isin(exc)
    # rank by value (the exception is ranked too: it is in the plot)
    t["rank_by_kw"] = t["kw_above_base_window"].rank(ascending=False, method="min")
    # cumulative shares: the baseline passing sites (+ exception) as computed by the sensitivity run
    if cum is not None:
        c = cum.set_index(key)
        t["cumulative_share_without_exception"] = t[key].map(c["cum_share_without_exception"])
        t["cumulative_share_with_exception"] = t[key].map(c["cum_share_with_exception"])
        notin = ~t[key].isin(c.index)
        t["cumulative_share_note"] = np.where(notin, "not among the sites passing the baseline filters", "")
        t.loc[t["is_exception"], "cumulative_share_note"] = "exception: not in the share without exception"
    else:
        t["cumulative_share_without_exception"] = np.nan
        t["cumulative_share_with_exception"] = np.nan
        t["cumulative_share_note"] = "cumulative_share_with_without_exception.csv missing"
    # composite
    if composite is not None:
        cc = composite.set_index(key)
        t["composite_rank"] = t[key].map(cc["composite_rank"])
        t["composite_note"] = np.where(t["composite_rank"].isna(), "site not in the composite file", "")
    else:
        t["composite_rank"] = np.nan
        t["composite_note"] = "N/A: no composite in the data"
    t["rank_difference"] = t["composite_rank"] - t["rank_by_kw"]
    t["freq_rank"] = t["selection_frequency"].rank(ascending=False, method="min")
    # PV
    cs = comp.set_index(key)
    pv = truthy(cs["pv_flag"]).reindex(t[key]).fillna(False).values if "pv_flag" in cs else np.zeros(len(t), bool)
    exs = pd.to_numeric(cs["export_share"], errors="coerce").reindex(t[key]).values if "export_share" in cs else np.full(len(t), np.nan)
    t["pv_flag"] = ["yes" if p else "no" for p in pv]
    t["export_share"] = exs
    t["pv_text"] = [("yes" + (f" (export {e:.1%} of import)" if pd.notna(e) else " (no export share)")) if p else "no"
                    for p, e in zip(pv, exs)]
    # tightest / binding margin
    m = marg.set_index(key)
    def tight(s):
        r = m.loc[s]
        if isinstance(r.get("binding_filter"), str) and r["binding_filter"]:
            rel = r.get("binding_rel_margin")
            return f"binding: {r['binding_filter']}" + (f" ({rel:+.1%})" if pd.notna(rel) else "")
        tf = r.get("tightest_filter")
        if isinstance(tf, str) and tf:
            rel = r.get("tightest_rel_margin")
            return f"tightest: {tf}" + (f" ({rel:+.1%})" if pd.notna(rel) else "")
        return "N/A"
    t["tightest_margin"] = t[key].map(tight)
    t["winter_coverage"] = pd.to_numeric(met["coverage_winter"], errors="coerce").values if "coverage_winter" in met else np.nan
    return t


# ======================================================================
# Part 2
# ======================================================================
def disagreements(cfg, t, composite, comp):
    """Returns (table, rule text lines, counts)."""
    key = cfg["site_key"]
    if composite is None:
        return None, ["composite not available: Part 2 skipped"], {}
    d = t.dropna(subset=["selection_frequency", "composite_rank"]).copy()
    n_comp = int(composite["composite_rank"].notna().sum())
    hi = d["selection_frequency"].quantile(cfg["high_freq_quantile"])
    lo = d["selection_frequency"].quantile(1 - cfg["high_freq_quantile"])
    d["high_freq"] = d["selection_frequency"] >= hi
    d["low_freq"] = d["selection_frequency"] <= lo
    d["comp_bottom"] = d["composite_rank"] > n_comp * cfg["composite_half"]
    d["comp_top"] = d["composite_rank"] <= n_comp * cfg["composite_half"]
    fwd = d["high_freq"] & d["comp_bottom"]
    rev = d["low_freq"] & d["comp_top"] & ~d["high_freq"]
    d["direction"] = np.where(fwd, "robust but low composite", np.where(rev, "fragile but high composite", ""))
    dd = d[d["direction"] != ""].copy()
    scores = [c for c in composite.columns if c.startswith("score_")]
    cc = composite.set_index(key)
    med = cc[scores].median()
    kw_cut = t["kw_above_base_window"].quantile(cfg["size_percentile"])
    rows = []
    for _, r in dd.iterrows():
        s = r[key]
        sc = cc.loc[s, scores] if scores else pd.Series(dtype=float)
        gap = (sc - med) if scores else pd.Series(dtype=float)
        fwd_dir = r["direction"].startswith("robust")
        driver, driver_gap = "", np.nan
        if len(gap.dropna()):
            driver = gap.idxmin() if fwd_dir else gap.idxmax()
            driver_gap = gap[driver]
        evidence, fits = [], []
        big_gap = pd.notna(driver_gap) and abs(driver_gap) >= cfg["min_factor_gap"]
        is_unrel = bool(driver) and any(k in driver for k in cfg["unreliable_factors"])
        raw_col = "raw__" + driver.replace("score_", "", 1) if driver else ""
        stretched = False
        if raw_col and raw_col in composite.columns:
            v = composite[raw_col].dropna()
            if len(v) >= 4 and abs(v.median()) > 0:
                spread = (v.quantile(0.75) - v.quantile(0.25)) / abs(v.median())
                stretched = spread < cfg["stretch_spread"]
                if stretched:
                    evidence.append(f"raw {driver.replace('score_', '', 1)} varies little between sites (IQR/median {spread:.1%}) "
                                    f"but the percentile score spreads it")
        if is_unrel and big_gap:
            fits.append("weak_factor")
            evidence.append(f"{driver} is a factor flagged as unreliable and sits {gap[driver]:+.2f} from the median site")
        elif stretched and big_gap:
            fits.append("weak_factor")
            evidence.append(f"{driver} gap {gap[driver]:+.2f} comes from percentile stretching")
        if r["pv_flag"] == "yes" and big_gap and any(k in driver for k in cfg["pv_distorted_factors"]):
            fits.append("solar")
            evidence.append(f"PV flag set ({r['pv_text']}) and {driver} is one that PV distorts")
        if fwd_dir and pd.notna(r["kw_above_base_window"]) and r["kw_above_base_window"] < kw_cut:
            fits.append("size")
            evidence.append(f"kw_above_base_window {r['kw_above_base_window']:.1f} kW is below the "
                            f"{cfg['size_percentile']:.0%} quantile ({kw_cut:.1f} kW); the composite has no size factor")
        order = ["solar", "weak_factor", "size"]
        cls = next((c for c in order if c in fits), "unclear")
        if cls == "unclear":
            evidence.append("no rule fits" if not big_gap else
                            f"largest deviation is {driver} ({driver_gap:+.2f}) but it is not flagged unreliable, "
                            f"PV-distorted or explained by size")
        elif len(fits) > 1:
            evidence.append("also fits: " + ", ".join(f for f in fits if f != cls))
        rows.append({
            key: s, "direction": r["direction"], "selection_frequency": r["selection_frequency"],
            "composite_rank": r["composite_rank"], "rank_by_kw": r["rank_by_kw"],
            "kw_above_base_window": r["kw_above_base_window"], "pv_flag": r["pv_flag"],
            **{c: sc.get(c, np.nan) for c in scores},
            "driving_factor": driver, "driving_factor_gap_vs_median": driver_gap,
            "classification": cls, "evidence": "; ".join(evidence)})
    out = pd.DataFrame(rows)
    rule = [
        f"high frequency = selection frequency >= {hi:.2f} (the {cfg['high_freq_quantile']:.0%} quantile; ties included); "
        f"low frequency <= {lo:.2f}; composite bottom half = rank > {n_comp * cfg['composite_half']:.1f} of {n_comp}",
        f"driving factor = the score furthest below (above, for fragile-but-high) the median site; a gap counts from {cfg['min_factor_gap']:.2f}",
        "solar: PV flag set and the driving factor is PV-distorted (" + ", ".join(cfg["pv_distorted_factors"]) + ")",
        "weak_factor: the driving factor is flagged unreliable (" + ", ".join(cfg["unreliable_factors"])
        + f") or its raw values spread < {cfg['stretch_spread']:.0%} (percentile stretching)",
        f"size: robust-but-low-composite site with kw_above_base_window under the {cfg['size_percentile']:.0%} quantile",
        "several fit: solar > weak_factor > size (the others are listed in the evidence); none fits: unclear"]
    counts = out["classification"].value_counts().to_dict() if len(out) else {}
    return out, rule, {"counts": counts, "n_fwd": int(fwd.sum()), "n_rev": int(rev.sum())}


# ======================================================================
# charts
# ======================================================================
STYLE = {  # greyscale-readable: shape AND fill differ
    "final": dict(marker="o", fc="#222222", ec="#222222", s=70),
    "trimmed": dict(marker="v", fc="white", ec="#222222", s=70),
    "exception": dict(marker="*", fc="#888888", ec="#222222", s=190),
    "excluded": dict(marker="s", fc="#cccccc", ec="#666666", s=55),
    "passing (not final)": dict(marker="D", fc="#999999", ec="#222222", s=55),
}


def _label_points(ax, fig, xs, ys, names, fontsize=7.5):
    """Greedy label placement: try 8 offsets per label, keep the first that
    does not overlap an earlier label."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    placed = []
    offs = [(6, 5), (6, -11), (-6, 5), (-6, -11), (0, 9), (0, -15), (14, 0), (-14, 0), (10, 13), (10, -19)]
    order = np.lexsort((xs, -np.asarray(ys, float)))
    for i in order:
        best = None
        for dx, dy in offs:
            ha = "left" if dx >= 0 else "right"
            tx = ax.annotate(names[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                             fontsize=fontsize, ha=ha, va="bottom", color="#111111")
            bb = tx.get_window_extent(rend).expanded(1.04, 1.1)
            if not any(bb.overlaps(p) for p in placed):
                best = (tx, bb)
                break
            tx.remove()
        if best is None:                       # every spot taken: use the first anyway
            dx, dy = offs[0]
            tx = ax.annotate(names[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize)
            best = (tx, tx.get_window_extent(rend))
        placed.append(best[1])


def _save(fig, out_dir, stem, pdf=True):
    fig.savefig(os.path.join(out_dir, stem + ".png"), dpi=200, bbox_inches="tight")
    if pdf:
        fig.savefig(os.path.join(out_dir, stem + ".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def chart_robustness_value(cfg, t, names, out_dir, log=False, stem="robustness_vs_value"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = t.dropna(subset=["selection_frequency", "kw_above_base_window"]).copy()
    skipped = []
    if log:
        bad = d["kw_above_base_window"] <= 0
        skipped = d.loc[bad, cfg["site_key"]].tolist()
        d = d[~bad]
    if d.empty:
        return None
    yt_all = d["kw_above_base_window"].median() if cfg["y_threshold"] == "median" else float(cfg["y_threshold"])
    d_ne = d[~d["is_exception"]]
    yt_ne = (d_ne["kw_above_base_window"].median() if cfg["y_threshold"] == "median" else yt_all) if len(d_ne) else yt_all
    xt = cfg["x_threshold"]
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for st in STATUS_ORDER + ["passing (not final)"]:
        g = d[d["status"] == st]
        if g.empty:
            continue
        sty = STYLE[st]
        ax.scatter(g["selection_frequency"], g["kw_above_base_window"], marker=sty["marker"], s=sty["s"],
                   facecolors=sty["fc"], edgecolors=sty["ec"], linewidths=1.3, label=st, zorder=3)
    pv = d[d["pv_flag"] == "yes"]
    if len(pv):
        ax.scatter(pv["selection_frequency"], pv["kw_above_base_window"], marker="o", s=300, facecolors="none",
                   edgecolors="#222222", linewidths=1.2, linestyles="--", label="PV site (outer ring)", zorder=2)
    ax.axvline(xt, color="#555555", lw=1.0, ls="--", zorder=1)
    ax.axhline(yt_all, color="#555555", lw=1.0, ls="--", zorder=1)
    if abs(yt_ne - yt_all) > 1e-9:
        ax.axhline(yt_ne, color="#555555", lw=0.9, ls=":", zorder=1)
    if log:
        ax.set_yscale("log")
    ax.set_xlim(-0.04, 1.08)
    lo, hi = d["kw_above_base_window"].min(), d["kw_above_base_window"].max()
    if log:
        ax.set_ylim(lo / 1.8, hi * 2.2)
    else:
        ax.set_ylim(min(0, lo * 1.1), hi * 1.12 if hi > 0 else 1)
    y0, y1 = ax.get_ylim()
    ylab = lambda f: (math.sqrt(y0 * y1) if log else (y0 + y1) / 2) if f is None else f
    # quadrant names in the corners
    kw = dict(fontsize=9, color="#444444", style="italic", transform=ax.transAxes)
    ax.text(0.99, 0.985, QUADRANT_NAMES[("robust", "large")], ha="right", va="top", **kw)
    ax.text(0.99, 0.015, QUADRANT_NAMES[("robust", "small")], ha="right", va="bottom", **kw)
    ax.text(0.01, 0.985, QUADRANT_NAMES[("fragile", "large")], ha="left", va="top", **kw)
    ax.text(0.01, 0.015, QUADRANT_NAMES[("fragile", "small")], ha="left", va="bottom", **kw)
    thr_txt = f"x threshold {xt:g}; y threshold {yt_all:,.1f} kW ({'median of all screened sites' if cfg['y_threshold'] == 'median' else 'fixed'})"
    if abs(yt_ne - yt_all) > 1e-9:
        thr_txt += f"\ndotted: {yt_ne:,.1f} kW (median without the exception site)"
    ax.text(0.5, 1.012, thr_txt, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_xlabel("Selection frequency (share of the 81 filter settings the site passes)")
    ax.set_ylabel("Mean load above base in the constrained window, winter (kW)" + (" - log scale" if log else ""))
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False, title="Status")
    _label_points(ax, fig, d["selection_frequency"].to_numpy(), d["kw_above_base_window"].to_numpy(),
                  [names[s] for s in d[cfg["site_key"]]])
    _save(fig, out_dir, stem)
    return {"x": xt, "y_all": float(yt_all), "y_no_exception": float(yt_ne), "skipped_log": skipped}


def chart_composite_vs_freq(cfg, t, names, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = t.dropna(subset=["selection_frequency", "composite_rank"]).copy()
    if d.empty:
        return False
    n = len(d)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for st in STATUS_ORDER + ["passing (not final)"]:
        g = d[d["status"] == st]
        if g.empty:
            continue
        sty = STYLE[st]
        ax.scatter(g["composite_rank"], g["selection_frequency"], marker=sty["marker"], s=sty["s"] * 0.9,
                   facecolors=sty["fc"], edgecolors=sty["ec"], linewidths=1.2, label=st, zorder=3)
    fmax, fmin = d["selection_frequency"].max(), d["selection_frequency"].min()
    ax.plot([1, d["composite_rank"].max()], [fmax, fmin], color="#777777", lw=1.0, ls="--", zorder=1,
            label="agreement (straight line, best to worst)")
    srt = np.sort(d["selection_frequency"].to_numpy())[::-1]
    ax.step(np.arange(1, n + 1), srt, where="mid", color="#aaaaaa", lw=1.0, zorder=1,
            label="agreement (exact: frequencies in rank order)")
    ax.set_xlabel("Composite rank (1 = best)")
    ax.set_ylabel("Selection frequency")
    ax.set_ylim(-0.04, 1.08)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    _label_points(ax, fig, d["composite_rank"].to_numpy(), d["selection_frequency"].to_numpy(),
                  [names[s] for s in d[cfg["site_key"]]])
    _save(fig, out_dir, "composite_vs_robustness")
    return True


def chart_frequency_distribution(cfg, t, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = t["selection_frequency"].dropna()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.linspace(0, 1, 11)
    ax.hist(f, bins=bins, color="#888888", edgecolor="#222222")
    at1 = int((f >= 0.9999).sum())
    ax.axvline(cfg["x_threshold"], color="#444444", ls="--", lw=1)
    ax.set_xlabel("Selection frequency")
    ax.set_ylabel("Number of sites")
    ax.set_title(f"{len(f)} sites; {at1} at 100 %; dashed line = x threshold {cfg['x_threshold']:g}", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(os.path.join(out_dir, "selection_frequency_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# main
# ======================================================================
def quadrant(cfg, f, kw, y_thr):
    return QUADRANT_NAMES[("robust" if f >= cfg["x_threshold"] else "fragile", "large" if kw >= y_thr else "small")]


def main(cfg=CONFIG, rerun=False):
    key = cfg["site_key"]
    if rerun:
        import filter_sensitivity as FS
        FS.main(FS.CONFIG)
    freq, marg, met, runs, cum, comp = load_inputs(cfg)
    composite, comp_how = composite_source(cfg, comp)
    t = build_table(cfg, freq, marg, met, runs, cum, comp, composite)
    mapping, an = build_anonymiser(list(t[key]), cfg)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["private_dir"], exist_ok=True)
    # key (private)
    kdf = pd.DataFrame({"real_site_id": list(mapping), "anonymised_id": list(mapping.values())})
    kdf.to_csv(os.path.join(cfg["private_dir"], "site_id_key.csv"), sep=cfg["csv_sep"], index=False, encoding="utf-8-sig")
    # table (anonymised)
    t = t.sort_values(["selection_frequency", "kw_above_base_window"], ascending=[False, False], kind="mergesort")
    names = dict(mapping)
    out = t.copy()
    out[key] = out[key].map(names)
    cols = [key, "selection_frequency", "kw_above_base_window", "rank_by_kw", "cumulative_share_without_exception",
            "cumulative_share_with_exception", "composite_rank", "rank_difference", "freq_rank", "pv_text",
            "tightest_margin", "status", "winter_coverage"]
    notes = out[["kw_na_reason", "cumulative_share_note", "composite_note"]].apply(
        lambda r: "; ".join(x for x in r if isinstance(x, str) and x), axis=1)
    tab = out[cols].rename(columns={"pv_text": "pv_flag"})
    tab["na_reasons"] = notes
    tab.to_csv(os.path.join(cfg["output_dir"], "site_robustness_value_table.csv"), sep=cfg["csv_sep"],
               decimal=cfg["csv_decimal"], index=False, encoding="utf-8-sig")
    # Part 2
    dis, rule, dcount = disagreements(cfg, t, composite, comp)
    if dis is not None:
        d2 = dis.copy()
        d2[key] = d2[key].map(names)
        d2.to_csv(os.path.join(cfg["output_dir"], "site_disagreements.csv"), sep=cfg["csv_sep"],
                  decimal=cfg["csv_decimal"], index=False, encoding="utf-8-sig")
    # charts
    q1 = chart_robustness_value(cfg, t, names, cfg["output_dir"], log=(cfg["y_scale"] == "log"))
    ratio = t["kw_above_base_window"].max() / max(t["kw_above_base_window"].median(), 1e-9)
    if cfg["also_log_version"] and cfg["y_scale"] != "log" and ratio > cfg["log_if_ratio_over"]:
        chart_robustness_value(cfg, t, names, cfg["output_dir"], log=True, stem="robustness_vs_value_log")
    made2 = chart_composite_vs_freq(cfg, t, names, cfg["output_dir"]) if composite is not None else False
    chart_frequency_distribution(cfg, t, cfg["output_dir"])
    # note
    f = t["selection_frequency"].dropna()
    n = len(f)
    at1 = int((f >= 0.9999).sum())
    near = int((f >= 0.9).sum())
    mid = int(((f > 0.1) & (f < 0.9)).sum())
    zero = int((f <= 0.0001).sum())
    sep = ("separates poorly: most sites sit near the top" if near / max(n, 1) > 0.5 else
           "separates reasonably: most sites are not at the top" if mid >= 0.25 * n else
           "mostly sites at the extremes (near 100 % or near 0 %)")
    fin = t[t["status"].isin(["final", "exception"])].dropna(subset=["selection_frequency", "kw_above_base_window"])
    quad = {}
    for _, r in fin.iterrows():
        quad.setdefault(quadrant(cfg, r["selection_frequency"], r["kw_above_base_window"], q1["y_all"] if q1 else 0),
                        []).append(names[r[key]] + ("*" if r["status"] == "exception" else ""))
    doubts = []
    na_kw = t[t["kw_above_base_window"].isna()][key].map(names).tolist()
    if na_kw:
        doubts.append(f"kW missing: {', '.join(na_kw)}")
    cov = t[t["winter_coverage"] < 0.8][key].map(names).tolist()
    if cov:
        doubts.append(f"winter coverage < 80 %: {', '.join(cov)}")
    if composite is None:
        doubts.append("composite not found, so Part 2 and the second chart are missing: " + comp_how)
    L = ["# Robustness against value -- note", ""]
    if dis is not None:
        L.append(f"1. Disagreements: {len(dis)} of {len(t)} sites ({dcount['n_fwd']} robust but low composite, "
                 f"{dcount['n_rev']} fragile but high composite); composite taken from {comp_how}.")
        L.append("2. Classes: " + (", ".join(f"{k} {v}" for k, v in sorted(dcount["counts"].items())) or "none")
                 + ". Rule and evidence: site_disagreements_rule.md and site_disagreements.csv.")
    else:
        L.append("1. Disagreements: not computed (no composite in the data).")
        L.append("2. Classes: not computed.")
    L.append(f"3. Selection frequency: {at1} of {n} sites at 100 %, {near} at 0.90 or more, {mid} between 0.10 and 0.90, "
             f"{zero} at 0; the measure {sep}.")
    ql = "; ".join(f"{q}: {', '.join(v)}" for q, v in quad.items())
    L.append(f"4. Final group (* = exception) by quadrant: {ql or 'n/a'}.")
    if q1:
        L.append(f"5. Lines: x = {q1['x']:g}; y = {q1['y_all']:,.1f} kW (median, all screened sites) and "
                 f"{q1['y_no_exception']:,.1f} kW (median without the exception site).")
    L.append("6. Doubtful or missing: " + ("; ".join(doubts) if doubts else "none") + ".")
    L.append("7. Quadrant, rank and class are descriptions; no site is recommended or dropped here.")
    if dis is not None:
        with open(os.path.join(cfg["output_dir"], "site_disagreements_rule.md"), "w", encoding="utf-8") as fh:
            fh.write("# Rule for the classes in Part 2\n\n" + "\n".join(f"- {r}" for r in rule) + "\n")
    with open(os.path.join(cfg["output_dir"], "robustness_value_note.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[robustness] outputs in {os.path.abspath(cfg['output_dir'])}; key (PRIVATE) in "
          f"{os.path.abspath(cfg['private_dir'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun-sensitivity", action="store_true")
    a = ap.parse_args()
    main(CONFIG, a.rerun_sensitivity)

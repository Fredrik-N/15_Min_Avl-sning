"""
Portfolio analysis: how much flexible load can a GROUP of cold-storage
sites reliably offer during the grid's constrained hours -- from the 15-min
meter data of all screened sites, from the 1-second disaggregation of the
case sites, and a comparison of the two.

One core (analyse_portfolio) works on a sites x time table of flexible kW.
Two adapters build that table:

  15-min   sites that pass the lpc_8 screening (site_comparison.csv), plus
           any verified cold stores added by hand (CONFIG["screening"]).
           Flexible kW = net exchange (import - export) minus the site's
           base load (p5 of its WINTER net load), clipped at 0.
  1-second the case sites' unilm_fhmm output folders (appliances.csv,
           switches.csv, timeseries_*.csv from `unilm_fhmm.py METER --out DIR`).
           Flexible kW = sum of rated kW of the diagnosed compressors that
           are on (a rack: rated kW per unit x units running).

Both put every site on one time axis (tz-aware, CONFIG["time_zone"]),
leave gaps missing (never imputed) and record resolution, period and
coverage per site.

Core analysis, inside the constrained window (the grid's top hours in
winter, derived from Öresundskraft's grid load exactly as in lpc_8):
  A  bundling: pairwise correlation of flexible kW (raw series and
     hour-of-day profiles); candidate bundles = groups with high
     correlation whose combined mean clears a kW threshold, flagged
     fragile if they clear it on too few days.
  B  reliability floor: p1 / p5 / p10 of the group's total flexible kW,
     plus each site left out in turn (sites whose removal barely moves
     the floor are flagged).
  C  coverage curve: sites by mean window flexible kW, cumulative share.

This is a GROUP-level FLOOR. It is not the per-site theoretical
potential (a per-site ceiling) and is not written into site_comparison.csv.

Run (from the folder with lpc_8.py, which supplies the meter reading,
grid file, seasons and site_comparison.csv settings):
    python portfolio.py --mode 15min
    python portfolio.py --mode 1s
    python portfolio.py --mode compare
Options:
    --full-day        use all 24 hours instead of the constrained window
    --config X.json   settings that override CONFIG below (same keys)
Outputs go to CONFIG["output_dir"]:
    portfolio_15min.csv, portfolio_1s.csv, portfolio_15min_case.csv
    portfolio_comparison.csv, portfolio_summary.json, portfolio_note.md
    availability_*.png, correlation_*.png, coverage_*.png, compare_scatter.png
"""
import argparse
import copy
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import lpc_8 as L

CONFIG = {
    "output_dir": "portfolio_out",

    # ---- time axis ----
    # Everything is put on one tz-aware axis in this zone. Hours of the
    # constraint window are local hours.
    "time_zone": "Europe/Stockholm",
    # How to read the naive timestamps of the meter exports and the grid
    # file. "Tidspunkt (CET)" may mean fixed CET (UTC+1 all year) or local
    # clock time with daylight saving:
    #   "auto"  -- look at the DST change days: readings between 02:00 and
    #              03:00 on the last Sunday of March exist only on a fixed-
    #              CET clock; a doubled 02:00-03:00 hour on the last Sunday
    #              of October exists only on a local clock. Falls back to
    #              "local" (and says so) when the data shows neither.
    #   "local" -- wall-clock time in time_zone. The doubled autumn hour is
    #              resolved by order (first = summer time); a single reading
    #              in that hour cannot be placed and is left out.
    #   "cet"   -- fixed UTC+1.
    # Timestamps are taken as the START of each interval (as in lpc_8).
    "meter_time_convention": "auto",
    "grid_time_convention": "auto",

    # ---- which sites (15-min) ----
    "screening": {
        # "filter_pass": every site passing lpc_8's stage-1 filter;
        # "selected": only lpc_8's stage-2 selection; "all": every site.
        "include": "filter_pass",
        # verified cold stores to include even if they fail the filter:
        # {site_id: why}
        "add_sites": {},
        # sites to leave out even if they pass: {site_id: why}
        "exclude_sites": {},
    },
    # Mängdkod value(s) of an EXPORT channel in the meter file, if the
    # export is a separate channel (net = import - export). [] = the kWh
    # channel already is the net exchange (negative when exporting) or the
    # site never exports.
    "export_channel_values": [],

    # ---- flexible kW from 15-min data ----
    "base_load": {
        "percentile": 5,             # p5 of the winter net load
        "months": None,              # None = lpc_8's winter (temp_corr_season_months["winter"])
        "min_quarter_hours": 960,    # at least 10 days of winter data; otherwise
                                     # the base load comes from all months and is FLAGGED
    },

    # ---- the constrained window ----
    "window": {
        "hours": None,               # None = derive from the grid's load (as lpc_8);
                                     # or fix it, e.g. [16, 17, 18, 19]
        "derive_months": None,       # months of grid load used to derive it (None = winter)
        "months": None,              # months analysed in --mode 15min (None = winter)
        "months_1s": [],             # months analysed in --mode 1s / compare: [] = whatever
                                     # the 1-second data covers (it is short)
        "weekdays_only": False,
    },

    # ---- 1-second case sites ----
    # One entry per case site:
    #   "site_id":          name used in the outputs
    #   "unilm_output_dir": folder written by `unilm_fhmm.py METER --out DIR`
    #   "meter_site_id":    the same site's row in site_comparison.csv (its
    #                       15-min meter file is used in --mode compare), or
    #   "meter_path":       the 15-min meter file directly
    "case_sites": [],
    "unilm": {
        "resolution_s": 1,               # 1 = native; e.g. 10 averages to 10 s (memory)
        "min_quarter_coverage": 0.9,     # a quarter-hour of 1 s data counts only if
                                         # >= 90 % of its seconds are present
        "max_power_factor": 0.95,        # compressors are inductive
    },

    # ---- A: correlation and bundles ----
    "correlation": {
        "min_overlap_hours": 24,     # a pair needs this many shared window hours
        "profile_bin": "hour",       # "hour" or "quarter" (of the day) for the profile correlation
    },
    "bundles": {
        "corr_basis": "raw",         # "raw" or "profile": which correlation groups sites
        "min_corr": 0.5,             # average-linkage: merge groups while their mean pairwise r >= this
        "min_kw": 100.0,             # a bundle's combined mean window flexible kW must reach this
        "min_day_share": 0.9,        # ... and its daily window mean must reach it on >= 90 % of
                                     # days; otherwise the bundle is flagged fragile
    },

    # ---- B: reliability floor ----
    "floor": {
        "percentiles": [1, 5, 10],
        "leave_one_out_percentile": 5,
        "loo_flag_share": 0.25,      # flag a site whose removal lowers the floor by less than
                                     # 25 % of its own mean window flexible kW
        "min_window_hours": 200,     # fewer window hours behind a floor -> flagged unstable
    },

    # ---- solar ----
    "solar": {
        "midday_dip": None,          # None = lpc_8's pv_hint_midday_dip (0.15)
        "export_share": 0.01,        # net <= 0 in more than 1 % of quarter-hours
    },

    # Overrides applied on top of lpc_8.CONFIG (its meter format, grid
    # file, comparison file, seasons, csv separator). Leave {} to use
    # lpc_8.py's own settings, e.g.
    # {"results_csv_path": "site_comparison.csv",
    #  "grid_consumption": {"csv_path": "natbelastning.xlsx"}}
    "lpc_overrides": {},
}


# ======================================================================
# settings
# ======================================================================
def _deep_update(base, new):
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def lpc_config(cfg):
    lc = copy.deepcopy(L.CONFIG)
    return _deep_update(lc, cfg.get("lpc_overrides"))


def winter_months(lcfg):
    return list((lcfg.get("temp_corr_season_months") or {}).get("winter", [11, 12, 1, 2]))


# ======================================================================
# time axis
# ======================================================================
def _last_sunday(year, month):
    d = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.dayofweek + 1) % 7)


def detect_time_convention(naive_index):
    """'cet', 'local' or None (no DST change day in the data) from naive
    timestamps that still contain their duplicates."""
    idx = pd.DatetimeIndex(naive_index)
    if not len(idx):
        return None, "no timestamps"
    cet = local = 0
    for y in sorted(set(idx.year)):
        sp = _last_sunday(y, 3)
        fa = _last_sunday(y, 10)
        on_sp = idx[(idx >= sp) & (idx < sp + pd.Timedelta(days=1))]
        if len(on_sp):
            gap = (on_sp >= sp + pd.Timedelta(hours=2)) & (on_sp < sp + pd.Timedelta(hours=3))
            around = ((on_sp >= sp + pd.Timedelta(hours=1)) & (on_sp < sp + pd.Timedelta(hours=2))).any() \
                and ((on_sp >= sp + pd.Timedelta(hours=3)) & (on_sp < sp + pd.Timedelta(hours=4))).any()
            if gap.any():
                cet += 1
            elif around:
                local += 1
        on_fa = idx[(idx >= fa + pd.Timedelta(hours=2)) & (idx < fa + pd.Timedelta(hours=3))]
        if len(on_fa) and on_fa.duplicated().any():
            local += 1
    if cet and not local:
        return "cet", "readings exist in the hour skipped by daylight saving in March"
    if local and not cet:
        return "local", "the March hour is skipped and/or the October hour is doubled"
    if cet and local:
        return "local", "mixed evidence on DST days; taken as local clock time"
    return None, "no daylight-saving change day in the data"


def localize(values, naive_index, convention, tz):
    """Naive interval-start timestamps -> tz-aware Series in tz. Returns
    (series, info). Never invents values: readings that cannot be placed
    are dropped and counted."""
    idx = pd.DatetimeIndex(naive_index)
    s = pd.Series(np.asarray(values, float), index=idx)
    s = s[s.index.notna()]
    order = np.argsort(s.index.values, kind="stable")
    s = s.iloc[order]
    info = {"time_convention_setting": convention}
    if convention == "auto":
        found, why = detect_time_convention(s.index)
        info["time_convention_detected"] = found
        info["time_convention_reason"] = why
        convention = found or "local"
    info["time_convention"] = convention
    if convention == "cet":
        loc = s.index.tz_localize("Etc/GMT-1").tz_convert(tz)
        out = pd.Series(s.to_numpy(), index=loc)
    else:
        n = len(s)
        a1 = s.index.tz_localize(tz, ambiguous=np.ones(n, bool), nonexistent="NaT")
        a0 = s.index.tz_localize(tz, ambiguous=np.zeros(n, bool), nonexistent="NaT")
        amb = (a1 != a0) & a1.notna() & a0.notna()
        dup_any = s.index.duplicated(keep=False)
        dup_second = s.index.duplicated(keep="first")
        if amb.any():
            # the doubled autumn hour: first reading = summer time, second = winter time
            ns = np.where(amb & dup_second, a0.as_unit("ns").asi8, a1.as_unit("ns").asi8)
            loc = pd.DatetimeIndex(ns.view("M8[ns]")).tz_localize("UTC").tz_convert(tz)
        else:
            loc = a1
        single_amb = amb & ~dup_any
        keep = loc.notna() & ~single_amb
        info["dropped_nonexistent"] = int(a1.isna().sum())
        info["dropped_ambiguous_single"] = int(single_amb.sum())
        out = pd.Series(s.to_numpy()[keep], index=loc[keep])
    n_dup = int(out.index.duplicated().sum())
    out = out[~out.index.duplicated(keep="first")]
    info["dropped_duplicates"] = n_dup
    return out.sort_index(), info


# ======================================================================
# 15-min adapter
# ======================================================================
def _values(rows, col):
    return L._to_float(rows[col]).to_numpy()


def load_meter_net(path, lcfg, cfg):
    """Net exchange (import - export) in average kW per 15 min, tz-aware,
    gaps NaN. Returns (series, info)."""
    tz = cfg["time_zone"]
    df = L._read_table(path)
    tcol = lcfg["timestamp_col"]
    if lcfg.get("timestamp_unit"):
        df[tcol] = pd.to_datetime(df[tcol], unit=lcfg["timestamp_unit"], errors="coerce")
    else:
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df[df[tcol].notna()]
    exp = None
    if lcfg.get("data_format", "wide") == "long":
        lf = lcfg["long_format"]
        rows = L._select_channel_rows(df, lf["channel_col"], lf["kwh_channel_values"])
        imp_t, imp_v = rows[tcol].to_numpy(), _values(rows, lf["value_col"])
        if cfg.get("export_channel_values"):
            er = L._select_channel_rows(df, lf["channel_col"], cfg["export_channel_values"])
            exp = (er[tcol].to_numpy(), _values(er, lf["value_col"]))
        energy = lf.get("value_kind", "energy_per_interval_kwh") == "energy_per_interval_kwh"
        divisor = 1.0 if energy else lcfg.get("power_divisor", 1.0)
    else:
        agg = pd.to_numeric(df[lcfg["aggregate_col"]], errors="coerce")
        for c in lcfg.get("subtract_cols", []):
            if c in df:
                agg = agg - pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        imp_t, imp_v = df[tcol].to_numpy(), agg.to_numpy()
        energy, divisor = False, lcfg.get("power_divisor", 1.0)
    if not len(imp_t):
        raise ValueError(f"no import/active rows in {path}")
    imp, info = localize(imp_v, imp_t, cfg["meter_time_convention"], tz)
    step_h = L._infer_interval_hours(imp.index.tz_convert("UTC").tz_localize(None))
    scale = (1.0 / step_h) if energy else (1.0 / divisor)
    net = imp * scale
    if exp is not None:
        e, _ = localize(exp[1], exp[0], info["time_convention"], tz)
        net = net - (e * scale).reindex(net.index)          # export missing -> net missing
        info["export_channel"] = True
    q = pd.Timedelta("15min")
    if step_h * 3600 > q.total_seconds() * 1.01:
        # coarser than 15 min: each reading is the average over its interval
        n = int(round(step_h * 4))
        net = net.resample("15min").asfreq().ffill(limit=n - 1)
    else:
        net = net.resample("15min").mean()
    info["native_interval_min"] = round(step_h * 60, 2)
    return net.rename("net_kw"), info


def solar_flag(net, lcfg, cfg):
    """'' or the reason this site looks like it has solar."""
    m = L.winter_and_hint_metrics(net.dropna(), None, lcfg)
    thr = cfg["solar"]["midday_dip"]
    if thr is None:
        thr = (lcfg.get("selection") or {}).get("pv_hint_midday_dip", 0.15)
    why = []
    ds, dw = m.get("pv_midday_dip_summer"), m.get("pv_midday_dip_winter")
    if ds is not None and dw is not None and ds - dw >= thr:
        why.append(f"summer midday dip {ds:.0%} vs winter {dw:.0%}")
    elif ds is not None and dw is None and ds >= thr:
        why.append(f"midday dip {ds:.0%} (no winter data to compare)")
    z = m.get("share_intervals_at_or_below_zero")
    if z is not None and z > cfg["solar"]["export_share"]:
        why.append(f"net <= 0 in {z:.1%} of quarter-hours")
    return "; ".join(why)


def base_load(net, lcfg, cfg):
    bl = cfg["base_load"]
    months = bl["months"] or winter_months(lcfg)
    w = net[net.index.month.isin(months)].dropna()
    if len(w) >= bl["min_quarter_hours"]:
        return float(w.quantile(bl["percentile"] / 100)), f"p{bl['percentile']} of winter net load ({len(w)} quarter-hours)"
    v = net.dropna()
    if not len(v):
        return None, "no data"
    return float(v.quantile(bl["percentile"] / 100)), \
        (f"FLAG: only {len(w)} winter quarter-hours -- p{bl['percentile']} of ALL months used "
         f"(solar or summer load may bias it)")


def screened_sites(cfg, lcfg):
    """[(site_id, meter path or None, why included)], [(site_id, why excluded)]"""
    df = L.read_comparison(lcfg)
    if df is None:
        raise FileNotFoundError(f"No comparison file at {lcfg.get('results_csv_path')!r}; run lpc_8.py first.")
    sc = cfg["screening"]
    inc_rule = sc.get("include", "filter_pass")
    add = {str(k).strip().lower(): (k, v) for k, v in (sc.get("add_sites") or {}).items()}
    excl = {str(k).strip().lower(): v for k, v in (sc.get("exclude_sites") or {}).items()}
    included, excluded = [], []
    seen = set()
    for _, r in df.iterrows():
        sid = str(r[L.SITE_KEY]).strip()
        key = sid.lower()
        seen.add(key)
        passed = str(r.get("filter_pass")).lower() in ("true", "1", "1.0")
        chosen = str(r.get("selected")).lower() in ("true", "1", "1.0")
        if key in excl:
            excluded.append((sid, f"excluded by hand: {excl[key]}"))
            continue
        if inc_rule == "all":
            ok, why = True, "all sites requested"
        elif inc_rule == "selected":
            ok, why = chosen, "selected by lpc_8 (stage 2)"
        else:
            ok, why = passed, "passes the lpc_8 screening filter"
        if not ok and key in add:
            ok, why = True, f"added by hand: {add[key][1]}"
        if not ok:
            reason = r.get("filter_fail_reasons")
            excluded.append((sid, "fails the screening filter: " + (str(reason) if isinstance(reason, str) and reason
                                                                       else "not selected")))
            continue
        path = L.find_meter_file(r, lcfg)
        if path is None:
            excluded.append((sid, "meter file not found (set meter_folder in lpc_8.py)"))
            continue
        included.append((sid, path, why))
    for key, (sid, why) in add.items():
        if key not in seen:
            excluded.append((sid, f"listed in add_sites ({why}) but not in {lcfg.get('results_csv_path')}"))
    return included, excluded


def build_15min(sites, cfg, lcfg, label=""):
    """sites: [(site_id, path, why)]. Returns (flex DataFrame, meta, excluded)."""
    cols, meta, excluded = {}, {}, []
    for sid, path, why in sites:
        try:
            net, info = load_meter_net(path, lcfg, cfg)
        except Exception as e:                                  # one bad file must not stop the rest
            excluded.append((sid, f"meter file unreadable: {e}"))
            continue
        b, b_why = base_load(net, lcfg, cfg)
        if b is None:
            excluded.append((sid, "no data in the meter file"))
            continue
        flex = (net - b).clip(lower=0)
        valid = net.dropna()
        span = pd.date_range(valid.index.min(), valid.index.max(), freq="15min")
        cols[sid] = flex
        meta[sid] = {"included_because": why, "meter_file": os.path.basename(path), "resolution": "15 min",
                     "period_start": str(valid.index.min()), "period_end": str(valid.index.max()),
                     "coverage_fraction": round(len(valid) / len(span), 4) if len(span) else None,
                     "base_load_kw": round(b, 2), "base_load_basis": b_why,
                     "solar_flag": solar_flag(net, lcfg, cfg), **{f"time_{k}": v for k, v in info.items()}}
        print(f"[15-min{label}] {sid}: {len(valid)} quarter-hours, coverage {meta[sid]['coverage_fraction']:.1%}, "
              f"base {b:.1f} kW ({b_why}), time {info['time_convention']}"
              + (f", SOLAR: {meta[sid]['solar_flag']}" if meta[sid]["solar_flag"] else ""))
    if not cols:
        raise ValueError("No site could be loaded.")
    flex = pd.concat(cols, axis=1).sort_index()
    flex = flex.reindex(pd.date_range(flex.index.min(), flex.index.max(), freq="15min"))
    return flex, meta, excluded


# ======================================================================
# 1-second adapter
# ======================================================================
def _compressor_rows(app, cfg):
    cat = app.get("category", pd.Series("refrigeration", index=app.index)).astype(str)
    role = app.get("role", pd.Series("", index=app.index)).fillna("").astype(str)
    pf = pd.to_numeric(app.get("power_factor"), errors="coerce")
    vsd = app.get("variable_speed_suspected", pd.Series(False, index=app.index)).astype(str).str.lower() == "true"
    kw = pd.to_numeric(app.get("rated_kW"), errors="coerce")
    keep = cat.str.startswith("refrigeration") & ~cat.str.contains("same compressor as") \
        & (pf < cfg["unilm"]["max_power_factor"]) & ~vsd & kw.notna() \
        & ~role.str.startswith(("defrost heater", "periodic inductive"))
    return app[keep].assign(rated_kW=kw[keep])


def load_unilm_site(case, cfg):
    """Flexible kW at 1 s from one unilm_fhmm output folder. Returns
    (series, info)."""
    d = case["unilm_output_dir"]
    if not os.path.isdir(d):
        raise FileNotFoundError(f"unilm output folder not found: {d}")
    app = pd.read_csv(os.path.join(d, "appliances.csv"))
    sw = pd.read_csv(os.path.join(d, "switches.csv"))
    tsf = sorted(glob.glob(os.path.join(d, "timeseries_*s.csv")))
    if not tsf:
        raise FileNotFoundError(f"no timeseries_*s.csv in {d}")
    ts = pd.read_csv(tsf[0], usecols=lambda c: c in ("time_local", "P_kW"))
    t_ts = pd.to_datetime(ts["time_local"], utc=True)
    t_s = _epoch_s(t_ts)
    r_ts = int(round(np.median(np.diff(t_s)))) if len(ts) > 1 else 60
    t0 = int(t_s[0])
    t1 = int(t_s[-1]) + r_ts
    sec = np.arange(t0, t1, dtype=np.int64)
    comp = _compressor_rows(app, cfg)
    flex = np.zeros(len(sec), np.float64)
    used = []
    for _, r in comp.iterrows():
        s = sw[sw["appliance"] == r["appliance"]].sort_values("time_utc_s")
        if s.empty:
            continue
        k = np.searchsorted(s["time_utc_s"].to_numpy(), sec, side="right") - 1
        st = np.where(k >= 0, s["new_state"].to_numpy()[np.clip(k, 0, None)], 0)
        flex += float(r["rated_kW"]) * st
        used.append(f"{r['appliance']} {float(r['rated_kW']):.1f} kW" + (f" x{int(r['units'])}" if r.get("units", 1) > 1 else ""))
    # gaps: the minutes where the recording itself has no data
    gap_min = t_s[ts["P_kW"].isna().to_numpy()]
    for g in gap_min:
        flex[max(0, g - t0): max(0, g - t0 + r_ts)] = np.nan
    s = pd.Series(flex, index=pd.to_datetime(sec, unit="s", utc=True).tz_convert(cfg["time_zone"]))
    res = int(cfg["unilm"].get("resolution_s", 1))
    if res > 1:
        s = s.resample(f"{res}s").mean()
    cat_all = app.get("category", pd.Series("", index=app.index)).fillna("").astype(str)
    pv_rows = app.loc[cat_all.str.contains("PV-like"), "appliance"].tolist()
    info = {"resolution": f"{res} s", "unilm_output_dir": d,
            "solar_flag": (f"unilm found PV-like steps ({', '.join(map(str, pv_rows))})" if pv_rows else "")
            + (" + " if pv_rows and case.get("solar_flag") else "") + (case.get("solar_flag") or ""),
            "period_start": str(s.index[0]), "period_end": str(s.index[-1]),
            "coverage_fraction": round(float(s.notna().mean()), 4),
            "compressors": "; ".join(used), "n_compressors": len(used),
            "rated_total_kw": round(float(comp["rated_kW"].sum()), 1)}
    print(f"[1-s] {case['site_id']}: {len(s):,} samples ({s.index[0]} .. {s.index[-1]}), "
          f"coverage {info['coverage_fraction']:.1%}, {len(used)} compressors: {info['compressors']}")
    return s, info


def build_1s(cfg):
    if not cfg["case_sites"]:
        raise ValueError('CONFIG["case_sites"] is empty: list the 1-second case sites and their unilm output folders.')
    cols, meta, excluded = {}, {}, []
    for case in cfg["case_sites"]:
        try:
            s, info = load_unilm_site(case, cfg)
        except Exception as e:
            excluded.append((case["site_id"], f"1-second results unreadable: {e}"))
            print(f"[1-s] {case['site_id']}: LEFT OUT -- {e}")
            continue
        cols[case["site_id"]] = s
        meta[case["site_id"]] = {"included_because": "1-second case site", **info}
    if not cols:
        raise ValueError("No 1-second case site could be loaded.")
    return pd.concat(cols, axis=1).sort_index(), meta, excluded


def to_15min(flex, cfg):
    """Mean per quarter-hour; a quarter-hour with too few samples is missing."""
    step = _step_s(flex.index)
    need = cfg["unilm"]["min_quarter_coverage"] * 900 / step
    m = flex.resample("15min").mean()
    n = flex.notna().resample("15min").sum()
    return m.where(n >= need)


# ======================================================================
# window
# ======================================================================
def load_grid(cfg, lcfg):
    gc = lcfg.get("grid_consumption") or {}
    if not gc.get("csv_path"):
        return None
    g = L.load_grid_consumption(lcfg)
    s, info = localize(g.to_numpy(), g.index, cfg["grid_time_convention"], cfg["time_zone"])
    print(f"[grid] time convention: {info['time_convention']}"
          + (f" ({info.get('time_convention_reason')})" if info.get("time_convention_reason") else ""))
    return s


def window_hours(cfg, lcfg, grid):
    w = cfg["window"]
    if w.get("hours"):
        return sorted(w["hours"]), "fixed in CONFIG['window']['hours']"
    if grid is None:
        raise ValueError("No constraint window: set lpc_8's grid_consumption csv_path, or CONFIG['window']['hours'], "
                         "or run with --full-day.")
    months = w.get("derive_months") or winter_months(lcfg)
    g = grid[grid.index.month.isin(months)].dropna()
    if g.empty:
        raise ValueError(f"The grid file has no data in months {months} to derive the window from.")
    hourly = g.groupby(g.index.hour).mean()
    p = lcfg["grid_consumption"]["constraint_percentile"]
    hours = sorted(hourly[hourly >= hourly.quantile(p)].index.tolist())
    return hours, f"grid load months {months}, hours at or above the {p:.0%} quantile of its hour-of-day profile"


def window_mask(index, hours, months, weekdays_only):
    m = np.ones(len(index), bool)
    if hours is not None:
        m &= np.isin(index.hour, hours)
    if months:
        m &= np.isin(index.month, months)
    if weekdays_only:
        m &= index.dayofweek < 5
    return m


# ======================================================================
# core
# ======================================================================
def _step_s(index):
    return float(np.median(np.diff(index[:1000].as_unit("ns").asi8)) / 1e9) if len(index) > 1 else 900.0


def _epoch_s(t):
    """Whole UTC seconds since 1970 of a tz-aware Series/Index (any pandas time unit)."""
    t = pd.Series(pd.DatetimeIndex(t))
    return ((t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")).to_numpy(np.int64)


def _cluster(corr, min_corr):
    """Average-linkage grouping on a correlation matrix: merge the two
    groups with the highest mean pairwise r while it is >= min_corr."""
    groups = [[s] for s in corr.index]

    def link(a, b):
        v = corr.loc[a, b].to_numpy().ravel()
        v = v[~np.isnan(v)]
        return v.mean() if len(v) == len(a) * len(b) else -np.inf
    while len(groups) > 1:
        best, pair = -np.inf, None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                v = link(groups[i], groups[j])
                if v > best:
                    best, pair = v, (i, j)
        if pair is None or best < min_corr:
            break
        i, j = pair
        groups[i] = groups[i] + groups[j]
        del groups[j]
    return groups


def _floor(W, percentiles):
    C = W.dropna(how="any")
    tot = C.sum(axis=1)
    return {p: (float(tot.quantile(p / 100)) if len(tot) else None) for p in percentiles}, tot


def analyse_portfolio(flex, meta, cfg, label, hours, months):
    """The core: same for every input. flex = sites x time, flexible kW,
    gaps NaN, regular tz-aware index. Returns (result dict, per-site table)."""
    fc, bc, cc = cfg["floor"], cfg["bundles"], cfg["correlation"]
    step = _step_s(flex.index)
    mask = window_mask(flex.index, hours, months, cfg["window"]["weekdays_only"])
    W = flex[mask]
    n = W.shape[1]
    h_per = step / 3600.0
    res = {"label": label, "n_sites": n, "resolution_s": step,
           "window_hours_of_day": hours, "window_months": months or "all covered",
           "window_timestamps": int(len(W)), "window_hours_total": round(len(W) * h_per, 1)}

    # per site
    rows = []
    for s in W.columns:
        x = W[s].dropna()
        rows.append({"site_id": s, **{k: v for k, v in meta.get(s, {}).items()},
                     "window_hours_with_data": round(len(x) * h_per, 1),
                     "window_coverage": round(len(x) / len(W), 4) if len(W) else None,
                     "mean_window_flex_kw": round(float(x.mean()), 2) if len(x) else None,
                     "p5_window_flex_kw": round(float(x.quantile(0.05)), 2) if len(x) else None,
                     "share_window_time_zero_flex": round(float((x <= 0).mean()), 3) if len(x) else None})
    site = pd.DataFrame(rows).set_index("site_id")

    # A. correlation
    minp = max(3, int(cc["min_overlap_hours"] / h_per))
    raw = W.corr(min_periods=minp)
    if cc.get("profile_bin") == "quarter":
        key = W.index.hour * 4 + W.index.minute // 15
    else:
        key = W.index.hour
    prof = W.groupby(key).mean()
    prof_c = prof.corr(min_periods=3)
    res["corr_raw"] = raw.round(3).to_dict()
    res["corr_profile"] = prof_c.round(3).to_dict()
    res["corr_profile_points"] = int(len(prof))

    # B. floor
    floors, tot = _floor(W, fc["percentiles"])
    th = len(tot) * h_per
    res["floor_kw"] = {f"p{p}": (round(v, 2) if v is not None else None) for p, v in floors.items()}
    res["floor_timestamps"] = int(len(tot))
    res["floor_hours"] = round(th, 1)
    res["floor_days"] = int(len(set(tot.index.date))) if len(tot) else 0
    res["floor_unstable"] = bool(th < fc["min_window_hours"])
    res["mean_total_flex_kw"] = round(float(tot.mean()), 2) if len(tot) else None
    res["_total_series"] = tot
    lp = fc["leave_one_out_percentile"]
    full = floors.get(lp)
    if full is None:
        full = _floor(W, [lp])[0][lp]
    loo = {}
    for s in W.columns:
        if n < 2:
            break
        f2, t2 = _floor(W.drop(columns=s), [lp])
        f2 = f2[lp]
        contrib = (full - f2) if (full is not None and f2 is not None) else None
        mean_s = site.loc[s, "mean_window_flex_kw"]
        flag = (contrib is not None and mean_s and mean_s > 0 and contrib < fc["loo_flag_share"] * mean_s)
        loo[s] = (f2, contrib, flag, len(t2) * h_per)
    site[f"floor_p{lp}_without_site_kw"] = [round(loo[s][0], 2) if s in loo and loo[s][0] is not None else None for s in site.index]
    site[f"floor_p{lp}_change_kw"] = [round(loo[s][1], 2) if s in loo and loo[s][1] is not None else None for s in site.index]
    site["floor_change_over_own_mean"] = [round(loo[s][1] / site.loc[s, "mean_window_flex_kw"], 3)
                                          if s in loo and loo[s][1] is not None and site.loc[s, "mean_window_flex_kw"]
                                          else None for s in site.index]
    site["barely_moves_floor"] = [bool(loo[s][2]) if s in loo else None for s in site.index]
    site["floor_hours_without_site"] = [round(loo[s][3], 1) if s in loo else None for s in site.index]

    # C. coverage curve
    site = site.sort_values("mean_window_flex_kw", ascending=False, na_position="last")
    m = site["mean_window_flex_kw"].fillna(0).clip(lower=0)
    site.insert(0, "rank_by_window_flex", np.arange(1, len(site) + 1))
    site["cumulative_share"] = (m.cumsum() / m.sum()).round(3) if m.sum() > 0 else np.nan

    # A. bundles
    basis = prof_c if bc["corr_basis"] == "profile" else raw
    groups = [g for g in _cluster(basis, bc["min_corr"]) if len(g) >= 2] if n >= 2 else []
    bundles = []
    for g in groups:
        B = W[g].dropna(how="any")
        bt = B.sum(axis=1)
        mean_b = float(bt.mean()) if len(bt) else 0.0
        daily = bt.groupby(bt.index.date).mean() if len(bt) else pd.Series(dtype=float)
        share = float((daily >= bc["min_kw"]).mean()) if len(daily) else 0.0
        sub = basis.loc[g, g].to_numpy()
        r_mean = float(np.nanmean(sub[np.triu_indices(len(g), 1)]))
        bundles.append({"sites": g, "mean_pairwise_r": round(r_mean, 3), "combined_mean_window_kw": round(mean_b, 2),
                        "clears_threshold": bool(mean_b >= bc["min_kw"]),
                        "share_of_days_clearing": round(share, 3), "days": int(len(daily)),
                        "fragile": bool(mean_b >= bc["min_kw"] and share < bc["min_day_share"]),
                        "hours_common": round(len(bt) * h_per, 1)})
    res["bundles"] = bundles
    res["bundle_settings"] = dict(bc)
    bid = {}
    for i, b in enumerate(bundles, 1):
        for s in b["sites"]:
            bid[s] = f"B{i}" + (" (fragile)" if b["fragile"] else "") + ("" if b["clears_threshold"] else " (below kW threshold)")
    site["bundle"] = [bid.get(s, "") for s in site.index]
    res["solar_flagged_sites"] = [s for s in site.index if meta.get(s, {}).get("solar_flag")]
    res["base_load_not_from_winter"] = [s for s in site.index
                                        if str(meta.get(s, {}).get("base_load_basis", "")).startswith("FLAG")]
    return res, site


def print_result(res, site, cfg):
    lp = cfg["floor"]["leave_one_out_percentile"]
    print(f"\n=== {res['label']}: {res['n_sites']} site(s), resolution {res['resolution_s']:.0f} s, "
          f"window hours {res['window_hours_of_day'] if res['window_hours_of_day'] is not None else 'all day'}, "
          f"months {res['window_months']}")
    fl = ", ".join(f"{k} {v:,.1f} kW" if v is not None else f"{k} n/a" for k, v in res["floor_kw"].items())
    print(f"  floor of the group total ({res['n_sites']} sites, {res['floor_hours']:.0f} window hours "
          f"on {res['floor_days']} days where every site has data): {fl}"
          + ("   ** UNSTABLE: fewer than "
             f"{cfg['floor']['min_window_hours']} window hours **" if res["floor_unstable"] else ""))
    if res["n_sites"] < 5:
        print(f"  NOTE: only {res['n_sites']} sites -- this tests the method, it is not a real portfolio.")
    cols = [c for c in ["rank_by_window_flex", "mean_window_flex_kw", "cumulative_share", "window_coverage",
                        f"floor_p{lp}_change_kw", "barely_moves_floor", "bundle", "solar_flag"] if c in site.columns]
    with pd.option_context("display.width", 200, "display.max_colwidth", 45):
        print(site[cols].to_string())
    for i, b in enumerate(res["bundles"], 1):
        print(f"  bundle B{i}: {', '.join(b['sites'])}  r={b['mean_pairwise_r']:.2f}  "
              f"combined {b['combined_mean_window_kw']:.1f} kW, clears {cfg['bundles']['min_kw']:.0f} kW on "
              f"{b['share_of_days_clearing']:.0%} of {b['days']} days"
              + ("  FRAGILE" if b["fragile"] else "") + ("" if b["clears_threshold"] else "  (below threshold)"))
    if not res["bundles"]:
        print(f"  no group of 2+ sites with r >= {cfg['bundles']['min_corr']} ({cfg['bundles']['corr_basis']})")
    if res["solar_flagged_sites"]:
        print(f"  SOLAR flagged: {', '.join(res['solar_flagged_sites'])}")


# ======================================================================
# outputs
# ======================================================================
def _csv(df, path, lcfg):
    df.to_csv(path, sep=lcfg.get("results_csv_sep", ";"), decimal=lcfg.get("results_csv_decimal", ","),
              encoding="utf-8-sig")
    print(f"  wrote {path}")


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _json_ready(res):
    return _clean({k: v for k, v in res.items() if not k.startswith("_")})


def update_summary(cfg, key, value):
    path = os.path.join(cfg["output_dir"], "portfolio_summary.json")
    d = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            d = {}
    d[key] = _clean(value)
    d["_updated"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    d["_note"] = ("Group-level reliability FLOOR of flexible kW in the constrained window. Not the per-site "
                  "theoretical potential (a per-site ceiling).")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, default=str, ensure_ascii=False)
    print(f"  wrote {path} [{key}]")


# ---- plots ----
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"


def _plt():
    import matplotlib
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    return plt


def _short(s, n=22):
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


def plot_availability(res, cfg, path):
    plt = _plt()
    tot = res["_total_series"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 2]})
    if len(tot):
        # days with window data side by side (months without data are skipped)
        loc = tot.index.tz_localize(None)
        days = pd.Index(loc.normalize())
        uniq = days.unique()
        pos = uniq.get_indexer(days)
        h = (loc - days) / pd.Timedelta(hours=24)
        hh = np.asarray(h, float)
        frac = (hh - hh.min()) / max(hh.max() - hh.min(), 1e-9) * 0.8 if len(hh) else hh
        a1.plot(pos + frac, tot.to_numpy(), ".", ms=1.5 if len(tot) > 5000 else 3, color=BLUE, alpha=0.6)
        step = max(1, len(uniq) // 10)
        a1.set_xticks(np.arange(0, len(uniq), step), [d.strftime("%Y-%m-%d") for d in uniq[::step]])
        a1.set_xlabel(f"days with window data ({len(uniq)}), in order; other days left out")
        srt = np.sort(tot.to_numpy())[::-1]
        a2.plot(np.linspace(0, 100, len(srt)), srt, color=BLUE, lw=2)
    styles = {1: ":", 5: "--", 10: "-."}
    for k, v in res["floor_kw"].items():
        if v is None:
            continue
        p = int(k[1:])
        a1.axhline(v, color=ORANGE, lw=1.2, ls=styles.get(p, "--"))
        a2.axhline(v, color=ORANGE, lw=1.2, ls=styles.get(p, "--"), label=f"{k} floor: {v:,.0f} kW")
    if any(v is not None for v in res["floor_kw"].values()):
        a2.legend(loc="upper right", frameon=False)
    unst = "  — UNSTABLE (few hours)" if res["floor_unstable"] else ""
    a1.set_title(f"{res['label']}: total flexible kW of {res['n_sites']} sites in the constrained window "
                 f"({res['floor_hours']:.0f} h where every site has data){unst}", loc="left", color=INK)
    a1.set_ylabel("kW")
    a2.set_xlabel("share of window time the total is at least this (%)")
    a2.set_ylabel("kW")
    a2.set_xlim(0, 100)
    a2.set_title("availability (duration) curve with floor lines p1 / p5 / p10", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def _diverging():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", [ORANGE, "#f2f1ed", BLUE])


def plot_correlation(res, cfg, path, meta):
    plt = _plt()
    mats = [("raw series", pd.DataFrame(res["corr_raw"])),
            (f"{'hour' if cfg['correlation']['profile_bin'] != 'quarter' else 'quarter-hour'}-of-day profile "
             f"({res['corr_profile_points']} points)", pd.DataFrame(res["corr_profile"]))]
    n = len(mats[0][1])
    size = max(4.5, 0.55 * n + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(2 * size + 1, size))
    for k, (ax, (title, m)) in enumerate(zip(axes, mats)):
        names = [(_short(s) + (" (solar?)" if meta.get(s, {}).get("solar_flag") else "")) for s in m.index]
        im = ax.imshow(m.to_numpy(dtype=float), cmap=_diverging(), vmin=-1, vmax=1)
        ax.set_xticks(range(n), names, rotation=60, ha="right")
        ax.set_yticks(range(n), names if k == 0 else [""] * n)
        ax.grid(False)
        if n <= 14:
            for i in range(n):
                for j in range(n):
                    v = m.iat[i, j]
                    ax.text(j, i, "–" if pd.isna(v) else f"{v:.2f}", ha="center", va="center", fontsize=7, color=INK)
        ax.set_title(title, loc="left", color=INK)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Pearson r")
    fig.suptitle(f"{res['label']}: correlation of flexible kW in the constrained window", x=0.01, ha="left", color=INK)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_coverage(site, res, path):
    plt = _plt()
    s = site.dropna(subset=["mean_window_flex_kw"])
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(s) + 3), 4.5))
    x = np.arange(len(s))
    solar = s["solar_flag"].fillna("").astype(bool) if "solar_flag" in s else pd.Series(False, index=s.index)
    ax.bar(x, s["mean_window_flex_kw"], color=BLUE, width=0.7,
           hatch=["//" if v else "" for v in solar], edgecolor="white")
    for i, (kw, c) in enumerate(zip(s["mean_window_flex_kw"], s["cumulative_share"])):
        ax.text(i, kw, f"{c:.0%}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [_short(v) + (" (solar?)" if f else "") for v, f in zip(s.index, solar)], rotation=60, ha="right")
    ax.set_ylabel("mean flexible kW in the window")
    ax.set_title(f"{res['label']}: sites by mean window flexible kW\n"
                 f"label = cumulative share of the group; hatched = solar flag", loc="left", color=INK, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_scatter(proxy, meas, ratios, path, mask):
    plt = _plt()
    sites = list(proxy.columns)
    n = len(sites)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4 * rows), squeeze=False)
    for ax, s in zip(axes.ravel(), sites):
        a = proxy[s][mask]
        b = meas[s][mask]
        ok = a.notna() & b.notna()
        ax.plot(a[ok], b[ok], "o", ms=3.5, color=BLUE, alpha=0.6, mec="white", mew=0.4)
        hi = float(np.nanmax([a[ok].max(), b[ok].max(), 1])) if ok.any() else 1
        ax.plot([0, hi], [0, hi], color=MUTED, lw=1, ls="--")
        ax.set_xlim(0, hi * 1.05)
        ax.set_ylim(0, hi * 1.05)
        r = ratios.get(s)
        ax.set_title(f"{_short(s, 28)}\n1 s ÷ 15-min proxy = {r:.2f}" if r is not None else _short(s, 28),
                     loc="left", color=INK)
        ax.set_xlabel("15-min proxy: net − base (kW)")
        ax.set_ylabel("1 s compressors on, 15-min mean (kW)")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Flexible kW per quarter-hour in the window: 15-min proxy vs 1-second diagnosis (dashed = 1:1)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def write_outputs(res, site, meta, cfg, lcfg, tag):
    out = cfg["output_dir"]
    _csv(site, os.path.join(out, f"portfolio_{tag}.csv"), lcfg)
    plot_availability(res, cfg, os.path.join(out, f"availability_{tag}.png"))
    if res["n_sites"] >= 2:
        plot_correlation(res, cfg, os.path.join(out, f"correlation_{tag}.png"), meta)
    plot_coverage(site, res, os.path.join(out, f"coverage_{tag}.png"))


# ======================================================================
# modes
# ======================================================================
def run_15min(cfg, lcfg, hours, grid=None):
    included, excluded = screened_sites(cfg, lcfg)
    print(f"\n[15-min] {len(included)} site(s) in, {len(excluded)} out")
    flex, meta, bad = build_15min(included, cfg, lcfg)
    excluded += bad
    months = cfg["window"].get("months") or winter_months(lcfg)
    res, site = analyse_portfolio(flex, meta, cfg, "15-min, screened sites", hours, months)
    res["excluded_sites"] = dict(excluded)
    res["included_sites"] = {s: meta[s]["included_because"] for s in meta}
    print_result(res, site, cfg)
    for s, why in excluded:
        print(f"  OUT  {s}: {why}")
    write_outputs(res, site, meta, cfg, lcfg, "15min")
    update_summary(cfg, "15min", _json_ready(res))
    return res, site, excluded


def run_1s(cfg, lcfg, hours):
    flex, meta, excluded = build_1s(cfg)
    months = cfg["window"].get("months_1s") or None
    res, site = analyse_portfolio(flex, meta, cfg, "1-second case sites", hours, months)
    res["excluded_sites"] = dict(excluded)
    print_result(res, site, cfg)
    write_outputs(res, site, meta, cfg, lcfg, "1s")
    update_summary(cfg, "1s", _json_ready(res))
    return res, site, flex, meta, excluded


def _case_meter_sites(cfg, lcfg):
    df = L.read_comparison(lcfg)
    out, missing = [], []
    for case in cfg["case_sites"]:
        path = case.get("meter_path")
        if not path and case.get("meter_site_id") and df is not None:
            hit = df[df[L.SITE_KEY].astype(str).str.strip().str.lower() == str(case["meter_site_id"]).strip().lower()]
            if len(hit):
                path = L.find_meter_file(hit.iloc[0], lcfg)
        if path and os.path.exists(path):
            out.append((case["site_id"], path, "1-second case site (its 15-min meter file)"))
        else:
            missing.append((case["site_id"], "no 15-min meter file (set meter_site_id or meter_path in case_sites)"))
    return out, missing


def run_compare(cfg, lcfg, hours):
    lp = cfg["floor"]["leave_one_out_percentile"]
    r1, site1, f1, meta1, ex1 = run_1s(cfg, lcfg, hours)
    months = cfg["window"].get("months_1s") or None
    agg = to_15min(f1, cfg)                                   # 1 s -> 15-min means
    sites, missing = _case_meter_sites(cfg, lcfg)
    sites = [s for s in sites if s[0] in agg.columns]
    if not sites:
        raise ValueError("None of the case sites has a 15-min meter file to compare with.")
    proxy_full, meta15, bad = build_15min(sites, cfg, lcfg, label=" case")
    names = [s for s in agg.columns if s in proxy_full.columns]
    for s in names:                       # solar seen by either source is flagged in both
        both_flags = "; ".join(x for x in (meta15[s].get("solar_flag"), meta1[s].get("solar_flag")) if x)
        meta15[s]["solar_flag"] = meta1[s]["solar_flag"] = both_flags
    # only the 1-second period: quarter-hours where both exist, per site
    idx = agg.index
    proxy = proxy_full.reindex(idx)[names]
    meas = agg[names]
    both = proxy.notna() & meas.notna()
    proxy_c, meas_c = proxy.where(both), meas.where(both)
    for s in names:
        meta15[s]["quarter_hours_compared"] = int(both[s].sum())
    r15, site15 = analyse_portfolio(proxy_c, meta15, cfg, "15-min proxy, case sites, 1-s period", hours, months)
    rm, sitem = analyse_portfolio(meas_c, meta1, cfg, "1-s diagnosis as 15-min means", hours, months)
    # what averaging to quarters hides: the 1 s data at 1 s vs its own 15-min means
    ra, _ = analyse_portfolio(agg[names], meta1, cfg, "1-s diagnosis, all 15-min means", hours, months)
    rn, _ = analyse_portfolio(f1[names], meta1, cfg, "1-s diagnosis, native", hours, months)
    for r in (r15, rm):
        print_result(r, site15 if r is r15 else sitem, cfg)
    _csv(site15, os.path.join(cfg["output_dir"], "portfolio_15min_case.csv"), lcfg)

    rows = []
    ratios = {}
    for s in names:
        a, b = site15.loc[s, "mean_window_flex_kw"], sitem.loc[s, "mean_window_flex_kw"]
        ratio = (b / a) if (a is not None and b is not None and not pd.isna(a) and a > 0) else None
        ratios[s] = ratio
        rows.append({"section": "mean window flexible kW", "item": s, "proxy_15min": a, "unilm_as_15min": b,
                     "ratio_unilm_over_proxy": round(ratio, 3) if ratio is not None else None,
                     "unilm_native_1s": site1.loc[s, "mean_window_flex_kw"] if s in site1.index else None,
                     "window_hours": site15.loc[s, "window_hours_with_data"],
                     "solar_flag": meta15[s].get("solar_flag", ""),
                     "note": "ratio = correction factor for scaling the 15-min proxy of the other sites"})
    ta, tb = site15["mean_window_flex_kw"].sum(), sitem["mean_window_flex_kw"].sum()
    rows.append({"section": "mean window flexible kW", "item": "GROUP (sum of site means)", "proxy_15min": round(ta, 2),
                 "unilm_as_15min": round(tb, 2), "ratio_unilm_over_proxy": round(tb / ta, 3) if ta > 0 else None,
                 "unilm_native_1s": round(site1["mean_window_flex_kw"].sum(), 2), "window_hours": None,
                 "solar_flag": ", ".join(s for s in names if meta15[s].get("solar_flag")), "note": ""})
    for p in cfg["floor"]["percentiles"]:
        k = f"p{p}"
        a, b = r15["floor_kw"][k], rm["floor_kw"][k]
        rows.append({"section": f"floor {k} of group total", "item": f"GROUP ({len(names)} sites)", "proxy_15min": a,
                     "unilm_as_15min": b, "ratio_unilm_over_proxy": round(b / a, 3) if a and b is not None else None,
                     "unilm_native_1s": rn["floor_kw"][k], "window_hours": r15["floor_hours"],
                     "solar_flag": "",
                     "note": ("UNSTABLE: fewer than %d window hours" % cfg["floor"]["min_window_hours"])
                     if r15["floor_unstable"] else ""})
    for i, s in enumerate(names):
        for t in names[i + 1:]:
            rows.append({"section": "correlation (raw)", "item": f"{s} | {t}",
                         "proxy_15min": r15["corr_raw"][s][t], "unilm_as_15min": rm["corr_raw"][s][t],
                         "ratio_unilm_over_proxy": None, "unilm_native_1s": rn["corr_raw"][s][t],
                         "window_hours": None, "solar_flag": "", "note": ""})
            rows.append({"section": "correlation (profile)", "item": f"{s} | {t}",
                         "proxy_15min": r15["corr_profile"][s][t], "unilm_as_15min": rm["corr_profile"][s][t],
                         "ratio_unilm_over_proxy": None, "unilm_native_1s": rn["corr_profile"][s][t],
                         "window_hours": None, "solar_flag": "", "note": ""})
    comp = pd.DataFrame(rows)
    _csv(comp.set_index("section"), os.path.join(cfg["output_dir"], "portfolio_comparison.csv"), lcfg)
    mask = window_mask(idx, hours, months, cfg["window"]["weekdays_only"])
    plot_scatter(proxy_c, meas_c, ratios, os.path.join(cfg["output_dir"], "compare_scatter.png"), mask)

    res_eff = {"floor_kw_native_1s": rn["floor_kw"], "floor_kw_15min_means": ra["floor_kw"],
               "floor_hours_native_1s": rn["floor_hours"], "floor_hours_15min_means": ra["floor_hours"],
               "corr_raw_native_1s": rn["corr_raw"], "corr_raw_15min_means": ra["corr_raw"]}
    summary = {"sites": names, "excluded": dict(ex1 + missing + bad),
               "proxy_15min": _json_ready(r15), "unilm_as_15min": _json_ready(rm),
               "ratio_unilm_over_proxy": {s: (round(v, 3) if v is not None else None) for s, v in ratios.items()},
               "group_ratio": round(tb / ta, 3) if ta > 0 else None,
               "resolution_effect_within_1s_data": res_eff}
    update_summary(cfg, "compare", summary)

    print("\n=== comparison (only the case sites, only the 1-second period, same quarter-hours)")
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(comp[["section", "item", "proxy_15min", "unilm_as_15min", "ratio_unilm_over_proxy",
                    "unilm_native_1s"]].to_string(index=False))
    print(f"\n  what averaging to quarter-hours hides (1-second data only, {len(names)} sites):")
    for k in rn["floor_kw"]:
        print(f"    floor {k}: native 1 s {rn['floor_kw'][k]}, 15-min means {ra['floor_kw'][k]}")
    for i, s in enumerate(names):
        for t in names[i + 1:]:
            print(f"    r({_short(s)}, {_short(t)}): native 1 s {rn['corr_raw'][s][t]}, 15-min means {ra['corr_raw'][s][t]}")
    if len(names) < 5:
        print(f"  NOTE: {len(names)} case sites -- this tests the method, it is not a real portfolio.")
    return comp


# ======================================================================
# note
# ======================================================================
def write_note(cfg, lcfg, hours, how, mode):
    path = os.path.join(cfg["output_dir"], "portfolio_summary.json")
    try:
        with open(path, encoding="utf-8") as fh:
            S = json.load(fh)
    except (OSError, ValueError):
        return
    L_ = [f"# Portfolio analysis note ({pd.Timestamp.now():%Y-%m-%d})", "",
          f"Constrained window: hours {hours if hours is not None else 'all day (--full-day)'} ({how}).", ""]
    if "15min" in S:
        r = S["15min"]
        L_ += ["## Sites in the 15-min portfolio", ""]
        for s, why in r.get("included_sites", {}).items():
            L_.append(f"- **{s}**: {why}")
        L_ += ["", "## Sites left out", ""]
        for s, why in r.get("excluded_sites", {}).items():
            L_.append(f"- {s}: {why}")
        if not r.get("excluded_sites"):
            L_.append("- none")
        fl = ", ".join(f"{k} {v:,.0f} kW" for k, v in r["floor_kw"].items() if v is not None)
        L_ += ["", f"Floor of the group total: {fl} ({r['n_sites']} sites, {r['floor_hours']:.0f} window hours "
                   f"on {r['floor_days']} days){' -- UNSTABLE' if r['floor_unstable'] else ''}."]
        if r.get("solar_flagged_sites"):
            L_.append(f"Solar flagged: {', '.join(r['solar_flagged_sites'])}.")
    if "1s" in S:
        r = S["1s"]
        L_ += ["", "## 1-second case sites", "",
               f"{r['n_sites']} site(s); floor {', '.join(f'{k} {v:,.0f} kW' for k, v in r['floor_kw'].items() if v is not None)} "
               f"over {r['floor_hours']:.0f} window hours{' -- UNSTABLE' if r['floor_unstable'] else ''}."]
        for k, v in (r.get("excluded_sites") or {}).items():
            L_.append(f"- left out {k}: {v}")
    L_ += ["", "## Ready to show / provisional", ""]
    ready = []
    prov = []
    if "15min" in S:
        ready.append("15-min portfolio of the screened sites: coverage curve, correlations, candidate bundles and "
                     "the p1/p5/p10 floor (the choice between them is open)")
    if cfg["window"].get("hours"):
        prov.append("the constraint window is fixed by hand, not derived from the grid file")
    else:
        prov.append("the constraint window is derived from the grid load file; it is provisional until "
                    "Öresundskraft confirms the constrained hours")
    if "compare" not in S:
        prov.append("the correction factor (1 s ÷ 15-min proxy): needs the comparison run on real 1-second data")
    else:
        c = S["compare"]
        pr = c.get("proxy_15min", {})
        prov.append(f"correction factor 1 s ÷ 15-min proxy = {c.get('group_ratio')} (group; per site "
                    + ", ".join(f"{k} {v}" for k, v in (c.get("ratio_unilm_over_proxy") or {}).items())
                    + f") from {len(c.get('sites', []))} case sites over {pr.get('floor_hours')} window hours "
                    "-- a method test while there are only a few case sites")
        if c.get("excluded"):
            prov.append("case sites left out of the comparison: "
                        + "; ".join(f"{k} ({v})" for k, v in c["excluded"].items()))
    if any(v.get("floor_unstable") for k, v in S.items() if isinstance(v, dict) and "floor_unstable" in v):
        prov.append("at least one floor rests on fewer than "
                    f"{cfg['floor']['min_window_hours']} window hours (flagged unstable)")
    nb = sorted({x for k, v in S.items() if isinstance(v, dict) for x in v.get("base_load_not_from_winter", [])}
                | {x for x in ((S.get("compare") or {}).get("proxy_15min") or {}).get("base_load_not_from_winter", [])})
    if nb:
        prov.append("base load not from winter data (too little winter in the meter file): " + ", ".join(nb))
    L_ += [f"- Ready: {x}" for x in ready] + [f"- Provisional: {x}" for x in prov]
    L_ += ["", "The floor is a group-level number; it is not the per-site theoretical potential (a ceiling)."]
    p = os.path.join(cfg["output_dir"], "portfolio_note.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L_) + "\n")
    print(f"  wrote {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--mode", choices=["15min", "1s", "compare"], default="15min")
    ap.add_argument("--full-day", action="store_true", help="all 24 hours instead of the constrained window")
    ap.add_argument("--config", help="JSON file whose keys override CONFIG")
    a = ap.parse_args(argv)
    cfg = copy.deepcopy(CONFIG)
    if a.config:
        with open(a.config, encoding="utf-8") as fh:
            _deep_update(cfg, json.load(fh))
    lcfg = lpc_config(cfg)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    grid = None
    if a.full_day:
        hours, how = None, "full day requested"
    else:
        if not cfg["window"].get("hours"):
            grid = load_grid(cfg, lcfg)
        hours, how = window_hours(cfg, lcfg, grid)
    print(f"[window] hours {hours if hours is not None else 'all day'} ({how})")
    if a.mode == "15min":
        run_15min(cfg, lcfg, hours, grid)
    elif a.mode == "1s":
        run_1s(cfg, lcfg, hours)
    else:
        run_compare(cfg, lcfg, hours)
    write_note(cfg, lcfg, hours, how, a.mode)


if __name__ == "__main__":
    main()

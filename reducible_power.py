"""
reducible_power.py -- reducible power per site and clock hour (script 4 of
the revised plan).

  1 s    P_1s(s,h) = sum_k c_k * P_rated,k * units_k * a_k(h)
         c_k = controllable (compressor_list.csv), a_k(h) = running share
         (compressor_hourly.csv), both from `unilm_fhmm.py METER --out DIR`.
  15 min P_15(s,h) = f * max(0, mean 15-min load in h - L_base(s, period))
         for every f in CONFIG["f_values"]. L_base = P5 of the site's 15-min
         load in the same period, exporting quarter-hours left out -- the
         SAME functions as lpc_11 (seasonal_base_load / monthly_base_load),
         not copies. CONFIG["base_period"] = "season" (default, one level per
         season) or "month" (one level per calendar month, follows the drift
         of the base load with the weather). The load is the GROSS load where
         lpc_11 finds a generation file, else the net.

Output: reducible_power.csv with site, hour, source, f, kW plus season,
series (net / gross / 1 s compressors), base load, block length (1 h) and
percentile (none: hourly values). base_period_compare.csv puts the two
base-load definitions side by side per site, month and f (base level, mean
reducible power, hours) whichever one is chosen. Site codes only: the private key is read
here to turn site ids into codes and never written.

Run:  python reducible_power.py     (lpc_11.py in the same folder)
"""
import os
import sys

import numpy as np
import pandas as pd

CONFIG = {
    "lpc_module": "lpc_11",
    "sites_15min": "selected",        # "selected" | "all" | [site ids as in site_comparison.csv]
    "include_negative_controls": False,
    "exclude_sites": [],              # site ids (as in site_comparison.csv) or codes to leave out, e.g. ["H"]
    "f_values": [0.3, 0.5, 0.7, 1.0],
    "base_period": "season",          # "season" | "month": period of the base load L_base (15-min sites)
    # 1 s sites: {site code: unilm_fhmm output folder}
    "one_second_sites": {},
    "output_dir": "reducible_power_out",
    "csv_sep": ";",
    "csv_decimal": ",",
}


def load_lpc(cfg):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.getcwd())
    return __import__(cfg["lpc_module"])


def season_name(month, lcfg):
    for name, months in lcfg["seasons"].items():
        if month in months:
            return name
    return None


def p15_site(L, lcfg, path, code, f_values, base_period="season"):
    """Hourly P_15 for one meter file, every f. Returns (long DataFrame with the
    chosen base period, comparison DataFrame season vs month per calendar month)."""
    c = dict(lcfg, csv_path=path, show_plot=False)
    net, info = L.load_net_series(c)
    gross = L.gross_series(net, c)
    load, series = (gross, "gross (net + generation)") if gross is not None else \
        (net, "net exchange" + (" -- PV site without production data" if info.get("pv_flag") else ""))
    h = L.hourly_mean(load, c)
    seas = pd.Series([season_name(m, lcfg) for m in h.index.month], index=h.index)
    b_s = L.hourly_base_load(load, h, c, "season")
    b_m = L.hourly_base_load(load, h, c, "month")
    b = b_m if base_period == "month" else b_s
    above = (h - b).clip(lower=0)
    rows, comp = [], []
    for f in f_values:
        rows.append(pd.DataFrame({"site": code, "hour": h.index.strftime("%Y-%m-%d %H:00"), "source": "15 min",
                                  "f": f, "kW": np.round(f * above.to_numpy(), 3), "season": seas.to_numpy(),
                                  "series": series, "base_load_kw": np.round(b.to_numpy(), 2),
                                  "base_period": base_period}))
        a_s = f * (h - b_s).clip(lower=0)
        a_m = f * (h - b_m).clip(lower=0)
        ok = a_s.notna() & a_m.notna()            # same hours for both, so the two columns compare like with like
        g = pd.DataFrame({"base_season_kw": b_s[ok], "base_month_kw": b_m[ok], "kW_season_base": a_s[ok],
                          "kW_month_base": a_m[ok]}).groupby(h.index[ok].month)
        t = g.mean().round(2)
        t["hours"] = g.size()
        t.insert(0, "f", f)
        t.insert(0, "month", t.index)
        t.insert(0, "site", code)
        comp.append(t.reset_index(drop=True))
    out = pd.concat(rows, ignore_index=True)
    return out[out["kW"].notna()], pd.concat(comp, ignore_index=True)


def p1s_site(code, folder):
    """Hourly P_1s from one unilm_fhmm output folder."""
    hh = pd.read_csv(os.path.join(folder, "compressor_hourly.csv"))
    lst = pd.read_csv(os.path.join(folder, "compressor_list.csv"))
    meta = lst.set_index("appliance")[["controllable", "rated_kW", "units"]]
    hh = hh.join(meta, on="compressor_id", rsuffix="_list")
    c = pd.to_numeric(hh["controllable_list"], errors="coerce").fillna(0)
    kw = pd.to_numeric(hh["rated_kW_list"], errors="coerce").fillna(0)
    u = pd.to_numeric(hh["units_list"], errors="coerce").fillna(1)
    hh["p"] = c * kw * u * hh["running_share"]
    # an hour counts only when every controllable compressor has a share
    ctrl = c > 0
    g = hh[ctrl].groupby("hour_start_local")
    p = g["p"].sum(min_count=1)
    complete = g["running_share"].apply(lambda s: s.notna().all())
    p = p[complete]
    t = pd.to_datetime(p.index.str.slice(0, 19))           # local clock time, offset dropped
    return pd.DataFrame({"site": code, "hour": t.strftime("%Y-%m-%d %H:00"), "source": "1 s", "f": np.nan,
                         "kW": np.round(p.to_numpy(), 3), "season": None,
                         "series": "1 s: controllable compressors (rated x units x running share)",
                         "base_load_kw": np.nan})


def main(cfg=CONFIG):
    L = load_lpc(cfg)
    lcfg = L.CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)
    codes = L._code_map(lcfg)
    if not codes:
        print("[reducible] WARNING: lpc CONFIG['site_code_key'] not set -- outputs hold REAL site ids")
    if cfg["base_period"] not in ("season", "month"):
        raise SystemExit(f"[reducible] CONFIG base_period must be 'season' or 'month', not {cfg['base_period']!r}")
    parts, status, comps = [], [], []
    df = L.read_comparison(lcfg)
    if df is not None and cfg["sites_15min"]:
        if cfg["sites_15min"] == "selected":
            pick = df[df["selected"].astype(str).str.lower().isin(["true", "1", "1.0"])]
        elif cfg["sites_15min"] == "all":
            pick = df
        else:
            want = {str(s).strip().lower() for s in cfg["sites_15min"]}
            pick = df[df[L.SITE_KEY].astype(str).str.strip().str.lower().isin(want)]
        if not cfg["include_negative_controls"] and "is_negative_control" in pick:
            pick = pick[~pick["is_negative_control"].astype(str).str.lower().isin(["true", "1", "1.0"])]
        drop = {str(x).strip().lower() for x in cfg.get("exclude_sites") or []}
        for _, r in pick.iterrows():
            sid = str(r[L.SITE_KEY])
            code = codes.get(sid.strip().lower(), sid if not codes else f"UNCODED:{sid}")
            if sid.strip().lower() in drop or str(code).strip().lower() in drop:
                status.append((code, "15 min", "excluded (CONFIG exclude_sites)"))
                continue
            path = L.find_meter_file(r, lcfg)
            if path is None:
                status.append((code, "15 min", "meter file not found"))
                continue
            try:
                o, cmp_ = p15_site(L, lcfg, path, code, cfg["f_values"], cfg["base_period"])
                parts.append(o)
                comps.append(cmp_)
                status.append((code, "15 min", "ok"))
            except Exception as e:                      # one bad file must not stop the rest
                status.append((code, "15 min", f"{type(e).__name__}: {e}"))
    for code, folder in (cfg.get("one_second_sites") or {}).items():
        try:
            parts.append(p1s_site(code, folder))
            status.append((code, "1 s", "ok"))
        except Exception as e:
            status.append((code, "1 s", f"{type(e).__name__}: {e}"))
    if not parts:
        raise SystemExit("[reducible] nothing computed: " + "; ".join(f"{a} {b}: {c}" for a, b, c in status))
    out = pd.concat(parts, ignore_index=True)
    som = {m: n for n, ms in lcfg["seasons"].items() for m in ms}
    miss = out["season"].isna()
    out.loc[miss, "season"] = pd.to_datetime(out.loc[miss, "hour"]).dt.month.map(som)
    out["source_resolution"] = out["source"]
    out["block_length_h"] = 1
    out["percentile"] = "none (hourly value)"
    path = os.path.join(cfg["output_dir"], "reducible_power.csv")
    out.to_csv(path, sep=cfg["csv_sep"], decimal=cfg["csv_decimal"], index=False, encoding="utf-8-sig")
    if comps:
        cp = pd.concat(comps, ignore_index=True)
        cp.to_csv(os.path.join(cfg["output_dir"], "base_period_compare.csv"), sep=cfg["csv_sep"],
                  decimal=cfg["csv_decimal"], index=False, encoding="utf-8-sig")
        print(f"[reducible] base period used: {cfg['base_period']}")
        f0 = 0.5 if 0.5 in cfg["f_values"] else cfg["f_values"][0]
        c0 = cp[np.isclose(cp["f"], f0)].copy()
        for col in ("kW_season_base", "kW_month_base"):
            c0[col] = c0[col] * c0["hours"]
        g = c0.groupby("month")[["kW_season_base", "kW_month_base", "hours"]].sum()
        g["kW_season_base"] = (g["kW_season_base"] / g["hours"]).round(1)
        g["kW_month_base"] = (g["kW_month_base"] / g["hours"]).round(1)
        print(f"[reducible] mean reducible power per hour, all 15-min sites, f = {f0} (hours-weighted):")
        print(g.rename(columns={"kW_season_base": "season base", "kW_month_base": "month base"}).to_string())
    st = pd.DataFrame(status, columns=["site", "source", "status"])
    st.to_csv(os.path.join(cfg["output_dir"], "reducible_power_status.csv"), sep=cfg["csv_sep"], index=False,
              encoding="utf-8-sig")
    print(st.to_string(index=False))
    summ = out.groupby(["site", "source", "f"], dropna=False)["kW"].agg(["count", "median", "mean"]).round(1)
    print(summ.to_string())
    print(f"[reducible] {len(out)} rows -> {path}")
    return out


if __name__ == "__main__":
    main()

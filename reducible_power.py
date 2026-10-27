"""
reducible_power.py -- reducible power per site and clock hour (script 4 of
the revised plan).

  1 s    P_1s(s,h) = sum_k c_k * P_rated,k * units_k * a_k(h)
         c_k = controllable (compressor_list.csv), a_k(h) = running share
         (compressor_hourly.csv), both from `unilm_fhmm.py METER --out DIR`.
  15 min P_15(s,h) = f * max(0, mean 15-min load in h - L_base(s, season))
         for every f in CONFIG["f_values"]. L_base = P5 of the site's 15-min
         load in the same season, exporting quarter-hours left out -- the
         SAME function as lpc_11 (seasonal_base_load), not a copy. The load is
         the GROSS load where lpc_11 finds a generation file, else the net.

Output: reducible_power.csv with site, hour, source, f, kW plus season,
series (net / gross / 1 s compressors), base load, block length (1 h) and
percentile (none: hourly values). Site codes only: the private key is read
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
    "f_values": [0.3, 0.5, 0.7, 1.0],
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


def p15_site(L, lcfg, path, code, f_values):
    """Hourly P_15 for one meter file, every f. Returns a long DataFrame."""
    c = dict(lcfg, csv_path=path, show_plot=False)
    net, info = L.load_net_series(c)
    gross = L.gross_series(net, c)
    load, series = (gross, "gross (net + generation)") if gross is not None else \
        (net, "net exchange" + (" -- PV site without production data" if info.get("pv_flag") else ""))
    base = L.seasonal_base_load(load, c)
    h = L.hourly_mean(load, c)
    seas = pd.Series([season_name(m, lcfg) for m in h.index.month], index=h.index)
    b = seas.map({k: v[0] for k, v in base.items()}).astype(float)
    above = (h - b).clip(lower=0)
    rows = []
    for f in f_values:
        rows.append(pd.DataFrame({"site": code, "hour": h.index.strftime("%Y-%m-%d %H:00"), "source": "15 min",
                                  "f": f, "kW": np.round(f * above.to_numpy(), 3), "season": seas.to_numpy(),
                                  "series": series, "base_load_kw": np.round(b.to_numpy(), 2)}))
    out = pd.concat(rows, ignore_index=True)
    return out[out["kW"].notna()]


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
    parts, status = [], []
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
        for _, r in pick.iterrows():
            sid = str(r[L.SITE_KEY])
            code = codes.get(sid.strip().lower(), sid if not codes else f"UNCODED:{sid}")
            path = L.find_meter_file(r, lcfg)
            if path is None:
                status.append((code, "15 min", "meter file not found"))
                continue
            try:
                parts.append(p15_site(L, lcfg, path, code, cfg["f_values"]))
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

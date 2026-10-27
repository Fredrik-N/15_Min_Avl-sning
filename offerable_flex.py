"""
offerable_flex.py -- offerable flexibility per site and for the pool, for every
hour of the year (script 6, core of the revised plan).

Input: reducible_power.csv (reducible_power.py). Per site the hourly
reducible power is P_1s where that hour exists, otherwise P_15 at the given f
(every hour's source is kept: column share_1s). A site with only 1 s data
has no f; it enters every f run with the same values.

1. Time slices: month x hour of day x day type (weekday / weekend).
2. Levels per site and slice: typical = P50, firm = P10; sensitivity P5, P20.
3. Blocks of 1, 2, 4 consecutive hours: offerable power of a block starting
   at hour h = the minimum hour in the block (an LFM bid is held as an hourly
   average). Rebound (load in the 2 h after a block minus the same hours on
   comparable days) needs real activations or an event list: there is none in
   the data, so it is N/A with that reason.
4. Classification per site, season (and year) and block length:
   stand-alone (firm >= 100 kW), pool candidate (firm >= pool_candidate_kw,
   default 10), not relevant. firm = P10 of the block power over the hours.
5. Pool: sites summed hour by hour (only hours where every pool site has a
   value), then the same percentiles. Share of hours with pool >= 0.1 MW, and
   the number of sites needed for a firm 0.1 MW when sites are added in order
   of their firm level.
Outputs: offer_levels.csv, offer_classification.csv, pool_levels.csv,
pool_sites_needed.csv, pool_hourly.csv (for figures.py). Every row records
source resolution, season, block length, percentile and f.
"""
import os

import numpy as np
import pandas as pd

CONFIG = {
    "input_csv": os.path.join("reducible_power_out", "reducible_power.csv"),
    "output_dir": "offerable_flex_out",
    "f_values": None,                 # None = every f in the input
    "blocks_h": [1, 2, 4],
    "levels": {"typical": 50, "firm": 10, "sens_P5": 5, "sens_P20": 20},
    "standalone_kw": 100.0,
    "pool_candidate_kw": 10.0,
    "lfm_min_kw": 100.0,              # 0.1 MW
    "pool_sites": "all",              # "all" or [site codes]
    "seasons": {"winter": [11, 12, 1, 2], "summer": [6, 7, 8], "shoulder": [3, 4, 5, 9, 10]},
    "weekend_days": [5, 6],
    "csv_sep": ";",
    "csv_decimal": ",",
}


def read_input(cfg):
    d = pd.read_csv(cfg["input_csv"], sep=cfg["csv_sep"], decimal=cfg["csv_decimal"], encoding="utf-8-sig")
    d["hour"] = pd.to_datetime(d["hour"])
    return d


def site_series(d, f):
    """{site: DataFrame(kW, is_1s) on a regular hourly index}."""
    out = {}
    for site, g in d.groupby("site"):
        s15 = g[(g["source"] == "15 min") & np.isclose(g["f"].astype(float), f)].groupby("hour")["kW"].mean()
        s1 = g[g["source"] == "1 s"].groupby("hour")["kW"].mean()
        if s15.empty and s1.empty:
            continue
        idx = s15.index.union(s1.index)
        kw = s15.reindex(idx)
        is1 = pd.Series(False, index=idx)
        if len(s1):
            kw.loc[s1.index] = s1
            is1.loc[s1.index] = True
        full = pd.date_range(idx.min(), idx.max(), freq="1h")
        out[site] = pd.DataFrame({"kW": kw.reindex(full), "is_1s": is1.reindex(full).fillna(False).astype(bool)})
    return out


def block_min(s, L):
    """Value at the block's start hour = min over the L hours from there
    (NaN if any hour of the block is missing)."""
    return s.rolling(L, min_periods=L).min().shift(-(L - 1))


def slice_keys(idx, cfg):
    dt = np.where(np.isin(idx.dayofweek, cfg["weekend_days"]), "weekend", "weekday")
    return pd.DataFrame({"month": idx.month, "hour_of_day": idx.hour, "day_type": dt}, index=idx)


def season_of(idx, cfg):
    som = {m: k for k, ms in cfg["seasons"].items() for m in ms}
    return pd.Series([som.get(m) for m in idx.month], index=idx)


def levels_table(x, is1, cfg):
    """Percentile levels per slice of one hourly block series."""
    df = slice_keys(x.index, cfg).assign(kW=x.to_numpy(), is1=is1.to_numpy()).dropna(subset=["kW"])
    g = df.groupby(["month", "hour_of_day", "day_type"])
    t = g["kW"].agg(n_hours="count")
    for name, p in cfg["levels"].items():
        t[f"{name}_P{p}_kw"] = g["kW"].quantile(p / 100)
    t["share_1s"] = g["is1"].mean()
    return t.reset_index()


def classify(firm, cfg):
    if pd.isna(firm):
        return "n/a"
    return "stand-alone" if firm >= cfg["standalone_kw"] else \
        "pool candidate" if firm >= cfg["pool_candidate_kw"] else "not relevant"


def main(cfg=CONFIG):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    d = read_input(cfg)
    fs = cfg["f_values"] or sorted(d.loc[d["source"] == "15 min", "f"].dropna().astype(float).unique()) or [1.0]
    pf = cfg["levels"]["firm"] / 100
    lv_rows, cl_rows, pool_rows, need_rows, pool_hourly = [], [], [], [], []
    for f in fs:
        ser = site_series(d, f)
        blocks = {}
        for site, sdf in ser.items():
            src = "1 s" if sdf["is_1s"].all() else "15 min" if not sdf["is_1s"].any() else "1 s where available, else 15 min"
            for L in cfg["blocks_h"]:
                b = block_min(sdf["kW"], L)
                blocks[(site, L)] = b
                t = levels_table(b, sdf["is_1s"], cfg)
                t.insert(0, "site", site)
                t.insert(1, "f", f if not sdf["is_1s"].all() else np.nan)
                t.insert(2, "block_h", L)
                t["source"] = src
                lv_rows.append(t)
                seas = season_of(b.index, cfg)
                for season in list(cfg["seasons"]) + ["year"]:
                    v = b if season == "year" else b[(seas == season).to_numpy()]
                    v = v.dropna()
                    firm = v.quantile(pf) if len(v) else np.nan
                    sl = levels_table(v, sdf["is_1s"].reindex(v.index), cfg) if len(v) else pd.DataFrame()
                    fk = f"firm_P{cfg['levels']['firm']}_kw"
                    cl_rows.append({"site": site, "f": f if not sdf["is_1s"].all() else np.nan, "block_h": L,
                                    "season": season, "n_hours": len(v),
                                    "typical_P50_kw": v.median() if len(v) else np.nan, "firm_P10_kw": firm,
                                    "class": classify(firm, cfg),
                                    "share_slices_standalone": (sl[fk] >= cfg["standalone_kw"]).mean() if len(sl) else np.nan,
                                    "share_slices_pool_or_better": (sl[fk] >= cfg["pool_candidate_kw"]).mean() if len(sl) else np.nan,
                                    "share_hours_ge_100kw": (v >= cfg["standalone_kw"]).mean() if len(v) else np.nan,
                                    "source": src, "rebound_kw": np.nan,
                                    "rebound_note": "N/A: no activations / event list in the data"})
        # ---- pool
        sites = list(ser) if cfg["pool_sites"] == "all" else [s for s in cfg["pool_sites"] if s in ser]
        if not sites:
            continue
        for L in cfg["blocks_h"]:
            mat = pd.concat({s: blocks[(s, L)] for s in sites}, axis=1)
            is1 = pd.concat({s: ser[s]["is_1s"] for s in sites}, axis=1).reindex(mat.index).fillna(False)
            pool = mat.sum(axis=1, min_count=len(sites))
            pool[mat.isna().any(axis=1)] = np.nan
            t = levels_table(pool, is1.mean(axis=1), cfg)
            t.insert(0, "f", f)
            t.insert(1, "block_h", L)
            t["n_sites"] = len(sites)
            pool_rows.append(t)
            seas = season_of(pool.index, cfg)
            ph = pd.DataFrame({"hour": pool.index, "f": f, "block_h": L, "pool_kw": pool.to_numpy(),
                               "n_sites": len(sites)})
            pool_hourly.append(ph)
            # sites needed: add in order of the site's firm level (year)
            firm_site = {s: blocks[(s, L)].dropna().quantile(pf) for s in sites}
            order = sorted(sites, key=lambda s: -np.nan_to_num(firm_site[s], nan=-1))
            for season in list(cfg["seasons"]) + ["year"]:
                m = mat if season == "year" else mat[(seas == season).to_numpy()]
                v_all = pool if season == "year" else pool[(seas == season).to_numpy()]
                v_all = v_all.dropna()
                n_need, cum = None, []
                for k in range(1, len(order) + 1):
                    sub = m[order[:k]]
                    ps = sub.sum(axis=1, min_count=k)[~sub.isna().any(axis=1)]
                    fv = ps.quantile(pf) if len(ps) else np.nan
                    cum.append(f"{order[k - 1]}:{fv:.1f}" if pd.notna(fv) else f"{order[k - 1]}:n/a")
                    if n_need is None and pd.notna(fv) and fv >= cfg["lfm_min_kw"]:
                        n_need = k
                need_rows.append({"f": f, "block_h": L, "season": season, "n_sites_available": len(order),
                                  "pool_hours": len(v_all),
                                  "share_hours_pool_ge_0_1MW": (v_all >= cfg["lfm_min_kw"]).mean() if len(v_all) else np.nan,
                                  "pool_firm_P10_kw": v_all.quantile(pf) if len(v_all) else np.nan,
                                  "sites_needed_for_firm_0_1MW": n_need if n_need else "not reached",
                                  "order_and_cumulative_firm_kw": " | ".join(cum)})
    meta = {"source_resolution": "see source / share_1s", "percentile_def": str(cfg["levels"])}
    w = dict(sep=cfg["csv_sep"], decimal=cfg["csv_decimal"], index=False, encoding="utf-8-sig")

    def wr(rows, name):
        if not rows:
            print(f"[offer] {name}: nothing to write")
            return None
        t = pd.concat(rows, ignore_index=True) if isinstance(rows[0], pd.DataFrame) else pd.DataFrame(rows)
        if "season" not in t and "month" in t:
            t["season"] = t["month"].map({m: k for k, ms in cfg["seasons"].items() for m in ms})
        for k, v in meta.items():
            t[k] = v
        t.round(3).to_csv(os.path.join(cfg["output_dir"], name), **w)
        return t
    wr(lv_rows, "offer_levels.csv")
    cl = wr(cl_rows, "offer_classification.csv")
    wr(pool_rows, "pool_levels.csv")
    nd = wr(need_rows, "pool_sites_needed.csv")
    if pool_hourly:
        pd.concat(pool_hourly, ignore_index=True).round(3).to_csv(
            os.path.join(cfg["output_dir"], "pool_hourly.csv"), **w)
    if cl is not None:
        print(cl[cl["season"] == "year"].pivot_table(index=["site"], columns=["f", "block_h"], values="class",
                                                     aggfunc="first").to_string())
    if nd is not None:
        print(nd[["f", "block_h", "season", "share_hours_pool_ge_0_1MW", "pool_firm_P10_kw",
                  "sites_needed_for_firm_0_1MW"]].round(3).to_string(index=False))
    print(f"[offer] outputs in {os.path.abspath(cfg['output_dir'])}")


if __name__ == "__main__":
    main()

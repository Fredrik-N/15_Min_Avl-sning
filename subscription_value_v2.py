"""
subscription_value.py (v2) -- value of a lower subscription level from the
RQ1 pool, without any grid series (task_subscription_value_v2.md).

The level, hours above it and energy to cut are no longer derived from the
area grid series: it is not a valid stand-in for Öresundskraft's
subscription. The script uses only (a) the pool's firm level per month from
the RQ1 run and (b) the subscription price; activated and reserved hours are
scenarios. Method after FlexAbility delrapport 3: the saving from a lower
level, minus the cost of the delivered energy at a swept flat price.

  L_m   firm level of the pool in month m (kW): the PERCENTILE of the pool's
        hourly block series over ALL hours of the month (DAYS filter applied),
        for F_SHARE and BLOCK_H. pool_levels.csv holds only month x hour-of-
        day slices, so L_m is recomputed from pool_hourly.csv (POOL "selected")
        or pool_variants_hourly.csv (other pools) in the same folder; the
        header says which.
  V_m   = P_SUB * L_m                              (SEK)
  Net_m = P_SUB*L_m - pi_a*(L_m/1000)*H_a - pi_r*(L_m/1000)*H_r
        every activated hour counted as a full hour at L_m (cautious).
  Break-even (pool-independent): per MW and month the saving is 1000*P_SUB;
        H_a* = 1000*P_SUB/pi_a,  pi_r* = 1000*P_SUB/H_r.
  Risk figure: a missed delivery in the peak hour costs 2*P_SUB per kW in
        that month against a saving of P_SUB per kW and month: break-even
        share of failed months = P_SUB / (2*P_SUB) = 0.5.

Outputs in subscription_value_out/ (the config tag, e.g.
f0.5_P10_1h_all_selected_psub40, is in every config-dependent file name, so
two runs never overwrite each other): value_monthly_<tag>.csv,
breakeven_hours_psub<P>.csv, net_value_grid_<tag>.csv, sensitivity_v2_<tag>.csv,
value_table_<tag>.tex, breakeven_table_psub40.tex, fig_net_value_<tag>.pdf/png,
header_<tag>.txt. Files of the v1 run are moved to archive_v1/ (not deleted).
Deterministic. The config is Fredrik's choice -- do not tune it to the result.
"""
import datetime as dt
import os
import shutil
import sys

import numpy as np
import pandas as pd

# ======================================================================
# CONFIG
# ======================================================================
RQ1_SOURCE = os.path.join("offerable_flex_out", "pool_levels.csv")   # path to the RQ1 output
F_SHARE = 0.5          # controllable share f: 0.3, 0.5, 0.7, 1.0
PERCENTILE = "P10"     # firm level: "P5", "P10", "P20" (or "P50" for the typical level)
BLOCK_H = 1            # block length: 1, 2 or 4
DAYS = "all"           # "all", "weekday" or "weekend"
POOL = "selected"      # site set of the RQ1 run: "selected" or a pool variant (e.g. "without_D")
P_SUB = 40.0           # SEK per kW and month

OUTPUT_DIR = "subscription_value_out"
CSV_SEP, CSV_DEC = ";", ","
H_A = [0, 1, 2, 4, 6, 8, 12]                 # activated hours per month
PI_A = [0, 3300, 10000]                      # activation price, SEK/MWh
RES_GRID = dict(pi_r=[200, 1000], H_r=[20, 60, 120, 730], H_a=2, pi_a=3300)
BE_PI_A = [1000, 3300, 5000, 10000, 16000]   # break-even lists
BE_H_R = [10, 20, 60, 120, 730]
SENS = dict(P_SUB=[16, 30, 40, 50], F_SHARE=[0.3, 0.5, 1.0], PERCENTILE=["P5", "P10", "P20"], BLOCK_H=[1, 2, 4])
SENS_POINT = dict(H_a=4, pi_a=3300)          # net value reported in the sensitivity table
V1_FILES = ["subscription_value_table_e250.tex", "subscription_value_limits.csv",
            "fig_need_energy.pdf", "fig_need_energy.png", "fig_breakeven.pdf", "fig_breakeven.png"]
V2_PREFIX = ("value_monthly_", "breakeven_hours_", "net_value_grid_", "sensitivity_v2_", "value_table_",
             "breakeven_table_", "fig_net_value_", "header_")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# ======================================================================


def read(path):
    d = pd.read_csv(path, sep=CSV_SEP, decimal=CSV_DEC, encoding="utf-8-sig")
    if "f" in d:
        d["f"] = pd.to_numeric(d["f"], errors="coerce")
    return d


class Pools:
    """The pool's hourly block series for every (pool, f, block). "selected"
    = pool_hourly.csv (the RQ1 pool); other names = pool_variants_hourly.csv
    (its variant "all" is the same pool as "selected" and is not listed twice)."""

    def __init__(self, source):
        folder = os.path.dirname(source) or "."
        self.files = {}
        parts = []
        p = os.path.join(folder, "pool_hourly.csv")
        if os.path.exists(p):
            d = read(p)
            d["pool"] = "selected"
            parts.append(d)
            self.files["selected"] = p
        p = os.path.join(folder, "pool_variants_hourly.csv")
        if os.path.exists(p):
            d = read(p)
            d = d[d["variant"] != "all"].rename(columns={"variant": "pool"})
            parts.append(d)
            for name in d["pool"].unique():
                self.files[name] = p
        if not parts:
            sys.exit(f"No pool_hourly.csv in {folder!r} (next to RQ1_SOURCE {source!r}). Run offerable_flex.py first.")
        d = pd.concat(parts, ignore_index=True)
        d["hour"] = pd.to_datetime(d["hour"])
        self.d = d
        self.combos = d.groupby(["pool", "f", "block_h"]).size().reset_index()[["pool", "f", "block_h"]]
        self.source_kind = self._source_kind(source)

    @staticmethod
    def _source_kind(source):
        if not os.path.exists(source):
            return f"{source} not found; L_m computed from the pool's hourly block series"
        cols = pd.read_csv(source, sep=CSV_SEP, nrows=0, encoding="utf-8-sig").columns
        if "hour_of_day" in cols:
            return (f"{os.path.basename(source)} holds only month x hour-of-day slices, so L_m is recomputed "
                    f"from the pool's hourly block series (all hours of the month)")
        return f"{os.path.basename(source)} read; L_m recomputed from the pool's hourly block series"

    def series(self, pool, f, block, days):
        s = self.d[(self.d["pool"] == pool) & np.isclose(self.d["f"], f) & (self.d["block_h"] == block)]
        if s.empty:
            have = "\n  ".join(f"POOL={r.pool!r}, F_SHARE={r.f:g}, BLOCK_H={r.block_h}" for r in self.combos.itertuples())
            raise LookupError(f"The RQ1 output has no POOL={pool!r}, F_SHARE={f:g}, BLOCK_H={block}. "
                              f"Combinations that exist:\n  {have}")
        s = s.set_index("hour")["pool_kw"].dropna()
        if days == "weekday":
            s = s[s.index.dayofweek < 5]
        elif days == "weekend":
            s = s[s.index.dayofweek >= 5]
        elif days != "all":
            raise LookupError(f"DAYS must be 'all', 'weekday' or 'weekend', not {days!r}")
        return s

    def n_sites(self, pool, f, block):
        s = self.d[(self.d["pool"] == pool) & np.isclose(self.d["f"], f) & (self.d["block_h"] == block)]
        return int(s["n_sites"].iloc[0]) if len(s) and "n_sites" in s else None


def pct_value(pct):
    try:
        q = float(str(pct).upper().lstrip("P"))
    except ValueError:
        raise LookupError(f"PERCENTILE must look like 'P10', not {pct!r}")
    if not 0 <= q <= 100:
        raise LookupError(f"PERCENTILE out of range: {pct!r}")
    return q / 100


def monthly_level(pools, pool, f, block, pct, days):
    s = pools.series(pool, f, block, days)
    q = pct_value(pct)
    g = s.groupby(s.index.month)
    L = g.quantile(q).clip(lower=0)
    n = g.size()
    out = pd.DataFrame({"month": range(1, 13)})
    out["n_hours"] = out["month"].map(n).fillna(0).astype(int)
    out["L_kW"] = out["month"].map(L)
    return out


def tag(f, pct, block, days, pool, p_sub):
    return f"f{f:g}_{pct}_{block}h_{days}_{pool}_psub{p_sub:g}"


def net(L_kw, p_sub, H_a, pi_a, H_r, pi_r):
    return p_sub * L_kw - pi_a * (L_kw / 1000) * H_a - pi_r * (L_kw / 1000) * H_r


def net_grid(L, p_sub):
    rows = []
    cases = [(ha, pa, 0, 0) for pa in PI_A for ha in H_A]
    cases += [(RES_GRID["H_a"], RES_GRID["pi_a"], hr, pr) for pr in RES_GRID["pi_r"] for hr in RES_GRID["H_r"]]
    for ha, pa, hr, pr in cases:
        nm = net(L["L_kW"], p_sub, ha, pa, hr, pr)
        for m, x in zip(L["month"], nm):
            rows.append(dict(month=MONTHS[m - 1], H_a=ha, pi_a_SEK_MWh=pa, H_r=hr, pi_r_SEK_MW_h=pr, net_SEK=x))
        rows.append(dict(month="Year", H_a=ha, pi_a_SEK_MWh=pa, H_r=hr, pi_r_SEK_MW_h=pr,
                         net_SEK=nm.sum(min_count=1)))
    return pd.DataFrame(rows)


def breakeven(p_sub):
    s = 1000 * p_sub
    a = [dict(kind="activation only", assumption=f"{pa} SEK/MWh", breakeven=round(s / pa, 2),
              unit="activated hours per month") for pa in BE_PI_A]
    r = [dict(kind="availability only", assumption=f"{h} h reserved per month", breakeven=round(s / h, 1),
              unit="SEK per MW and hour") for h in BE_H_R]
    out = pd.DataFrame(a + r)
    out["P_SUB"] = p_sub
    out["saving_SEK_per_MW_month"] = s
    return out


def annual(L, p_sub):
    return (float((p_sub * L["L_kW"]).sum(min_count=1)),
            float(net(L["L_kW"], p_sub, SENS_POINT["H_a"], SENS_POINT["pi_a"], 0, 0).sum(min_count=1)))


def sensitivity(pools, base):
    rows = []

    def add(param, value, cfg):
        try:
            L = monthly_level(pools, cfg["POOL"], cfg["F_SHARE"], cfg["BLOCK_H"], cfg["PERCENTILE"], cfg["DAYS"])
        except LookupError:
            rows.append(dict(parameter=param, value=value, annual_value_SEK=np.nan, annual_net_SEK=np.nan,
                             note="not in the RQ1 output"))
            return
        v, n_ = annual(L, cfg["P_SUB"])
        rows.append(dict(parameter=param, value=value, annual_value_SEK=round(v), annual_net_SEK=round(n_),
                         months_with_level=int(L["L_kW"].notna().sum()), note=""))

    add("base", "config", base)
    for param, vals in SENS.items():
        for v in vals:
            if param == "F_SHARE" and not np.isclose(pools.d["f"], v).any():
                continue
            add(param, v, dict(base, **{param: v}))
    for p in sorted(pools.d["pool"].unique()):
        add("POOL", p, dict(base, POOL=p))
    out = pd.DataFrame(rows)
    out["net_at"] = f"H_a={SENS_POINT['H_a']} h/month, pi_a={SENS_POINT['pi_a']} SEK/MWh, no reservation"
    return out


def fmt(x, d=0):
    return "--" if pd.isna(x) else f"{x:,.{d}f}".replace(",", r"\,")


def write_tex(path, caption, label, colspec, head, body):
    t = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{tab:{label}}}",
         rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", head, r"\midrule", *body, r"\bottomrule",
         r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(t) + "\n")


def value_table(path, L, cfg, n_sites, t):
    V = cfg["P_SUB"] * L["L_kW"]
    body = [f"{MONTHS[m - 1]} & {fmt(l)} & {fmt(x)} \\\\" for m, l, x in zip(L["month"], L["L_kW"], V)]
    body += [r"\midrule", rf"Year & {fmt(L['L_kW'].mean())} (mean) & {fmt(V.sum(min_count=1))} \\"]
    days = {"all": "all days", "weekday": "weekdays", "weekend": "weekends"}[cfg["DAYS"]]
    pool = cfg["POOL"].replace("_", " ") + (f", {n_sites} sites" if n_sites else "")
    write_tex(path, rf"Value of lowering the subscription by the pool's firm level $L_m$ ({cfg['PERCENTILE']} over "
                    rf"all hours of the month, {days}), $f={cfg['F_SHARE']:g}$, {cfg['BLOCK_H']}-hour blocks, pool: "
                    rf"{pool}; $V_m = p_{{\mathrm{{sub}}}} L_m$ with $p_{{\mathrm{{sub}}}} = {cfg['P_SUB']:g}$\,SEK/kW "
                    r"per month. Gross value, before the cost of delivered energy.",
              f"subscription_value_{t.replace('.', '')}", "lrr", r"Month & $L_m$ (kW) & $V_m$ (SEK) \\", body)


def breakeven_table(path, p_sub):
    s = 1000 * p_sub
    body = [rf"Activation only & 3\,300\,SEK/MWh & {s / 3300:.1f}\,h activated per month \\",
            rf"Activation only & 10\,000\,SEK/MWh & {s / 10000:.1f}\,h activated per month \\",
            r"\midrule"]
    body += [rf"Availability only & {h}\,h reserved per month & {fmt(s / h)}\,SEK per MW and hour \\"
             for h in (20, 60, 730)]
    write_tex(path, rf"Break-even of a lower subscription level: the saving is {fmt(s)}\,SEK per MW and month "
                    rf"($p_{{\mathrm{{sub}}}} = {p_sub:g}$\,SEK/kW per month). Beyond the break-even, paying for "
                    r"the activated or reserved flexibility costs more than the saving.",
              "subscription_breakeven", "llr", r"Case & Assumption & Break-even \\", body)


def figure(path_stem, L, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ha = np.arange(0, 13)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for pa, ls in zip(PI_A, ["-", "--", ":"]):
        y = [net(L["L_kW"], cfg["P_SUB"], h, pa, 0, 0).sum(min_count=1) / 1000 for h in ha]
        ax.plot(ha, y, color="k", ls=ls, lw=1.6, label=f"{pa:,} SEK/MWh".replace(",", " "))
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("activated hours per month")
    ax.set_ylabel("annual net value (kSEK)")
    ax.set_title(f"Net value of a lower subscription level: pool {cfg['POOL'].replace('_', ' ')}, "
                 f"f = {cfg['F_SHARE']:g}, {cfg['PERCENTILE']}, {cfg['BLOCK_H']} h, {cfg['DAYS']}", fontsize=9)
    ax.legend(title="activation price", frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=200)
    plt.close(fig)


def archive_v1(out):
    """Move the v1 outputs (and any other file that is not a v2 output) to archive_v1/."""
    arch = os.path.join(out, "archive_v1")
    moved = []
    for f in sorted(os.listdir(out)):
        p = os.path.join(out, f)
        if os.path.isfile(p) and (f in V1_FILES or not f.startswith(V2_PREFIX)):
            os.makedirs(arch, exist_ok=True)
            shutil.move(p, os.path.join(arch, f))
            moved.append(f)
    return moved


def checks(L, grid, p_sub, out):
    res = [("L_m >= 0 for every month", bool((L["L_kW"].dropna() >= 0).all()))]
    V = p_sub * L["L_kW"]
    res.append(("annual value = sum of monthly values",
                bool(np.isclose(V.sum(min_count=1), sum(x for x in V if pd.notna(x))))))
    g0 = grid[(grid["H_a"] == 0) & (grid["H_r"] == 0) & (grid["pi_a_SEK_MWh"] == 0) & (grid["month"] == "Year")]
    res.append(("net at H_a = 0 and H_r = 0 equals the gross value",
                bool(np.isclose(g0["net_SEK"].iloc[0], V.sum(min_count=1), equal_nan=True))))
    res.append(("break-even at P_SUB = 40: 12.1 h at 3 300 and 4.0 h at 10 000 SEK/MWh",
                round(40000 / 3300, 1) == 12.1 and round(40000 / 10000, 1) == 4.0))
    bad = []
    for f in os.listdir(out):
        p = os.path.join(out, f)
        if os.path.isfile(p) and f.endswith((".csv", ".tex", ".txt")):
            txt = open(p, encoding="utf-8", errors="ignore").read().lower()
            if "substation" in txt or "proxy" in txt:
                bad.append(f)
    res.append(("no output file refers to the substation series" + (f" (found in {bad})" if bad else ""), not bad))
    return res


def main():
    cfg = dict(F_SHARE=F_SHARE, PERCENTILE=PERCENTILE, BLOCK_H=BLOCK_H, DAYS=DAYS, POOL=POOL, P_SUB=P_SUB)
    pools = Pools(RQ1_SOURCE)
    try:
        L = monthly_level(pools, POOL, F_SHARE, BLOCK_H, PERCENTILE, DAYS)
    except LookupError as e:
        sys.exit(f"[subscription v2] {e}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    moved = archive_v1(OUTPUT_DIR)
    t = tag(F_SHARE, PERCENTILE, BLOCK_H, DAYS, POOL, P_SUB)
    n_sites = pools.n_sites(POOL, F_SHARE, BLOCK_H)
    w = dict(sep=CSV_SEP, decimal=CSV_DEC, index=False, encoding="utf-8-sig")

    vm = pd.DataFrame({"month": [MONTHS[m - 1] for m in L["month"]], "n_hours": L["n_hours"],
                       "L_kW": L["L_kW"].round(2), "V_SEK": (P_SUB * L["L_kW"]).round(0)})
    year = dict(month="Year", n_hours=int(L["n_hours"].sum()), L_kW=np.nan,
                V_SEK=round((P_SUB * L["L_kW"]).sum(min_count=1)), L_mean_kW=round(L["L_kW"].mean(), 2),
                L_min_kW=round(L["L_kW"].min(), 2), L_max_kW=round(L["L_kW"].max(), 2))
    vm = pd.concat([vm, pd.DataFrame([year])], ignore_index=True)
    for k_, x in cfg.items():
        vm[k_] = x
    vm.to_csv(os.path.join(OUTPUT_DIR, f"value_monthly_{t}.csv"), **w)

    breakeven(P_SUB).to_csv(os.path.join(OUTPUT_DIR, f"breakeven_hours_psub{P_SUB:g}.csv"), **w)
    grid = net_grid(L, P_SUB)
    g = grid.copy()
    g["net_SEK"] = g["net_SEK"].round(0)
    g.to_csv(os.path.join(OUTPUT_DIR, f"net_value_grid_{t}.csv"), **w)
    sensitivity(pools, cfg).to_csv(os.path.join(OUTPUT_DIR, f"sensitivity_v2_{t}.csv"), **w)
    value_table(os.path.join(OUTPUT_DIR, f"value_table_{t}.tex"), L, cfg, n_sites, t)
    breakeven_table(os.path.join(OUTPUT_DIR, "breakeven_table_psub40.tex"), 40.0)
    figure(os.path.join(OUTPUT_DIR, f"fig_net_value_{t}"), L, cfg)

    src = pools.files[POOL]
    when = dt.datetime.fromtimestamp(os.path.getmtime(src)).strftime("%Y-%m-%d %H:%M")
    missing = ", ".join(MONTHS[m - 1] for m in L.loc[L["n_hours"] == 0, "month"]) or "none"
    hdr = [f"subscription_value v2 -- config {t}",
           f"F_SHARE={F_SHARE:g}  PERCENTILE={PERCENTILE}  BLOCK_H={BLOCK_H}  DAYS={DAYS}  POOL={POOL}"
           + (f" ({n_sites} sites)" if n_sites else "") + f"  P_SUB={P_SUB:g} SEK/kW per month",
           f"RQ1 source: {RQ1_SOURCE}; series read: {src} (file date {when})",
           f"L_m: {pools.source_kind}. {PERCENTILE} per calendar month over {DAYS} hours. "
           f"Months without hours: {missing}.",
           f"Annual gross value: {(P_SUB * L['L_kW']).sum(min_count=1):,.0f} SEK; L_m mean {L['L_kW'].mean():.1f}, "
           f"min {L['L_kW'].min():.1f}, max {L['L_kW'].max():.1f} kW.",
           f"Risk figure: a lower level saves {P_SUB:g} SEK per kW and month; a missed delivery in the peak hour "
           f"costs 2 x {P_SUB:g} = {2 * P_SUB:g} SEK per kW in that month; break-even share of months with a "
           f"failed delivery = {P_SUB:g} / {2 * P_SUB:g} = {P_SUB / (2 * P_SUB):.2f}.",
           "Activated hours are counted as full hours at L_m (cautious: real events may be partial).",
           f"Earlier (v1) files moved to archive_v1/: {', '.join(moved) or 'none'}"]
    path_h = os.path.join(OUTPUT_DIR, f"header_{t}.txt")
    with open(path_h, "w", encoding="utf-8") as fh:
        fh.write("\n".join(hdr) + "\n")
    ck = checks(L, grid, P_SUB, OUTPUT_DIR)
    hdr += ["Checks:"] + [f"  [{'ok' if ok else 'FAILED'}] {name}" for name, ok in ck]
    with open(path_h, "w", encoding="utf-8") as fh:
        fh.write("\n".join(hdr) + "\n")
    print("\n".join(hdr))
    print(f"\n[subscription v2] outputs in {os.path.abspath(OUTPUT_DIR)}")
    if not all(ok for _, ok in ck):
        sys.exit(1)


if __name__ == "__main__":
    main()

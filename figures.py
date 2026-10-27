"""
figures.py -- thesis figures and tables of the revised plan (script 8).

Reads only the outputs of offerable_flex.py and reducible_power.py (site
codes); never the private key.
  1. pool_firm_heatmap_<f>_<block>h_<daytype>   month x hour heatmap of the pool's
     firm level (P10). overlay_high_load_hours() marks the high-load hours of any
     hourly series: now the area substation PROXY (lpc_11 --export-proxy), later
     the subscription series. The overlay is labelled with what it is.
  2. availability_curve_<f>_<block>h   share of hours in which at least X kW can
     be offered, for the pool and each site.
  3. classification_table.csv / .tex   sites per class by season and block
     length (LaTeX booktabs).
Each figure PNG + PDF, greyscale-readable.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

CONFIG = {
    "offer_dir": "offerable_flex_out",
    "reducible_csv": os.path.join("reducible_power_out", "reducible_power.csv"),
    "output_dir": "figures_out",
    "f": 0.5,                          # f of the 15-min sites shown
    "block_h": 1,
    "day_type": "weekday",             # "weekday" | "weekend"
    "level_col": "firm_P10_kw",
    "overlay_csv": os.path.join("screening_out", "substation_proxy_hourly.csv"),  # None = no overlay
    "overlay_label": "area substation load (proxy)",
    "overlay_value_col": "load_kw",
    "overlay_top_share": 0.25,         # mark the month x hour cells in the top 25 % of the overlay series
    "x_max_kw": None,                  # availability curve x range (None = auto)
    "lfm_min_kw": 100.0,
    "csv_sep": ";",
    "csv_decimal": ",",
}


def rd(path, cfg):
    return pd.read_csv(path, sep=cfg["csv_sep"], decimal=cfg["csv_decimal"], encoding="utf-8-sig")


def save(fig, cfg, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(cfg["output_dir"], f"{stem}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def overlay_high_load_hours(ax, series_csv, cfg, label, value_col="load_kw", top_share=0.25):
    """Outline the month x hour cells where an hourly series is in its top
    `top_share` (by its month x hour mean). Works for any hourly series with a
    timestamp column and a value column."""
    s = rd(series_csv, cfg)
    t = pd.to_datetime(s.iloc[:, 0].astype(str).str.slice(0, 19))
    v = pd.to_numeric(s[value_col], errors="coerce")
    m = pd.DataFrame({"month": t.dt.month, "hour": t.dt.hour, "v": v}).groupby(["month", "hour"])["v"].mean()
    thr = m.quantile(1 - top_share)
    for (mo, hh), val in m.items():
        if val >= thr:
            ax.add_patch(Rectangle((hh - 0.5, mo - 0.5), 1, 1, fill=False, ec="k", lw=1.4, hatch="//"))
    ax.add_patch(Rectangle((0, 0), 0, 0, fill=False, ec="k", hatch="//",
                           label=f"{label}: top {top_share:.0%} month-hour cells"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=8)


def heatmap(cfg):
    p = rd(os.path.join(cfg["offer_dir"], "pool_levels.csv"), cfg)
    p = p[np.isclose(p["f"].astype(float), cfg["f"]) & (p["block_h"] == cfg["block_h"])
          & (p["day_type"] == cfg["day_type"])]
    if p.empty:
        print("[figures] no pool levels for this f / block / day type")
        return
    grid = p.pivot_table(index="month", columns="hour_of_day", values=cfg["level_col"]).reindex(
        index=range(1, 13), columns=range(24))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    im = ax.imshow(grid.values, aspect="auto", origin="lower", cmap="Greys",
                   extent=[-0.5, 23.5, 0.5, 12.5])
    cs = ax.contour(np.arange(24), np.arange(1, 13), grid.values, levels=[cfg["lfm_min_kw"]],
                    colors="k", linewidths=1.0, linestyles="--") if np.nanmax(grid.values) >= cfg["lfm_min_kw"] else None
    ax.set_xticks(range(0, 24, 3))
    ax.set_yticks(range(1, 13))
    ax.set_xlabel("hour of day (start of hour)")
    ax.set_ylabel("month")
    ax.set_title(f"Pool firm level (P10), {cfg['block_h']} h blocks, {cfg['day_type']}s, f = {cfg['f']}"
                 + ("; dashed = 0.1 MW" if cs else "; 0.1 MW not reached"), fontsize=10)
    fig.colorbar(im, ax=ax, label="kW")
    if cfg.get("overlay_csv") and os.path.exists(cfg["overlay_csv"]):
        overlay_high_load_hours(ax, cfg["overlay_csv"], cfg, cfg["overlay_label"], cfg["overlay_value_col"],
                                cfg["overlay_top_share"])
    elif cfg.get("overlay_csv"):
        print(f"[figures] overlay file {cfg['overlay_csv']} not found -- heatmap without overlay")
    save(fig, cfg, f"pool_firm_heatmap_f{cfg['f']}_{cfg['block_h']}h_{cfg['day_type']}")


def availability(cfg):
    import offerable_flex as O
    d = O.read_input(dict(O.CONFIG, input_csv=cfg["reducible_csv"], csv_sep=cfg["csv_sep"],
                          csv_decimal=cfg["csv_decimal"]))
    ser = O.site_series(d, cfg["f"])
    ph = rd(os.path.join(cfg["offer_dir"], "pool_hourly.csv"), cfg)
    ph = ph[np.isclose(ph["f"].astype(float), cfg["f"]) & (ph["block_h"] == cfg["block_h"])]
    curves = {s: O.block_min(df["kW"], cfg["block_h"]).dropna() for s, df in ser.items()}
    pool = ph["pool_kw"].dropna()
    hi = cfg["x_max_kw"] or max([pool.max() if len(pool) else 0] + [c.max() for c in curves.values() if len(c)])
    x = np.linspace(0, hi * 1.02, 400)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    styles = ["--", ":", "-.", (0, (5, 1)), (0, (1, 2)), (0, (3, 1, 1, 1))]
    for i, (s, c) in enumerate(sorted(curves.items())):
        ax.plot(x, [(c >= v).mean() for v in x], color="0.45", lw=1, ls=styles[i % len(styles)], label=f"Site {s}")
    if len(pool):
        ax.plot(x, [(pool >= v).mean() for v in x], color="k", lw=2.2, label=f"pool ({int(ph['n_sites'].iloc[0])} sites)")
    ax.axvline(cfg["lfm_min_kw"], color="k", lw=0.8)
    ax.text(cfg["lfm_min_kw"], 1.01, " 0.1 MW", fontsize=8, va="bottom")
    ax.set_xlabel("offerable power X (kW)")
    ax.set_ylabel("share of hours with at least X")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Availability, {cfg['block_h']} h blocks, f = {cfg['f']} (15-min sites)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    save(fig, cfg, f"availability_curve_f{cfg['f']}_{cfg['block_h']}h")


def latex_booktabs(df, caption, label):
    cols = list(df.columns)
    esc = lambda x: str(x).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "l" * 3 + "r" * (len(cols) - 3) + "}", r"\toprule",
             " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(esc(v) for v in r.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def class_table(cfg):
    c = rd(os.path.join(cfg["offer_dir"], "offer_classification.csv"), cfg)
    c = c[c["f"].isna() | np.isclose(c["f"].astype(float), cfg["f"])]
    t = c.pivot_table(index=["season", "block_h"], columns="class", values="site", aggfunc="count", fill_value=0)
    for k in ["stand-alone", "pool candidate", "not relevant"]:
        if k not in t:
            t[k] = 0
    t = t[["stand-alone", "pool candidate", "not relevant"]].reset_index()
    order = {"winter": 0, "shoulder": 1, "summer": 2, "year": 3}
    t = t.sort_values(["season", "block_h"], key=lambda s: s.map(order) if s.name == "season" else s)
    t.insert(2, "f", cfg["f"])
    t.columns.name = None
    t.to_csv(os.path.join(cfg["output_dir"], "classification_table.csv"), sep=cfg["csv_sep"],
             decimal=cfg["csv_decimal"], index=False, encoding="utf-8-sig")
    tex = latex_booktabs(t.rename(columns={"block_h": "block (h)"}),
                         f"Number of sites per LFM class by season and block length (firm = P10; f = {cfg['f']} "
                         f"for 15-min sites)", "tab:lfm_classes")
    with open(os.path.join(cfg["output_dir"], "classification_table.tex"), "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(t.to_string(index=False))


def main(cfg=CONFIG):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    heatmap(cfg)
    availability(cfg)
    class_table(cfg)
    print(f"[figures] outputs in {os.path.abspath(cfg['output_dir'])}")


if __name__ == "__main__":
    main()

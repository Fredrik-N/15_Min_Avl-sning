"""
Aggregate load-shape characterization from 15-minute interval data.

This is a companion to the FHMM disaggregation script, but answers a
different question: not "what sub-loads make up this signal" but "what
does this signal's *shape* look like, and how does it vary" -- daily
profile, load duration curve, recurring day-types, seasonality, and
ramp behavior. Nothing here is disaggregated per-appliance; everything
operates on the single aggregate (or net) power column.

Pipeline
--------
1. Load & resample to a clean 15-min grid (any finer input is
   downsampled by averaging; gaps are NOT filled across long outages --
   see `max_gap_fill_periods`).
2. Summary metrics: peak/min/average demand, load factor, capacity
   factor, base-load estimate, total energy.
3. Load duration curve (LDC): demand sorted descending vs. % of time
   at/above that demand.
4. Daily profile: mean +/- percentile band by time-of-day, split into
   weekday vs weekend.
5. Day-type clustering: each calendar day becomes a 96-point shape
   vector (min-max normalized so clustering finds *shape*, not scale);
   k-means over a small grid of k, k auto-selected by silhouette score
   (mirrors the model-selection spirit of the FHMM script, just with a
   clustering-appropriate criterion instead of BIC). Reports how often
   each day-type occurs and which weekdays/months it clusters on.
6. Seasonality: monthly average profile, monthly energy & peak, and a
   day-of-week x hour-of-day heatmap.
7. Ramp-rate analysis: interval-to-interval kW swings, distribution and
   worst cases (useful for sizing/screening demand-response or storage).
8. Peak-timing analysis: what time of day the daily peak tends to land.
9. Multi-panel plot + printed report.

Notes / limitations
--------------------
- This script does not disaggregate anything; it describes the shape of
  the aggregate itself. Pair it with the FHMM script for sub-load
  detail.
- Day-type clustering normalizes each day's shape (min-max) before
  clustering, so it groups days by *pattern* (e.g. "two-shift weekday",
  "single-shift weekend") rather than by absolute magnitude. If you
  want magnitude-aware clusters instead (e.g. to separate a normal day
  from an unusually heavy one with the same shape), set
  `cluster_on_normalized_shape` to False.
- A day with too much missing data (see `min_periods_per_day`) is
  dropped from the daily-profile, clustering, and peak-timing steps but
  still contributes to the LDC and summary stats via whatever
  15-min samples it does have.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ======================================================================
# 1. USER CONFIG
# ======================================================================
CONFIG = {
    # ---- data ----
    # Extension-aware: .csv/.tsv is read with pd.read_csv, .xlsx/.xls with
    # pd.read_excel (openpyxl) -- see `_read_table()`. Öresundskraft's
    # per-meter exports come as .xlsx, not .csv.
    "csv_path": "7359991240001455955 langebergavagen 190.xlsx",
    "data_format": "long",          # "wide": one column per measurement
                                     # (the original Logistic_LA_15min.csv
                                     # shape -- aggregate_col/subtract_cols
                                     # below). "long": a real meter export
                                     # where every row is ONE reading and a
                                     # channel/quantity-code column says
                                     # which measurement that row is --
                                     # here, "Mängdkod" (Excel column C,
                                     # which is what "filter C1" refers to)
                                     # holding "kWh" (active) or "kVarh"
                                     # (reactive) -- readings interleaved as
                                     # rows rather than separate columns.
                                     # See `long_format` below and
                                     # `inspect_long_format_channels()`.
    "timestamp_col": "Tidspunkt (CET)",
    "timestamp_unit": None,         # "s" for Unix epoch seconds, None
                                     # since this column is already
                                     # parseable text ("2025-01-01 00:00")
    "aggregate_col": "Aggregate",   # gross meter / import power column
                                     # (data_format == "wide" only)
    "subtract_cols": ["PV", "CHP"], # on-site generation to net out, if any
                                     # (data_format == "wide" only)

    # ---- long-format (Mängdkod / "column C" channel-filter) settings ----
    # Only used when data_format == "long". Öresundskraft's meter export
    # stores active and reactive readings as separate ROWS of the same
    # file (same timestamp appears twice), with "Mängdkod" (column C)
    # saying which is which and "Värde" (column B) holding the reading.
    # If a different export uses different header names, run
    # `inspect_long_format_channels(cfg)` once to check, then edit below.
    "long_format": {
        "channel_col": "Mängdkod",     # column that selects the reading type
        "value_col": "Värde",          # column holding the numeric reading
        "kwh_channel_values": ["kWh"], # values in channel_col meaning
                                        # active energy/power
        "kvarh_channel_values": ["kVarh"],  # values meaning reactive
                                             # energy/power; set to [] or
                                             # None if the file has no
                                             # reactive channel
        "value_kind": "energy_per_interval_kwh",  # what "Värde" actually
                                     # is. Öresundskraft's export reports
                                     # ENERGY per 15-min interval in kWh
                                     # (e.g. 23 kWh in a quarter-hour), not
                                     # instantaneous power -- that 23 kWh
                                     # is converted to its average power
                                     # (23 / 0.25h = 92 kW) using the
                                     # interval length detected straight
                                     # from the timestamps. Set to
                                     # "power_kw" or "power_w" instead if
                                     # a given export already reports
                                     # power directly (then power_divisor
                                     # below is used as normal).
    },

    "resample_freq": "15min",       # target grid; finer input is averaged
                                     # down to this, coarser input is left
                                     # as-is (never invented via upsampling)
    "power_divisor": 1.0,           # e.g. 1000.0 to convert W -> kW.
                                     # Leave at 1.0 for the long-format
                                     # path above, whose energy->power
                                     # conversion already yields kW.
    "max_gap_fill_periods": 2,      # forward-fill gaps of up to this many
                                     # periods (missed reads); longer gaps
                                     # are left as NaN, not fabricated

    # ---- daily-profile / day-type settings ----
    "periods_per_day": 96,          # 24h / 15min
    "min_periods_per_day": 88,      # drop a day from profile/clustering/
                                     # peak-timing steps if it has fewer
                                     # valid samples than this (~92% complete)
    "weekend_days": [5, 6],         # Mon=0 ... Sun=6
    "cluster_on_normalized_shape": True,   # True: cluster by shape (min-max
                                            # normalized); False: by raw kW
    "day_type_k_options": [2, 3, 4, 5, 6, 7, 8],  # k search grid
    "random_seed": 0,

    # ---- reporting thresholds ----
    "load_duration_percentiles": [1, 5, 10, 25, 50, 75, 90, 95, 99],
    "ramp_warn_kw_per_15min": None,  # None = auto (95th pct of |ramp|);
                                      # or set a fixed kW/15min threshold

    "plot_days_for_overlay": 14,     # how many most-recent days to overlay
                                      # (lightly) behind the mean daily profile

    "output_png": "load_shape_report.png",

    # ---- ambient-temperature correlation ----
    # Coarse, cheap proxy for how much of the load is likely cooling/
    # compressor-driven: correlate the load against outdoor temperature
    # over the same period. Source is a local SMHI "öppna data" hourly
    # air-temperature export (semicolon-delimited, metadata header rows
    # before the real table, times in UTC) -- see `load_smhi_temperature`.
    "weather_csv_path": "smhi_helsingborg_2025.csv",
    "weather_timestamps_are_utc": True,   # SMHI's "Tid (UTC)" column is
                                     # UTC; converted to Europe/Stockholm
                                     # local time to align with the meter
                                     # data's "Tidspunkt (CET)" column.
                                     # NOTE: that column name suggests
                                     # fixed CET (UTC+1) year-round rather
                                     # than DST-aware local time -- if
                                     # correlation looks oddly weak only
                                     # in summer, that's the first thing
                                     # to check (see docstring below).
    "temp_corr_lag_hours_search": list(range(0, 7)),  # 0 = same-hour; try
                                     # a few positive lags too (load
                                     # responding to temperature with
                                     # delay, e.g. thermal mass/deadband).
                                     # Kept small and non-negative since
                                     # physically the load should follow
                                     # temperature, not precede it.

    # ---- grid constraint-window metrics ----
    # Peak-during-constraint-window needs to know WHEN the grid is
    # actually constrained. Rather than guessing a fixed hour range,
    # put Öresundskraft's own grid consumption data here (any period is
    # fine -- it just needs to cover a representative set of hours) and
    # the constrained hours are derived from it directly: the hours of
    # the day where the grid's own average load sits in the top
    # `constraint_percentile` of its typical daily profile.
    #
    # Leave csv_path as None/"" until that data is available -- every
    # constraint-window-dependent metric (this one now, willingness-proxy
    # beta later) is then explicitly SKIPPED with a clear one-line reason
    # printed, never silently estimated or filled with a guessed default.
    "grid_consumption": {
        "csv_path": None,           # None or "" -> skip constraint-window
                                     # metrics entirely. Set to the grid
                                     # consumption file's path to enable.
        "data_format": "wide",      # "wide": one timestamp_col + one
                                     # value_col. "long": same channel-
                                     # filter idea as the customer meter
                                     # loader, using channel_col/value_col/
                                     # channel_values below (in case
                                     # Öresundskraft exports grid data in
                                     # the same Mängdkod-style long format).
        "timestamp_col": "Timestamp",
        "value_col": "Value",
        "timestamp_unit": None,     # "s" for Unix epoch seconds, else None
        # ---- only used when data_format == "long" ----
        "channel_col": "Mängdkod",
        "channel_values": ["kWh"],  # which channel value(s) to keep;
                                     # only matters for units, not for
                                     # deriving the constraint window
                                     # (see docstring of
                                     # derive_constraint_window_hours)
        "constraint_percentile": 0.75,  # hours whose average grid load is
                                     # >= this percentile of the grid's
                                     # own hour-of-day profile count as
                                     # "constrained". 0.75 = the top
                                     # quarter of hours by typical grid
                                     # demand; tune once real data is in.
    },
}


# ======================================================================
# 2. Data loading
# ======================================================================
def _read_table(path):
    """
    Read a data file regardless of whether it's a delimited text file or
    an Excel workbook. Öresundskraft's meter exports come as .xlsx, which
    pd.read_csv cannot parse (it's a binary/zip format, not text -- that
    mismatch is exactly what produces a
    "UnicodeDecodeError: 'utf-8' codec can't decode byte ..." when you
    point read_csv at an .xlsx file).
    """
    lower = str(path).lower()
    if lower.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _infer_interval_hours(index):
    """Median spacing between consecutive timestamps, in hours -- used to
    convert an energy-per-interval reading (kWh) to average power (kW)
    without assuming the interval length up front."""
    diffs = pd.Series(index).sort_values().diff().dropna()
    if diffs.empty:
        raise ValueError("Can't infer interval length: fewer than 2 timestamps.")
    return diffs.median().total_seconds() / 3600.0


def inspect_long_format_channels(cfg, n_examples=3):
    """
    Diagnostic helper -- run this FIRST against a real long-format file,
    before setting `long_format` in CONFIG. Prints every distinct value
    found in the configured channel column (e.g. "C1"), how many rows
    carry it, and a few example readings, so you can tell which value(s)
    mean "active power/energy (kWh)" and which mean "reactive power/
    energy (kVarh)" and copy the exact strings into
    CONFIG["long_format"]["kwh_channel_values"] /
    ["kvarh_channel_values"].

    Does not require data_format to be set to "long" yet -- it just reads
    the raw CSV and reports on the channel column.
    """
    lf = cfg["long_format"]
    channel_col = lf["channel_col"]
    value_col = lf["value_col"]

    df = _read_table(cfg["csv_path"])
    if channel_col not in df.columns:
        raise KeyError(
            f"channel_col '{channel_col}' not found in {cfg['csv_path']}. "
            f"Columns present: {list(df.columns)}"
        )

    print(f"\n[inspect] '{cfg['csv_path']}' -- channel column '{channel_col}'")
    counts = df[channel_col].value_counts(dropna=False)
    for val, n in counts.items():
        examples = df.loc[df[channel_col] == val, value_col].head(n_examples).tolist()
        print(f"  {val!r}: {n} row(s), e.g. {value_col}={examples}")
    print("[inspect] set kwh_channel_values / kvarh_channel_values in "
          "CONFIG['long_format'] to whichever value(s) above correspond "
          "to active-power/energy and reactive-power/energy respectively.")
    return counts


def _select_channel_rows(df, channel_col, wanted_values):
    """Case-/whitespace-insensitive match against a list of channel values."""
    wanted = {str(v).strip().lower() for v in wanted_values}
    mask = df[channel_col].astype(str).str.strip().str.lower().isin(wanted)
    return df[mask]


def load_series_long_format(cfg, return_reactive=False):
    """
    Load a long-format meter export: every row is a single reading, and a
    channel column (cfg["long_format"]["channel_col"], e.g. "C1")
    indicates which measurement that row is (active power/energy vs
    reactive power/energy, typically). This filters to the rows meaning
    "active" to build the P series used everywhere else in this script,
    and -- if requested and present -- also builds a Q (reactive) series
    from the rows meaning "reactive".

    Unlike the wide-format `aggregate_col`/`subtract_cols` path, there's
    no netting of on-site generation here; if that's needed for a given
    file, subtract it before/after this call.
    """
    lf = cfg["long_format"]
    channel_col = lf["channel_col"]
    value_col = lf["value_col"]

    df = _read_table(cfg["csv_path"])

    if channel_col not in df.columns:
        raise KeyError(
            f"channel_col '{channel_col}' not found in {cfg['csv_path']}. "
            f"Columns present: {list(df.columns)}. Run "
            f"inspect_long_format_channels(cfg) to check the real column "
            f"names/values first."
        )

    if cfg["timestamp_unit"]:
        df[cfg["timestamp_col"]] = pd.to_datetime(
            df[cfg["timestamp_col"]], unit=cfg["timestamp_unit"], errors="coerce"
        )
    else:
        df[cfg["timestamp_col"]] = pd.to_datetime(
            df[cfg["timestamp_col"]], errors="coerce"
        )
    df = df[df[cfg["timestamp_col"]].notnull()]

    def _series_for(channel_values):
        if not channel_values:
            return None
        rows = _select_channel_rows(df, channel_col, channel_values)
        if rows.empty:
            print(f"[load_series_long_format] warning: no rows matched "
                  f"{channel_values!r} in column '{channel_col}'")
            return None
        s = (
            rows.set_index(cfg["timestamp_col"])[value_col]
            .astype(float)
            .sort_index()
        )
        return s[~s.index.duplicated(keep="first")]

    p_raw = _series_for(lf["kwh_channel_values"])
    if p_raw is None or p_raw.empty:
        raise ValueError(
            f"No active-power/energy rows found for "
            f"kwh_channel_values={lf['kwh_channel_values']!r}. Run "
            f"inspect_long_format_channels(cfg) to see the real values in "
            f"'{channel_col}' and fix the config."
        )

    q_raw = _series_for(lf.get("kvarh_channel_values")) if return_reactive else None

    # Öresundskraft's export reports ENERGY per interval (kWh), not
    # instantaneous power -- convert to average power over that interval
    # using the actual spacing between timestamps (don't assume 15 min;
    # detect it, so this still works if a file turns out to be hourly or
    # 1-min instead).
    if lf.get("value_kind", "energy_per_interval_kwh") == "energy_per_interval_kwh":
        interval_hours = _infer_interval_hours(p_raw.index)
        print(f"[load_series_long_format] detected interval "
              f"{interval_hours * 60:.1f} min -- converting kWh/kVarh per "
              f"interval to average kW/kVAr")
        p_raw = p_raw / interval_hours
        if q_raw is not None:
            q_raw = q_raw / interval_hours

    return p_raw, q_raw


def _finish_power_series(raw, cfg):
    """Shared tail-end processing for either loading path: unit convert,
    downsample to the target grid, short-gap fill."""
    power = (raw / cfg["power_divisor"]).rename("power")
    power = power.resample(cfg["resample_freq"]).mean()
    power = power.ffill(limit=cfg["max_gap_fill_periods"])
    return power


def load_series(cfg, return_reactive=False):
    """
    Build the clean 15-min power series the rest of the script uses.
    Dispatches on cfg["data_format"]:
      - "wide" (default): the original one-column-per-measurement CSV,
        with on-site generation netted out via aggregate_col/subtract_cols.
      - "long": a real meter export where a channel column (e.g. "C1")
        selects between measurement types stored as rows; see
        `load_series_long_format` and `inspect_long_format_channels`.

    Pass return_reactive=True to also get back a reactive-power series
    (only meaningful/available for "long" files that carry a kVarh
    channel); the reactive series is None whenever it isn't requested,
    isn't configured, or isn't present in the file.
    """
    if cfg.get("data_format", "wide") == "long":
        p_raw, q_raw = load_series_long_format(cfg, return_reactive=return_reactive)
        power = _finish_power_series(p_raw, cfg)
        if not return_reactive:
            return power
        reactive = _finish_power_series(q_raw, cfg) if q_raw is not None else None
        return power, reactive

    # ---- original wide-format path ----
    df = _read_table(cfg["csv_path"])

    if cfg["timestamp_unit"]:
        df[cfg["timestamp_col"]] = pd.to_datetime(
            df[cfg["timestamp_col"]], unit=cfg["timestamp_unit"], errors="coerce"
        )
    else:
        df[cfg["timestamp_col"]] = pd.to_datetime(
            df[cfg["timestamp_col"]], errors="coerce"
        )

    df = df.set_index(cfg["timestamp_col"])
    df = df[df.index.notnull()]
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    agg = df.get(cfg["aggregate_col"], pd.Series(0.0, index=df.index)).fillna(0.0)
    for col in cfg["subtract_cols"]:
        agg = agg - df.get(col, pd.Series(0.0, index=df.index)).fillna(0.0)

    power = _finish_power_series(agg, cfg)

    if not return_reactive:
        return power
    return power, None


# ======================================================================
# 3. Summary metrics
# ======================================================================
def summary_metrics(power, cfg):
    valid = power.dropna()
    freq_hours = pd.Timedelta(cfg["resample_freq"]).total_seconds() / 3600.0

    peak = valid.max()
    min_demand = valid.min()
    avg_demand = valid.mean()
    total_energy_kwh = valid.sum() * freq_hours
    load_factor = avg_demand / peak if peak > 0 else np.nan
    base_load = valid.quantile(0.05)
    peak_to_base = peak / base_load if base_load > 0 else np.nan
    span_days = (valid.index.max() - valid.index.min()).total_seconds() / 86400.0
    coverage = valid.count() / max(len(power), 1)

    return {
        "span_days": span_days,
        "coverage_fraction": coverage,
        "peak_kw": peak,
        "min_kw": min_demand,
        "avg_kw": avg_demand,
        "base_load_kw_p5": base_load,
        "peak_to_base_ratio": peak_to_base,
        "load_factor": load_factor,
        "total_energy_kwh": total_energy_kwh,
    }


def print_summary(metrics):
    print("\n[summary] aggregate load characteristics")
    print(f"  data span:            {metrics['span_days']:.1f} days "
          f"({metrics['coverage_fraction']:.1%} of periods have data)")
    print(f"  peak demand:          {metrics['peak_kw']:.1f} kW")
    print(f"  min demand:           {metrics['min_kw']:.1f} kW")
    print(f"  average demand:       {metrics['avg_kw']:.1f} kW")
    print(f"  base load (p5):       {metrics['base_load_kw_p5']:.1f} kW")
    print(f"  peak / base ratio:    {metrics['peak_to_base_ratio']:.2f}x")
    print(f"  load factor:          {metrics['load_factor']:.2%}  "
          f"(avg/peak -- higher = flatter, more efficient use of capacity)")
    print(f"  total energy:         {metrics['total_energy_kwh']:,.0f} kWh")


# ======================================================================
# 4. Load duration curve
# ======================================================================
def load_duration_curve(power, cfg):
    valid = power.dropna().sort_values(ascending=False).to_numpy()
    n = len(valid)
    pct_time = np.arange(1, n + 1) / n * 100.0

    table = {}
    for p in cfg["load_duration_percentiles"]:
        # demand exceeded p% of the time
        idx = int(np.clip(round(p / 100.0 * n), 0, n - 1))
        table[p] = valid[idx]
    return pct_time, valid, table


def print_ldc_table(table):
    print("\n[load duration] demand exceeded X% of the time")
    for p in sorted(table):
        print(f"  exceeded {p:>2d}% of time: {table[p]:.1f} kW")


# ======================================================================
# 5. Daily profile (weekday vs weekend, with percentile band)
# ======================================================================
def daily_profile(power, cfg):
    valid = power.dropna()
    tod = valid.index.hour * 60 + valid.index.minute  # minute-of-day
    is_weekend = valid.index.dayofweek.isin(cfg["weekend_days"])

    def profile_for(mask):
        sub = valid[mask]
        sub_tod = tod[mask]
        grouped = pd.Series(sub.to_numpy(), index=sub_tod).groupby(level=0)
        return grouped.mean(), grouped.quantile(0.10), grouped.quantile(0.90)

    wd_mean, wd_p10, wd_p90 = profile_for(~is_weekend)
    we_mean, we_p10, we_p90 = profile_for(is_weekend)
    return {
        "weekday": (wd_mean, wd_p10, wd_p90),
        "weekend": (we_mean, we_p10, we_p90),
    }


# ======================================================================
# 6. Day-type clustering
# ======================================================================
def build_daily_matrix(power, cfg):
    """
    Reshape the series into one row per calendar day (periods_per_day
    columns). Days with too much missing data are dropped. Returns the
    raw matrix, a shape-normalized matrix (each row min-max scaled to
    [0, 1] so clustering groups *pattern*, not magnitude), and the dates
    kept.
    """
    ppd = cfg["periods_per_day"]
    df = power.to_frame("power")
    df["date"] = df.index.date
    df["period"] = df.index.hour * (60 // 15) + df.index.minute // 15

    pivot = df.pivot_table(index="date", columns="period", values="power")
    pivot = pivot.reindex(columns=range(ppd))

    valid_counts = pivot.notna().sum(axis=1)
    keep = valid_counts >= cfg["min_periods_per_day"]
    pivot = pivot[keep]

    # fill any small remaining gaps within a kept day via interpolation
    pivot = pivot.interpolate(axis=1, limit_direction="both")

    raw = pivot.to_numpy()
    row_min = raw.min(axis=1, keepdims=True)
    row_max = raw.max(axis=1, keepdims=True)
    span = np.clip(row_max - row_min, 1e-6, None)
    normalized = (raw - row_min) / span

    return raw, normalized, pd.to_datetime(pivot.index)


def auto_select_day_types(matrix, cfg):
    """
    Search k over cfg['day_type_k_options'] and pick the one with the
    best silhouette score (higher = better-separated, more cohesive
    clusters). This is the clustering analogue of the FHMM script's BIC
    search: let the data pick how many recurring day-shapes exist rather
    than assuming a fixed number.
    """
    n_days = matrix.shape[0]
    results = []
    print(f"\n[day-type search] evaluating k in {cfg['day_type_k_options']} "
          f"over {n_days} day(s)...")
    for k in cfg["day_type_k_options"]:
        if k >= n_days:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg["random_seed"])
        labels = km.fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels)
        results.append((score, k, km, labels))
        print(f"  k={k}:  silhouette={score:.3f}")

    if not results:
        # fall back to k=1 (no meaningful clustering possible)
        return 1, np.zeros(n_days, dtype=int), None

    results.sort(key=lambda r: -r[0])
    best_score, best_k, best_km, best_labels = results[0]
    print(f"[day-type search] selected k={best_k} (silhouette={best_score:.3f})")
    return best_k, best_labels, best_km


def day_type_report(dates, labels, raw_matrix, cfg):
    print("\n[day-types] recurring daily patterns")
    n_days = len(dates)
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for c in sorted(set(labels)):
        mask = labels == c
        n = mask.sum()
        days_in_cluster = dates[mask]
        dow_counts = pd.Series(days_in_cluster.dayofweek).value_counts()
        dow_share = {dow_names[d]: f"{cnt}" for d, cnt in dow_counts.items()}
        avg_peak = raw_matrix[mask].max(axis=1).mean()
        avg_mean = raw_matrix[mask].mean(axis=1).mean()
        print(f"  type {c}: {n} day(s) ({n / n_days:.1%} of days), "
              f"avg peak={avg_peak:.1f} kW, avg mean={avg_mean:.1f} kW, "
              f"weekday mix={dow_share}")


# ======================================================================
# 7. Seasonality: monthly pattern + day-of-week x hour heatmap
# ======================================================================
def monthly_summary(power, cfg):
    valid = power.dropna()
    freq_hours = pd.Timedelta(cfg["resample_freq"]).total_seconds() / 3600.0
    grouped = valid.groupby(valid.index.to_period("M"))
    out = grouped.agg(["mean", "max"])  # columns: "mean", "max"
    out.columns = ["mean_kw", "max_kw"]
    out["energy_kwh"] = grouped.sum() * freq_hours
    return out


def dow_hour_heatmap_matrix(power):
    valid = power.dropna()
    df = valid.to_frame("power")
    df["dow"] = df.index.dayofweek
    df["hour"] = df.index.hour
    pivot = df.pivot_table(index="dow", columns="hour", values="power", aggfunc="mean")
    pivot = pivot.reindex(index=range(7), columns=range(24))
    return pivot


# ======================================================================
# 8. Ramp-rate analysis
# ======================================================================
def ramp_analysis(power, cfg):
    valid = power.dropna()
    ramps = valid.diff().dropna()

    if cfg["ramp_warn_kw_per_15min"] is None:
        warn_thresh = ramps.abs().quantile(0.95)
    else:
        warn_thresh = cfg["ramp_warn_kw_per_15min"]

    worst_up = ramps.nlargest(5)
    worst_down = ramps.nsmallest(5)

    print("\n[ramp rates] interval-to-interval swings")
    print(f"  95th pct |ramp|:  {ramps.abs().quantile(0.95):.1f} kW / interval")
    print(f"  max ramp up:      {ramps.max():.1f} kW  at {ramps.idxmax()}")
    print(f"  max ramp down:    {ramps.min():.1f} kW  at {ramps.idxmin()}")
    print(f"  warn threshold:   {warn_thresh:.1f} kW / interval "
          f"({'auto p95' if cfg['ramp_warn_kw_per_15min'] is None else 'fixed'})")
    n_over = (ramps.abs() > warn_thresh).sum()
    print(f"  intervals over threshold: {n_over} "
          f"({n_over / len(ramps):.2%} of intervals)")

    return ramps, warn_thresh, worst_up, worst_down


# ======================================================================
# 8b. Ambient-temperature correlation
# ======================================================================
def load_smhi_temperature(cfg):
    """
    Parse an SMHI "öppna data" hourly air-temperature export into a clean
    Series indexed by local (Europe/Stockholm) naive timestamps.

    The file has a handful of metadata header blocks (station name,
    parameter description, measurement period/coordinates) before the
    real data table, which starts at the row beginning "Datum;Tid (UTC);
    Lufttemperatur;...". Everything above that is skipped rather than
    assumed to be a fixed number of lines, so this keeps working if SMHI
    changes how many header rows it emits.

    SMHI timestamps are UTC; they're localized to UTC and converted to
    Europe/Stockholm (which correctly applies the CET/CEST DST switch),
    then the tz is dropped so the result lines up with the meter data's
    naive "Tidspunkt (CET)" timestamps. If the meter export actually
    means FIXED CET (UTC+1) year-round rather than DST-aware local clock
    time -- some Swedish DSO exports do this -- summer correlations will
    be off by up to an hour; if temp_corr_best_lag_hours comes back
    suspiciously different between winter and summer months, that
    mismatch is the first thing to check.
    """
    path = cfg["weather_csv_path"]
    with open(path, encoding="utf-8-sig") as f:
        lines = f.readlines()

    header_idx = next(
        i for i, line in enumerate(lines) if line.startswith("Datum;Tid")
    )
    df = pd.read_csv(path, sep=";", skiprows=header_idx, encoding="utf-8-sig")

    ts_utc = pd.to_datetime(
        df["Datum"] + " " + df["Tid (UTC)"], errors="coerce"
    )
    temp = pd.Series(df["Lufttemperatur"].to_numpy(), index=ts_utc, name="temp_c")
    temp = temp[temp.index.notnull()]
    temp = temp[~temp.index.duplicated(keep="first")]
    temp = temp.sort_index()

    if cfg.get("weather_timestamps_are_utc", True):
        temp.index = (
            temp.index.tz_localize("UTC")
            .tz_convert("Europe/Stockholm")
            .tz_localize(None)
        )
        # The autumn DST fall-back (e.g. Oct 26 2025, 03:00 CEST -> 02:00
        # CET) makes one UTC hour map onto a local clock time that
        # already occurred an hour earlier. While still tz-aware the two
        # are distinct (different UTC offsets), so dedup has to happen
        # AFTER dropping the tz, once they actually collide as naive
        # timestamps; keep the first (CEST) occurrence.
        temp = temp[~temp.index.duplicated(keep="first")]

    return temp


def temperature_correlation(power, weather_series, cfg):
    """
    Correlate the load series against outdoor temperature: a cheap,
    aggregate-level proxy for how much of the load is likely cooling/
    compressor-driven, usable before any process-level (1-second) data
    exists.

    Reports the same-timestamp ("lag 0") Pearson correlation, plus the
    strongest correlation found over a small search of positive lags
    (temperature leading load by cfg["temp_corr_lag_hours_search"] hours)
    -- a cooling load driven by ambient heat gain often responds with a
    delay (thermal mass, control deadbands), so lag 0 alone can
    understate a real relationship.

    Interpretation: high correlation (either at lag 0 or at the best lag)
    is a good sign for a cooling-driven-flexibility candidate; correlation
    near zero suggests the load is driven by something else (a production
    schedule, occupancy) rather than weather.
    """
    temp_hourly = weather_series.resample("h").mean()
    # Upsample hourly temperature onto the load's (finer) grid via linear
    # interpolation -- this fills in *between* real hourly readings, it
    # never invents resolution the source doesn't have.
    temp_on_grid = temp_hourly.reindex(
        power.index.union(temp_hourly.index)
    ).interpolate(method="time").reindex(power.index)

    valid = pd.DataFrame({"power": power, "temp": temp_on_grid}).dropna()
    if len(valid) < 10:
        print("[temp correlation] warning: fewer than 10 overlapping "
              "samples between load and weather data -- check date "
              "ranges/timezone alignment.")
        return {
            "temp_corr_same_hour": np.nan,
            "temp_corr_best_lag_hours": np.nan,
            "temp_corr_best_lag_value": np.nan,
            "temp_corr_n_samples": len(valid),
        }

    same_hour_corr = valid["power"].corr(valid["temp"])

    periods_per_hour = round(3600.0 / pd.Timedelta(cfg["resample_freq"]).total_seconds())
    best_lag_h, best_corr = 0, same_hour_corr
    for lag_h in cfg["temp_corr_lag_hours_search"]:
        shifted = valid["temp"].shift(lag_h * periods_per_hour)
        c = valid["power"].corr(shifted)
        if pd.notna(c) and abs(c) > abs(best_corr):
            best_lag_h, best_corr = lag_h, c

    print("\n[temp correlation] load vs. outdoor temperature")
    print(f"  same-hour (lag 0):   r = {same_hour_corr:+.3f}  "
          f"(n={len(valid)} overlapping samples)")
    print(f"  best lag:            {best_lag_h}h  r = {best_corr:+.3f}")
    if abs(best_corr) > 0.5:
        print("  -> strong temperature relationship: good sign for a "
              "cooling/compressor-driven-flexibility candidate")
    elif abs(best_corr) > 0.25:
        print("  -> moderate relationship: some temperature-driven load, "
              "likely mixed with other drivers")
    else:
        print("  -> weak relationship: load looks driven by something "
              "other than weather (schedule, occupancy, process)")

    return {
        "temp_corr_same_hour": same_hour_corr,
        "temp_corr_best_lag_hours": best_lag_h,
        "temp_corr_best_lag_value": best_corr,
        "temp_corr_n_samples": len(valid),
    }


# ======================================================================
# 8c. Grid constraint-window metrics
# ======================================================================
def load_grid_consumption(cfg):
    """
    Load Öresundskraft's own grid consumption series, used only to work
    out WHEN the grid is constrained -- not compared in magnitude against
    any customer's load, so its units/resolution don't need to match
    theirs. Supports the same wide (one timestamp + one value column) or
    long (Mängdkod-style channel filter) shapes as the customer meter
    loader; see `grid_consumption` in CONFIG.
    """
    gc = cfg["grid_consumption"]
    df = _read_table(gc["csv_path"])

    ts_col = gc["timestamp_col"]
    if gc.get("timestamp_unit"):
        df[ts_col] = pd.to_datetime(df[ts_col], unit=gc["timestamp_unit"], errors="coerce")
    else:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df[df[ts_col].notnull()]

    if gc.get("data_format", "wide") == "long":
        channel_col = gc["channel_col"]
        if channel_col not in df.columns:
            raise KeyError(
                f"channel_col '{channel_col}' not found in "
                f"{gc['csv_path']}. Columns present: {list(df.columns)}"
            )
        df = _select_channel_rows(df, channel_col, gc["channel_values"])

    s = (
        df.set_index(ts_col)[gc["value_col"]]
        .astype(float)
        .sort_index()
    )
    s = s[~s.index.duplicated(keep="first")]
    return s.rename("grid_load")


def derive_constraint_window_hours(grid_series, cfg):
    """
    Turn a raw grid consumption series into a set of "constrained" hours
    of the day: those whose average grid load (averaged across every day
    in the series) sits at or above `constraint_percentile` of the
    grid's own typical daily profile.

    Deliberately relative, not absolute -- it only compares the grid to
    ITSELF across hours, so the series' units (kWh/interval, kW, MW,
    whatever the real export turns out to use) don't matter as long as
    they're consistent across the file, which a single export always is.
    """
    hourly_avg = grid_series.groupby(grid_series.index.hour).mean()
    threshold = hourly_avg.quantile(cfg["grid_consumption"]["constraint_percentile"])
    constrained_hours = sorted(hourly_avg[hourly_avg >= threshold].index.tolist())

    print(f"\n[constraint window] derived from grid consumption data "
          f"(top {1 - cfg['grid_consumption']['constraint_percentile']:.0%} "
          f"of hours by average grid load):")
    print(f"  constrained hours: {constrained_hours}")
    return constrained_hours, hourly_avg


def constraint_window_metrics(power, constraint_hours, overall_metrics, cfg):
    """
    The customer's own average/peak load specifically during the grid's
    constrained hours -- a customer whose load is high specifically when
    the GRID is stressed is a more valuable DR candidate than one whose
    peak happens to fall outside that window, even if the two customers'
    overall peaks are identical.
    """
    valid = power.dropna()
    in_window = valid[valid.index.hour.isin(constraint_hours)]

    if in_window.empty:
        print("[constraint window] warning: no load samples fall in the "
              "derived constrained hours -- check timezone alignment "
              "between the meter data and the grid consumption data.")
        return {
            "constraint_window_avg_kw": np.nan,
            "constraint_window_peak_kw": np.nan,
            "constraint_window_peak_frac_of_overall_peak": np.nan,
            "constraint_window_hours": constraint_hours,
        }

    avg_in_window = in_window.mean()
    peak_in_window = in_window.max()
    overall_peak = overall_metrics["peak_kw"]
    peak_frac = peak_in_window / overall_peak if overall_peak > 0 else np.nan

    print(f"\n[constraint window] this customer's load during those hours")
    print(f"  avg load in window:   {avg_in_window:.1f} kW")
    print(f"  peak load in window:  {peak_in_window:.1f} kW")
    print(f"  as fraction of overall peak: {peak_frac:.1%}")

    return {
        "constraint_window_avg_kw": avg_in_window,
        "constraint_window_peak_kw": peak_in_window,
        "constraint_window_peak_frac_of_overall_peak": peak_frac,
        "constraint_window_hours": constraint_hours,
    }


# ======================================================================
# 9. Peak-timing analysis
# ======================================================================
def peak_timing(raw_matrix, dates, cfg):
    ppd = cfg["periods_per_day"]
    peak_period = raw_matrix.argmax(axis=1)
    peak_hour = peak_period * (1440 // ppd) / 60.0  # fractional hour

    print("\n[peak timing] when the daily peak tends to land")
    hours_binned = np.floor(peak_hour).astype(int)
    counts = pd.Series(hours_binned).value_counts().sort_index()
    for h, c in counts.items():
        bar = "#" * int(c / max(counts.max(), 1) * 30)
        print(f"  {h:02d}:00  {c:>4d} day(s)  {bar}")
    return peak_hour


# ======================================================================
# 10. Plotting
# ======================================================================
def plot_report(power, profiles, pct_time, ldc_values, raw_matrix,
                 normalized_matrix, labels, dates, dow_hour_matrix,
                 monthly, cfg):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # -- daily profile (weekday vs weekend) --
    ax = axes[0, 0]
    for label, (mean, p10, p90) in profiles.items():
        x = mean.index / 60.0  # hour of day
        ax.plot(x, mean.values, linewidth=2, label=label)
        ax.fill_between(x, p10.values, p90.values, alpha=0.15)
    ax.set_title("Daily profile (mean, 10-90th pct band)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Power (kW)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -- load duration curve --
    ax = axes[0, 1]
    ax.plot(pct_time, ldc_values, color="darkred", linewidth=1.5)
    ax.set_title("Load duration curve")
    ax.set_xlabel("% of time at/above demand")
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.3)

    # -- day-type cluster centroids --
    ax = axes[0, 2]
    ppd = cfg["periods_per_day"]
    x = np.arange(ppd) * (1440 // ppd) / 60.0
    matrix_for_plot = normalized_matrix if cfg["cluster_on_normalized_shape"] else raw_matrix
    for c in sorted(set(labels)):
        mask = labels == c
        centroid = matrix_for_plot[mask].mean(axis=0)
        ax.plot(x, centroid, linewidth=2, label=f"type {c} (n={mask.sum()})")
    ax.set_title("Day-type shapes" + (" (normalized)" if cfg["cluster_on_normalized_shape"] else ""))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Normalized power" if cfg["cluster_on_normalized_shape"] else "Power (kW)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- day-of-week x hour heatmap --
    ax = axes[1, 0]
    cmap = LinearSegmentedColormap.from_list("load", ["#f7fbff", "#08306b"])
    im = ax.imshow(dow_hour_matrix.to_numpy(), aspect="auto", cmap=cmap)
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_title("Avg power: day-of-week x hour")
    ax.set_xlabel("Hour of day")
    fig.colorbar(im, ax=ax, label="kW")

    # -- monthly energy & peak --
    ax = axes[1, 1]
    months = monthly.index.astype(str)
    ax2 = ax.twinx()
    ax.bar(months, monthly["energy_kwh"], color="steelblue", alpha=0.6,
           label="Energy (kWh)")
    ax2.plot(months, monthly["max_kw"], color="darkorange",
              marker="o", linewidth=1.5, label="Peak (kW)")
    ax.set_title("Monthly energy & peak demand")
    ax.set_ylabel("Energy (kWh)")
    ax2.set_ylabel("Peak (kW)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # -- recent overlay of raw daily shapes vs mean --
    ax = axes[1, 2]
    n_overlay = min(cfg["plot_days_for_overlay"], raw_matrix.shape[0])
    for row in raw_matrix[-n_overlay:]:
        ax.plot(x, row, color="grey", alpha=0.25, linewidth=0.8)
    ax.plot(x, raw_matrix.mean(axis=0), color="black", linewidth=2, label="Mean (all days)")
    ax.set_title(f"Last {n_overlay} days vs. overall mean shape")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Power (kW)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(cfg["output_png"], dpi=120)
    plt.show()


# ======================================================================
# 11. Main
# ======================================================================
def main(cfg=CONFIG):
    power = load_series(cfg)

    metrics = summary_metrics(power, cfg)
    print_summary(metrics)

    pct_time, ldc_values, ldc_table = load_duration_curve(power, cfg)
    print_ldc_table(ldc_table)

    profiles = daily_profile(power, cfg)

    raw_matrix, normalized_matrix, dates = build_daily_matrix(power, cfg)
    cluster_matrix = normalized_matrix if cfg["cluster_on_normalized_shape"] else raw_matrix
    best_k, labels, _ = auto_select_day_types(cluster_matrix, cfg)
    day_type_report(dates, labels, raw_matrix, cfg)

    monthly = monthly_summary(power, cfg)
    dow_hour_matrix = dow_hour_heatmap_matrix(power)

    ramp_analysis(power, cfg)
    peak_timing(raw_matrix, dates, cfg)

    temp_corr = None
    if cfg.get("weather_csv_path"):
        weather = load_smhi_temperature(cfg)
        temp_corr = temperature_correlation(power, weather, cfg)

    constraint_window = None
    grid_csv_path = cfg.get("grid_consumption", {}).get("csv_path")
    if grid_csv_path:
        grid_load = load_grid_consumption(cfg)
        constraint_hours, _ = derive_constraint_window_hours(grid_load, cfg)
        constraint_window = constraint_window_metrics(power, constraint_hours, metrics, cfg)
    else:
        print("\n[constraint window] no grid consumption data configured "
              "(CONFIG['grid_consumption']['csv_path'] is empty) -- "
              "skipping constraint-window relevancy metrics. Nothing is "
              "guessed or defaulted in its place; this and anything "
              "downstream of it (e.g. the willingness-proxy beta ratio, "
              "once built) stay explicitly None/N/A until this is set.")

    plot_report(power, profiles, pct_time, ldc_values, raw_matrix,
                normalized_matrix, labels, dates, dow_hour_matrix, monthly, cfg)

    return {
        "power": power,
        "metrics": metrics,
        "ldc_table": ldc_table,
        "day_type_labels": labels,
        "day_type_dates": dates,
        "monthly": monthly,
        "temp_corr": temp_corr,
        "constraint_window": constraint_window,
    }


if __name__ == "__main__":
    main()

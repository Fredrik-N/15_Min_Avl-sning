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
   8b. Ambient-temperature correlation (SMHI hourly data).
   8c. Grid constraint window: Öresundskraft's own grid load (HOURLY
       Netto_kWh, Excel column V from row 2 down) is read, converted to
       average kW, and held constant over each hour's four quarter-hours
       so it sits on the same 15-min grid as the customer series. The
       grid's highest-load hours of the day define the "constrained"
       window the customer is then scored against.
9. Multi-panel plot + printed report.
10. Site comparison file: one row of headline results per site is
    written to `results_csv_path`. Every run adds its site; running a
    site again replaces that site's row, so the file always holds the
    latest result for every site analysed, side by side.
11. Candidate selection: every time the comparison file is written,
    (1) every site is checked against pass/fail filters (coverage, winter
    load factor, weekend/weekday ratio, winter temperature correlation),
    (2) the sites that pass are ranked by their average load above base in
    the grid's winter constraint window (kW) and the top ones are selected
    until they hold 80 % of that kW, (3) hint columns (solar, on/off
    cycling) support the final choice by hand. See CONFIG["selection"].
12. Rerun all sites: `python lpc_8.py --rerun-all` re-analyses every site
    already in the comparison file with the current script and settings
    (plots saved, not shown), then re-ranks.

Command line:
    python lpc_8.py                     run CONFIG["csv_path"] (as before)
    python lpc_8.py A.xlsx B.xlsx       run these meter files, one after another
    python lpc_8.py --rerun-all         rerun every site in the comparison file
    python lpc_8.py --rank-only         only recompute the selection (and print it with the
                                        robustness check)

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

import os

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

    # ---- site comparison file ----
    # One row per site with the headline results of every run, so sites
    # can be compared in one place (opens directly in Excel). A new site
    # adds a row; a site that is already in the file gets its row
    # REPLACED by the new run. Set to None/"" to skip.
    "results_csv_path": "site_comparison.csv",
    "site_id": None,                # name of this site in the comparison
                                     # file. None = the meter file name
                                     # without extension, e.g.
                                     # "7359991240001455955 langebergavagen 190".
                                     # Rows are matched on this (case and
                                     # surrounding spaces ignored), so keep
                                     # it the same between runs of one site.
    "results_csv_sep": ";",         # ";" + decimal comma = what Swedish
    "results_csv_decimal": ",",     # Excel opens directly in columns. Use
                                     # "," and "." for English Excel / pandas.
    "meter_folder": None,           # where --rerun-all looks for a site's
                                     # meter file when the comparison file
                                     # only has its name (rows written by
                                     # lpc_7). None = the comparison file's
                                     # own folder, then the working directory.
                                     # New rows also store the full path.
    "show_plot": True,              # False: save the figure, don't open a
                                     # window (always False in --rerun-all
                                     # and multi-file runs)

    # ---- selecting candidates from the comparison file ----
    # Three stages (see select_sites()):
    #   1. FILTER, pass/fail: is this a cold-storage-like load with enough
    #      data? Every failed check is written out with its reason.
    #   2. RANK the sites that pass by one physical number: the average
    #      load above base load in the grid's WINTER constraint window (kW).
    #      Cumulative share of that kW decides how many are selected.
    #   3. The final pick among the top few is by hand; columns with hints
    #      (solar, on/off cycling) are there to help, not scored.
    # Winter is used for load factor, temperature correlation and the
    # window because solar panels distort summer (midday imports near zero).
    "selection": {
        "filters": {
            # column: {min/max, plus how much to move it in the robustness check}
            "coverage_fraction":               {"min": 0.80, "shift": 0.05,
                                                "why": "enough data to judge the site"},
            "load_factor_winter":              {"min": 0.50, "shift": 0.05,
                                                "why": "cold storage runs 24/7, so its load is flat"},
            "weekend_to_weekday_ratio":        {"min": 0.85, "max": 1.15, "shift": 0.05,
                                                "why": "refrigeration does not stop at weekends"},
            "temp_corr_winter_best_lag_value": {"min": 0.20, "shift": 0.05,
                                                "why": "cooling load follows outdoor temperature"},
        },
        "missing_value_fails": True,       # a check that cannot be computed counts as failed
        "rank_by": "window_above_base_winter_kw",
        "select_cumulative_share": 0.80,   # select the top sites until they hold 80 % of the
                                           # rank_by kW of all passing sites
        "pv_hint_midday_dip": 0.15,        # stage-3 hint: summer midday dip >= 15 % more than winter's
        "cycling_hint_ramp_over_avg": 0.15,  # stage-3 hint: p95 |15-min ramp| >= 15 % of average load
    },

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

    # On-site solar (common on warehouse roofs) depresses NET metered
    # load specifically at midday on hot/sunny days -- exactly when
    # ambient temperature is highest -- which can weaken or even flip
    # the apparent temperature correlation above in summer, independent
    # of whether the underlying load is actually cooling-driven. Splits
    # the same correlation by season so winter (little/no PV output) can
    # be compared against summer directly, rather than guessing whether
    # a weak summer number means "not temperature-driven" or "has solar".
    "temp_corr_season_months": {
        "winter": [11, 12, 1, 2],   # low-sun months -- cleanest read on
                                     # a cooling relationship
        "summer": [5, 6, 7, 8],     # high-sun months -- where PV
                                     # self-generation can mask it
    },
    "temp_corr_season_min_samples": 50,  # below this, a season's split
                                     # correlation isn't reported (too
                                     # little data to trust)

    # ---- grid constraint-window metrics ----
    # Peak-during-constraint-window needs to know WHEN the grid is
    # actually constrained. Rather than guessing a fixed hour range,
    # put Öresundskraft's own grid consumption data here (any period is
    # fine -- it just needs to cover a representative set of hours) and
    # the constrained hours are derived from it directly: the hours of
    # the day where the grid's own average load sits in the top
    # `constraint_percentile` of its typical daily profile.
    #
    # Set csv_path to None/"" to skip -- every constraint-window-dependent
    # metric (this one now, willingness-proxy beta later) is then
    # explicitly SKIPPED with a clear one-line reason printed, never
    # silently estimated or filled with a guessed default. If a path IS
    # set but the file doesn't exist, the script stops with an error
    # rather than silently skipping (you asked for it, so it's a mistake).
    #
    # Öresundskraft's grid-load export: HOURLY net energy (kWh per hour)
    # in Excel column V ("Netto_kWh"), header in row 1, values from V2
    # down. Opens in Excel but may actually be a .csv (Excel shows the
    # "spara i ett Excel-format" banner for CSVs) -- both .xlsx and .csv
    # (comma OR semicolon delimited, decimal comma OR point) are handled.
    # Run `inspect_grid_consumption_file(CONFIG)` once on the real file to
    # see every column by letter and confirm which timestamp column was
    # picked up.
    "grid_consumption": {
        "csv_path": "natbelastning.xlsx",  # <- your grid-load file (.xlsx
                                     # or .csv). None/"" -> skip entirely.
        "data_format": "column_letter",  # "column_letter": pick columns by
                                     # Excel letter (settings right below)
                                     # -- the Öresundskraft net-load file.
                                     # "wide": one named timestamp_col +
                                     # one named value_col. "long": the
                                     # Mängdkod-style channel filter, same
                                     # as the customer meter loader.

        # ---- only used when data_format == "column_letter" ----
        "value_col_letter": "V",     # Netto_kWh
        "first_data_row": 2,         # Excel row number of the first value
                                     # (row 1 is the header)
        "timestamp_col_letter": None,  # None = auto-detect the column that
                                     # parses as date/time (leftmost wins,
                                     # so a "from"/"to" pair resolves to
                                     # "from"). Or force it: "A". Or, if
                                     # date and time are split across two
                                     # columns: ["A", "B"].
        "fallback_start_timestamp": None,  # LAST resort, only used if no
                                     # timestamp column exists at all: e.g.
                                     # "2025-01-01 00:00" -> rows are
                                     # assumed to be consecutive intervals
                                     # from there. Ignores DST shifts, so a
                                     # real timestamp column is much better.

        # ---- shared by every data_format ----
        "value_kind": "energy_per_interval_kwh",  # Netto_kWh is ENERGY per
                                     # interval; converted to average kW
                                     # via the interval length detected
                                     # from the timestamps (1h here, so the
                                     # numbers are unchanged: 12 045 kWh in
                                     # one hour = 12 045 kW average). Use
                                     # "power_kw" if a file is already kW.
        "timestamp_label": "start",  # "start": 00:00 labels 00:00-01:00.
                                     # "end": 01:00 labels 00:00-01:00 (some
                                     # DSO exports do this -- check the
                                     # first row: if the file starts at
                                     # 01:00 rather than 00:00 on the first
                                     # day, it's probably "end").
        "native_interval": "auto",   # "auto" = detect from timestamps, or
                                     # force e.g. "1h". Used for kWh->kW and
                                     # for spreading each hourly value
                                     # onto the 15-min grid (see
                                     # `_grid_to_target_freq`).

        # ---- only used when data_format == "wide" / "long" ----
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


def seasonal_temperature_correlation(power, weather_series, cfg):
    """
    Run `temperature_correlation` separately on winter and summer months
    (cfg["temp_corr_season_months"]) to separate a genuine cooling
    relationship from an on-site-solar artifact.

    Why this matters: rooftop PV (common on warehouses) reduces the NET
    metered load specifically around midday on hot, sunny days -- exactly
    when ambient temperature peaks -- so a customer whose load really is
    cooling-driven can still show a weak or even negative whole-period or
    summer-only correlation purely because self-generation is offsetting
    the load the meter sees, not because the underlying process isn't
    temperature-driven. Winter has little/no solar output, so a winter
    correlation reflects the underlying relationship far more cleanly.

    If winter comes back meaningfully stronger than summer, that's a
    positive sign for a cooling-driven-flexibility candidate whose true
    temperature relationship is being masked by solar in the full-period
    number -- worth flagging explicitly for candidate selection rather
    than being read as "not temperature-driven" from summer/whole-period
    data alone.
    """
    seasons = cfg["temp_corr_season_months"]
    min_n = cfg["temp_corr_season_min_samples"]
    results = {}

    print("\n[seasonal temp correlation] winter vs. summer split "
          "(on-site solar can mask a real relationship in summer)")
    for season_name, months in seasons.items():
        mask = power.index.month.isin(months)
        power_season = power[mask]
        if power_season.dropna().empty:
            print(f"  {season_name} (months {months}): no data in this period -- skipped")
            results[season_name] = None
            continue
        r = temperature_correlation(power_season, weather_series, cfg)
        if r["temp_corr_n_samples"] < min_n:
            print(f"  {season_name}: only {r['temp_corr_n_samples']} overlapping "
                  f"samples (< {min_n}) -- too little data to trust, not reported")
            results[season_name] = None
            continue
        results[season_name] = r

    winter, summer = results.get("winter"), results.get("summer")
    if winter and summer:
        w = winter["temp_corr_best_lag_value"]
        s = summer["temp_corr_best_lag_value"]
        print(f"\n[seasonal temp correlation] summary: winter r={w:+.3f}  "
              f"summer r={s:+.3f}")
        if w - s > 0.2:
            print("  -> winter correlation notably stronger than summer: "
                  "consistent with on-site solar masking a real cooling "
                  "relationship in the summer/whole-period numbers. Treat "
                  "the winter-only correlation as the more representative "
                  "one for this customer.")
        elif s - w > 0.2:
            print("  -> summer correlation notably stronger than winter: "
                  "not explained by solar masking (that would weaken "
                  "summer, not strengthen it) -- check for another "
                  "seasonal driver instead.")
        else:
            print("  -> winter and summer are broadly consistent: no "
                  "strong sign of solar masking the relationship.")

    return results


# ======================================================================
# 8c. Grid constraint-window metrics
# ======================================================================
def _excel_col_to_index(letter):
    """Excel column letter -> 0-based position: "A" -> 0, "V" -> 21,
    "AA" -> 26."""
    letter = str(letter).strip().upper()
    if not letter.isalpha():
        raise ValueError(f"Not an Excel column letter: {letter!r}")
    idx = 0
    for ch in letter:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _index_to_excel_col(idx):
    """0-based position -> Excel column letter (inverse of the above)."""
    idx += 1
    out = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


def _read_raw_sheet(path):
    """
    Read a grid-load file as a raw grid of cells with NO header handling,
    so columns can be addressed by Excel letter (position 0 = "A") and
    rows by Excel row number (position 0 = row 1).

    .xlsx/.xlsm/.xls -> read_excel, first sheet.
    anything else    -> treated as delimited text. The delimiter is
    picked from the HEADER line (the one line guaranteed to contain no
    decimal commas): whichever of ';', ',' or tab occurs most often
    there. Swedish Excel saves CSV with ';' and decimal commas, other
    tools use ',' and decimal points -- both work. Every cell is kept as
    text here; numbers are parsed later by `_to_float`, which handles
    either decimal style.
    """
    lower = str(path).lower()
    if lower.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(path, header=None)

    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with open(path, encoding=encoding) as f:
                first_line = f.readline()
            sep = max([";", ",", "\t"], key=first_line.count)
            return pd.read_csv(path, header=None, sep=sep, dtype=str,
                               encoding=encoding, skip_blank_lines=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} as UTF-8 or cp1252 text.")


def _to_float(values):
    """
    Parse a column to float, accepting Excel numbers as-is and text in
    either decimal style: "12045,598" (Swedish) or "12045.598". A value
    with BOTH separators ("12.045,598") is read as '.' = thousands,
    ',' = decimal. Spaces/non-breaking spaces (thousands separators in
    Swedish formatting) are dropped. Unparseable cells -> NaN.
    """
    s = pd.Series(values)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    s = s.astype(str).str.strip().str.replace(" ", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    both = s.str.contains(",", regex=False) & s.str.contains(".", regex=False)
    s = s.where(~both, s.str.replace(".", "", regex=False))
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _parse_timestamps(values):
    """
    Parse one column of timestamps. Excel datetime cells come through
    as-is; text is parsed flexibly. A text interval like
    "2025-01-01 00:00 - 01:00" is cut down to its first timestamp
    (combine with timestamp_label to say whether that's the start or end).
    """
    s = pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    s = s.astype(str).str.strip()
    s = s.str.split(r"\s+[-–]\s*(?=\d{1,2}:\d{2})", n=1, regex=True).str[0]
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed")
    except (TypeError, ValueError):  # pandas < 2.0 has no format="mixed"
        return pd.to_datetime(s, errors="coerce")


def _autodetect_timestamp_col(data, value_idx, sample=200):
    """
    Find the column that is a timestamp: the LEFTMOST column (other than
    the value column) where >= 90% of a sample of data rows parse as a
    date/time and which isn't really a numeric column (numbers can
    parse as epoch dates, so those are excluded up front). Leftmost wins
    so that a "från"/"till" (from/to) pair resolves to "från" -- which
    is what timestamp_label="start" expects.
    Returns the 0-based column position, or None.
    """
    head = data.iloc[:sample]
    for idx in range(head.shape[1]):
        if idx == value_idx:
            continue
        col = head.iloc[:, idx].dropna()
        if col.empty:
            continue
        if pd.api.types.is_datetime64_any_dtype(col):
            return idx
        if pd.api.types.is_numeric_dtype(col):
            continue
        if _to_float(col).notna().mean() > 0.5:
            continue  # numbers stored as text, not dates
        if col.map(lambda v: hasattr(v, "year")).mean() >= 0.9:
            return idx  # Excel datetime cells mixed in an object column
        if _parse_timestamps(col).notna().mean() >= 0.9:
            return idx
    return None


def inspect_grid_consumption_file(cfg, n_rows=4):
    """
    Diagnostic helper -- run once against the real grid-load file. Prints
    every column by Excel letter with its row-1 header and the first few
    values, and marks the value column (value_col_letter) and the
    timestamp column the loader would use (auto-detected or forced), so
    you can confirm both are right before trusting the constraint window.

        from lpc_5 import CONFIG, inspect_grid_consumption_file
        inspect_grid_consumption_file(CONFIG)
    """
    gc = cfg["grid_consumption"]
    raw = _read_raw_sheet(gc["csv_path"])
    first = gc.get("first_data_row", 2) - 1
    data = raw.iloc[first:]
    value_idx = _excel_col_to_index(gc.get("value_col_letter", "V"))

    forced = gc.get("timestamp_col_letter")
    if forced:
        letters = forced if isinstance(forced, (list, tuple)) else [forced]
        ts_idx = {_excel_col_to_index(l) for l in letters}
    else:
        auto = _autodetect_timestamp_col(data, value_idx)
        ts_idx = {auto} if auto is not None else set()

    print(f"\n[inspect grid] '{gc['csv_path']}': {raw.shape[1]} column(s), "
          f"{len(data)} data row(s) from Excel row {first + 1}")
    for idx in range(raw.shape[1]):
        header = raw.iat[0, idx] if first > 0 else ""
        sample = data.iloc[:n_rows, idx].tolist()
        tag = ""
        if idx == value_idx:
            tag = "   <-- VALUE column"
        elif idx in ts_idx:
            tag = "   <-- TIMESTAMP column" + ("" if forced else " (auto-detected)")
        print(f"  {_index_to_excel_col(idx):>3}  {str(header)[:22]:<22} {sample}{tag}")
    if not ts_idx:
        print("[inspect grid] no timestamp column found -- set "
              "timestamp_col_letter (or fallback_start_timestamp).")


def _load_grid_column_letter(cfg):
    """
    `data_format: "column_letter"` loader -- the Öresundskraft net-load
    export: values in column V (Netto_kWh) from row 2 down, timestamps
    from an auto-detected (or forced) column, or synthesized from
    fallback_start_timestamp as a last resort. Returns the raw series
    (values as in the file, timestamps as labelled in the file).
    """
    gc = cfg["grid_consumption"]
    raw = _read_raw_sheet(gc["csv_path"])
    first = gc.get("first_data_row", 2) - 1
    letter = gc.get("value_col_letter", "V")
    value_idx = _excel_col_to_index(letter)
    if value_idx >= raw.shape[1]:
        raise KeyError(
            f"value_col_letter '{letter}' is beyond the last column of "
            f"{gc['csv_path']} (it only has {raw.shape[1]} columns, up to "
            f"'{_index_to_excel_col(raw.shape[1] - 1)}'). If this is a CSV, "
            f"the delimiter may have been mis-detected -- run "
            f"inspect_grid_consumption_file(CONFIG)."
        )

    data = raw.iloc[first:]
    header = raw.iat[0, value_idx] if first > 0 else "(no header row)"
    values = _to_float(data.iloc[:, value_idx]).to_numpy()

    forced = gc.get("timestamp_col_letter")
    if forced:
        letters = forced if isinstance(forced, (list, tuple)) else [forced]
        idxs = [_excel_col_to_index(l) for l in letters]
        if len(idxs) == 1:
            ts = _parse_timestamps(data.iloc[:, idxs[0]])
        else:
            # date and time split across columns: normalize each part to
            # text ("2025-01-01" + "01:00:00") and parse the combination
            parts = []
            for i in idxs:
                col = data.iloc[:, i]
                if pd.api.types.is_datetime64_any_dtype(col):
                    col = col.dt.strftime("%Y-%m-%d")
                parts.append(col.astype(str).str.strip())
            joined = parts[0]
            for p in parts[1:]:
                joined = joined + " " + p
            ts = _parse_timestamps(joined)
        ts_source = f"column(s) {letters} (forced)"
    else:
        ts_idx = _autodetect_timestamp_col(data, value_idx)
        if ts_idx is not None:
            ts = _parse_timestamps(data.iloc[:, ts_idx])
            ts_header = raw.iat[0, ts_idx] if first > 0 else ""
            ts_source = (f"column {_index_to_excel_col(ts_idx)} "
                         f"('{ts_header}', auto-detected)")
        elif gc.get("fallback_start_timestamp"):
            n_valid = int(pd.notna(values).sum())
            values = values[pd.notna(values)]
            step = (gc.get("native_interval") if gc.get("native_interval", "auto")
                    != "auto" else "1h")
            ts = pd.Series(pd.date_range(gc["fallback_start_timestamp"],
                                         periods=n_valid, freq=step))
            ts_source = (f"SYNTHESIZED from fallback_start_timestamp="
                         f"{gc['fallback_start_timestamp']!r} every {step} "
                         f"-- assumes no missing rows and ignores DST")
            print("[grid load] WARNING: no timestamp column found; timestamps "
                  "were synthesized. If the export follows Swedish local time, "
                  "everything after the spring DST switch is shifted by an "
                  "hour. Prefer pointing timestamp_col_letter at a real column.")
        else:
            raise ValueError(
                f"No timestamp column could be auto-detected in "
                f"{gc['csv_path']}. Run inspect_grid_consumption_file(CONFIG) "
                f"to see every column by letter, then set "
                f"CONFIG['grid_consumption']['timestamp_col_letter'] (e.g. "
                f"'A', or ['A', 'B'] for separate date/time columns), or as a "
                f"last resort 'fallback_start_timestamp'."
            )

    s = pd.Series(values, index=pd.DatetimeIndex(ts), name="grid_load")
    s = s[s.index.notnull() & s.notna()]

    print(f"[grid load] read column {letter} ('{header}') from row "
          f"{first + 1} down: {len(s)} values; timestamps from {ts_source}")

    if len(s) and s.index.duplicated().mean() > 0.5:
        print("[grid load] WARNING: most timestamps repeat -- the timestamp "
              "column looks like a DATE without the hour. Set "
              "timestamp_col_letter to [date_column, hour_column].")
    return s


def _grid_to_target_freq(series_kw, interval_hours, cfg):
    """
    Put the grid series on the same 15-min grid as everything else.

    Coarser than 15 min (the hourly Netto_kWh case): each hour's AVERAGE
    kW is held constant over its four quarter-hours. That's the honest
    reading of an hourly energy value -- it says how much energy flowed
    in the hour, nothing about how it was distributed inside it -- and it
    conserves energy (4 x 0.25 h x avg kW = the hourly kWh). No
    interpolation, so no invented intra-hour shape. A genuinely missing
    hour stays NaN; the hold never reaches past its own hour.

    Finer than (or equal to) 15 min: averaged down, as for the customer
    meter series.

    timestamp_label="end" is handled first by shifting every timestamp
    back one interval, so 01:00 ("the hour ending 01:00") becomes the
    00:00-01:00 block it actually describes.
    """
    gc = cfg["grid_consumption"]
    target = pd.Timedelta(cfg["resample_freq"])
    native = pd.Timedelta(hours=interval_hours)
    s = series_kw.copy()

    if gc.get("timestamp_label", "start") == "end":
        s.index = s.index - native

    if native <= target * 1.01:
        return s.resample(cfg["resample_freq"]).mean().rename("grid_load")

    s.index = s.index.floor(target)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    n_sub = int(round(native / target))
    full_index = pd.date_range(s.index.min(), s.index.max() + native - target,
                               freq=target)
    out = s.reindex(full_index).ffill(limit=n_sub - 1)
    return out.rename("grid_load")


def load_grid_consumption(cfg):
    """
    Load Öresundskraft's own grid consumption series, used only to work
    out WHEN the grid is constrained -- not compared in magnitude against
    any customer's load, so its units don't need to match theirs.

    Three input shapes (CONFIG['grid_consumption']['data_format']):
      - "column_letter": columns picked by Excel letter -- the net-load
        export with hourly Netto_kWh in column V from row 2 down.
      - "wide": one named timestamp column + one named value column.
      - "long": Mängdkod-style channel filter, like the customer loader.

    Whatever the input resolution, the result is returned on the same
    15-min grid (cfg["resample_freq"]) as the customer series, in average
    kW, so it lines up 1:1 with everything else in the script.
    """
    gc = cfg["grid_consumption"]
    fmt = gc.get("data_format", "column_letter")

    if fmt == "column_letter":
        s = _load_grid_column_letter(cfg)
    else:
        df = _read_table(gc["csv_path"])
        ts_col = gc["timestamp_col"]
        if gc.get("timestamp_unit"):
            df[ts_col] = pd.to_datetime(df[ts_col], unit=gc["timestamp_unit"], errors="coerce")
        else:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df[df[ts_col].notnull()]

        if fmt == "long":
            channel_col = gc["channel_col"]
            if channel_col not in df.columns:
                raise KeyError(
                    f"channel_col '{channel_col}' not found in "
                    f"{gc['csv_path']}. Columns present: {list(df.columns)}"
                )
            df = _select_channel_rows(df, channel_col, gc["channel_values"])

        s = pd.Series(_to_float(df[gc["value_col"]]).to_numpy(),
                      index=pd.DatetimeIndex(df[ts_col]), name="grid_load")
        s = s[s.notna()]

    s = s.sort_index()
    n_dupes = int(s.index.duplicated().sum())
    s = s[~s.index.duplicated(keep="first")]
    if s.empty:
        raise ValueError(f"No usable grid-load values read from {gc['csv_path']}.")

    if gc.get("native_interval", "auto") == "auto":
        interval_hours = _infer_interval_hours(s.index)
    else:
        interval_hours = pd.Timedelta(gc["native_interval"]).total_seconds() / 3600.0

    if gc.get("value_kind", "energy_per_interval_kwh") == "energy_per_interval_kwh":
        s = s / interval_hours  # kWh per interval -> average kW

    out = _grid_to_target_freq(s, interval_hours, cfg)

    print(f"[grid load] native interval {interval_hours * 60:.0f} min -> "
          f"{cfg['resample_freq']} grid ({'held constant within each interval' if interval_hours * 60 > pd.Timedelta(cfg['resample_freq']).total_seconds() / 60 * 1.01 else 'averaged down'}); "
          f"span {out.first_valid_index()} .. {out.last_valid_index()}; "
          f"avg {out.mean():,.0f} kW, min {out.min():,.0f}, max {out.max():,.0f}"
          + (f"; {n_dupes} duplicate timestamp(s) dropped (e.g. DST fall-back hour)"
             if n_dupes else ""))
    return out


def report_grid_customer_overlap(grid_series, power):
    """
    Informational only. The constraint window is an hour-of-day profile,
    so it does NOT need the grid and customer data to cover the same
    dates -- but if they do overlap, that's worth knowing (and if they
    were meant to and don't, it usually means a date-format or timezone
    problem).
    """
    g = grid_series.dropna()
    p = power.dropna()
    common = g.index.intersection(p.index)
    print(f"[grid load] customer data: {p.index.min()} .. {p.index.max()}; "
          f"grid data: {g.index.min()} .. {g.index.max()}; "
          f"{len(common)} shared 15-min timestamps "
          f"({len(common) / max(len(p), 1):.0%} of the customer's)")


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
            "constraint_window_avg_outside_kw": np.nan,
            "constraint_window_avg_frac_of_overall_avg": np.nan,
            "constraint_window_peak_kw": np.nan,
            "constraint_window_peak_frac_of_overall_peak": np.nan,
            "constraint_window_hours": constraint_hours,
        }

    avg_in_window = in_window.mean()
    peak_in_window = in_window.max()
    overall_peak = overall_metrics["peak_kw"]
    peak_frac = peak_in_window / overall_peak if overall_peak > 0 else np.nan
    overall_avg = overall_metrics["avg_kw"]
    avg_frac = avg_in_window / overall_avg if overall_avg > 0 else np.nan
    out_window = valid[~valid.index.hour.isin(constraint_hours)]
    avg_outside = out_window.mean() if len(out_window) else np.nan

    print(f"\n[constraint window] this customer's load during those hours")
    print(f"  avg load in window:   {avg_in_window:.1f} kW")
    print(f"  peak load in window:  {peak_in_window:.1f} kW")
    print(f"  as fraction of overall peak: {peak_frac:.1%}")
    print(f"  avg in window / overall avg: {avg_frac:.2f}  "
          f"(>1 = this site uses more than its average when the grid is constrained)")

    return {
        "constraint_window_avg_kw": avg_in_window,
        "constraint_window_avg_outside_kw": avg_outside,
        "constraint_window_avg_frac_of_overall_avg": avg_frac,
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
    if cfg.get("show_plot", True):
        plt.show()
    plt.close(fig)


# ======================================================================
# 11. Site comparison file (one row per site, upserted)
# ======================================================================
SITE_KEY = "site_id"


def site_id_for(cfg):
    """The name this site gets in the comparison file: CONFIG['site_id'],
    or else the meter file name without folder and extension."""
    if cfg.get("site_id"):
        return str(cfg["site_id"]).strip()
    return os.path.splitext(os.path.basename(str(cfg["csv_path"])))[0].strip()


def _num(x):
    """Plain float (or None) so the CSV holds numbers, not numpy reprs."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


def winter_and_hint_metrics(power, grid_load, cfg):
    """
    Numbers for the candidate selection (see CONFIG["selection"]):
      load_factor_winter          winter average / winter peak
      base_load_winter_kw         winter 5th percentile
      constraint_window_hours_winter   the grid's constrained hours in winter
                                  (derived from the grid's winter load only)
      window_avg_winter_kw        the site's winter average in those hours
      window_above_base_winter_kw window_avg_winter_kw - base_load_winter_kw:
                                  load that could in principle be moved out
                                  of the window (net metered exchange)
      window_avg_frac_of_overall_avg_winter   window avg / winter avg
    and stage-3 hints (not scored):
      pv_midday_dip_summer / _winter  1 - (mean 11-14) / (mean 07-10 and 15-18), weekdays.
                                  Solar makes the summer dip much deeper than winter's.
      share_intervals_at_or_below_zero  quarter-hours with zero or negative import
      ramp_p95_over_avg           p95 |15-min ramp| / average load: big relative
                                  swings hint at on/off compressor cycling
    """
    out = {}
    valid = power.dropna()
    winter_m = (cfg.get("temp_corr_season_months") or {}).get("winter", [11, 12, 1, 2])
    summer_m = (cfg.get("temp_corr_season_months") or {}).get("summer", [5, 6, 7, 8])
    w = valid[valid.index.month.isin(winter_m)]
    out["winter_quarter_hours"] = int(len(w))
    if len(w):
        out["load_factor_winter"] = _num(w.mean() / w.max()) if w.max() > 0 else None
        out["base_load_winter_kw"] = _num(w.quantile(0.05))
        out["avg_winter_kw"] = _num(w.mean())
    if grid_load is not None and len(w):
        g = grid_load[grid_load.index.month.isin(winter_m)].dropna()
        if len(g):
            hourly_avg = g.groupby(g.index.hour).mean()
            thr = hourly_avg.quantile(cfg["grid_consumption"]["constraint_percentile"])
            hours = sorted(hourly_avg[hourly_avg >= thr].index.tolist())
            inw = w[w.index.hour.isin(hours)]
            out["constraint_window_hours_winter"] = ",".join(str(h) for h in hours)
            if len(inw):
                out["window_avg_winter_kw"] = _num(inw.mean())
                out["window_above_base_winter_kw"] = _num(inw.mean() - w.quantile(0.05))
                out["window_avg_frac_of_overall_avg_winter"] = _num(inw.mean() / w.mean()) if w.mean() > 0 else None

    def dip(x):
        x = x[x.index.dayofweek < 5]
        if x.empty:
            return None
        h = x.index.hour
        mid = x[(h >= 11) & (h < 14)].mean()
        shoulder = x[((h >= 7) & (h < 10)) | ((h >= 15) & (h < 18))].mean()
        return _num(1 - mid / shoulder) if shoulder and shoulder > 0 else None
    out["pv_midday_dip_summer"] = dip(valid[valid.index.month.isin(summer_m)])
    out["pv_midday_dip_winter"] = dip(w)
    out["share_intervals_at_or_below_zero"] = _num((valid <= 0).mean()) if len(valid) else None
    ramps = valid.diff().dropna()
    out["ramp_p95_over_avg"] = _num(ramps.abs().quantile(0.95) / valid.mean()) \
        if len(ramps) and valid.mean() > 0 else None
    return out


def build_site_row(cfg, power, metrics, ldc_table, profiles, best_k, labels,
                   ramps, ramp_warn, peak_hour, monthly, temp_corr,
                   temp_corr_seasonal, constraint_window, extra=None):
    """
    One flat row of headline numbers for this site. Everything here is
    already computed by main(); this only collects it. Columns that a run
    could not compute (no weather file, no grid file, too few days to
    cluster) are left empty, never guessed.
    """
    valid = power.dropna()
    row = {
        SITE_KEY: site_id_for(cfg),
        "meter_file": os.path.basename(str(cfg["csv_path"])),
        "meter_path": os.path.abspath(str(cfg["csv_path"])),
        "analysed_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "data_start": str(valid.index.min()) if len(valid) else None,
        "data_end": str(valid.index.max()) if len(valid) else None,
    }
    for k, v in metrics.items():
        row[k] = _num(v)

    for p in sorted(ldc_table):
        row[f"ldc_exceeded_{p}pct_kw"] = _num(ldc_table[p])

    for name, (mean, _, _) in profiles.items():
        row[f"{name}_avg_kw"] = _num(mean.mean()) if len(mean) else None
        row[f"{name}_profile_peak_kw"] = _num(mean.max()) if len(mean) else None
    wd, we = row.get("weekday_avg_kw"), row.get("weekend_avg_kw")
    row["weekend_to_weekday_ratio"] = _num(we / wd) if wd and we is not None else None

    n_days = len(labels)
    row["n_complete_days"] = n_days
    row["day_types_k"] = int(best_k) if n_days else None
    if n_days:
        shares = pd.Series(labels).value_counts(normalize=True)
        row["largest_day_type_share"] = _num(shares.iloc[0])

    if ramps is not None and len(ramps):
        row["ramp_p95_abs_kw"] = _num(ramps.abs().quantile(0.95))
        row["ramp_max_up_kw"] = _num(ramps.max())
        row["ramp_max_down_kw"] = _num(ramps.min())
        row["ramp_warn_threshold_kw"] = _num(ramp_warn)
        row["ramp_share_over_threshold"] = _num((ramps.abs() > ramp_warn).mean())

    if peak_hour is not None and len(peak_hour):
        hours = np.floor(peak_hour).astype(int)
        counts = pd.Series(hours).value_counts()
        row["peak_hour_most_common"] = int(counts.index[0])
        row["peak_hour_most_common_share"] = _num(counts.iloc[0] / len(hours))
        row["peak_hour_median"] = _num(np.median(peak_hour))

    if monthly is not None and len(monthly):
        row["peak_month"] = str(monthly["max_kw"].idxmax())
        row["highest_energy_month"] = str(monthly["energy_kwh"].idxmax())

    if temp_corr:
        for k, v in temp_corr.items():
            row[k] = _num(v)
    for season, r in (temp_corr_seasonal or {}).items():
        row[f"temp_corr_{season}_best_lag_value"] = _num(r["temp_corr_best_lag_value"]) if r else None
        row[f"temp_corr_{season}_best_lag_hours"] = _num(r["temp_corr_best_lag_hours"]) if r else None

    if constraint_window:
        for k, v in constraint_window.items():
            if k == "constraint_window_hours":
                row[k] = ",".join(str(h) for h in v) if v else None
            else:
                row[k] = _num(v)
    for k, v in (extra or {}).items():
        row[k] = v
    return row


def upsert_site_row(row, cfg):
    """
    Write `row` into the comparison CSV: added if the site is new,
    replacing the old row if the site is already there (matched on
    site_id, ignoring case and surrounding spaces). Columns are the union
    of old and new, so adding a metric in a later version of the script
    just leaves it empty for sites run before. Rows are sorted by
    site_id.

    The file is written to a temporary name first and then moved into
    place, so an interrupted run never leaves a half-written file. If the
    file is open in Excel (Windows locks it), the results are written
    next to it with a time stamp in the name instead, and a warning says
    so -- nothing is lost.
    """
    path = cfg.get("results_csv_path")
    if not path:
        return None
    sep = cfg.get("results_csv_sep", ";")
    dec = cfg.get("results_csv_decimal", ",")
    new = pd.DataFrame([row])

    if os.path.exists(path):
        old = pd.read_csv(path, sep=sep, decimal=dec, encoding="utf-8-sig")
        if SITE_KEY not in old.columns:
            raise ValueError(f"{path} has no '{SITE_KEY}' column -- not a "
                             f"comparison file written by this script?")
        key = row[SITE_KEY].strip().lower()
        same = old[SITE_KEY].astype(str).str.strip().str.lower() == key
        action = "replaced" if same.any() else "added"
        old = old[~same]
        cols = list(old.columns) + [c for c in new.columns if c not in old.columns]
        out = pd.concat([old, new], ignore_index=True)[cols] if len(old) else new
    else:
        action, out = "added (new file)", new

    out, written = _write_comparison(out, cfg)
    print(f"\n[site comparison] '{row[SITE_KEY]}' {action} in {written} "
          f"({len(out)} site(s) in the file)")
    return out


SELECTION_COLS = ["rank", "selected", "filter_pass", "filter_fail_reasons",
                  "window_above_base_winter_kw", "cumulative_share", "window_avg_frac_of_overall_avg_winter",
                  "hint_solar", "hint_cycling"]
OLD_RANK_COLS = ["rank_score", "ranking_factors_used"]     # from the first lpc_8 ranking


def _fmt(v):
    return f"{v:.2f}" if isinstance(v, (int, float, np.floating)) and abs(v) < 100 else f"{v:.0f}"


def _filter(df, filters, missing_fails):
    """Stage 1. Returns (pass Series, reasons Series)."""
    ok = pd.Series(True, index=df.index)
    reasons = pd.Series("", index=df.index)
    for col, spec in filters.items():
        v = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
        bad = pd.Series(False, index=df.index)
        msg = pd.Series("", index=df.index)
        if "min" in spec:
            lo = v < spec["min"]
            bad |= lo
            msg[lo] = [f"{col} {_fmt(x)} < {spec['min']}" for x in v[lo]]
        if "max" in spec:
            hi = v > spec["max"]
            bad |= hi
            msg[hi] = [f"{col} {_fmt(x)} > {spec['max']}" for x in v[hi]]
        if missing_fails:
            na = v.isna()
            bad |= na
            msg[na] = f"{col} missing"
        ok &= ~bad
        reasons[bad] = (reasons[bad] + "; " + msg[bad]).str.lstrip("; ")
    return ok, reasons


def _select(df, passed, sel):
    """Stage 2. Rank passing sites by rank_by (desc) and select the top ones
    until they hold select_cumulative_share of the passing sites' total."""
    key = sel["rank_by"]
    v = pd.to_numeric(df[key], errors="coerce") if key in df.columns else pd.Series(np.nan, index=df.index)
    elig = passed & v.notna()
    rank = pd.Series(pd.NA, index=df.index, dtype="Int64")
    cum = pd.Series(np.nan, index=df.index)
    selected = pd.Series(False, index=df.index)
    if elig.any():
        order = v[elig].sort_values(ascending=False)
        rank[order.index] = np.arange(1, len(order) + 1)
        pos = order.clip(lower=0)
        total = pos.sum()
        c = pos.cumsum() / total if total > 0 else pd.Series(np.nan, index=order.index)
        cum[order.index] = c.round(3)
        before = c.shift(1, fill_value=0.0)
        selected[order.index] = before < sel["select_cumulative_share"]
    return rank, cum, selected


def select_sites(df, cfg):
    """
    Candidate selection on the comparison table (CONFIG["selection"]).
    Stage 1 filter -> filter_pass, filter_fail_reasons.
    Stage 2 rank the passing sites by window_above_base_winter_kw ->
            rank (1 = most kW), cumulative_share, selected.
    Stage 3 hints for choosing by hand -> hint_solar, hint_cycling.
    """
    sel = cfg.get("selection") or {}
    df = df.drop(columns=[c for c in df.columns
                          if c.startswith("score_") or c in SELECTION_COLS + OLD_RANK_COLS
                          and c not in ("window_above_base_winter_kw", "window_avg_frac_of_overall_avg_winter")])
    passed, reasons = _filter(df, sel.get("filters", {}), sel.get("missing_value_fails", True))
    rank, cum, selected = _select(df, passed, sel)
    # stage-3 hints
    ds = pd.to_numeric(df.get("pv_midday_dip_summer"), errors="coerce") if "pv_midday_dip_summer" in df else None
    dw = pd.to_numeric(df.get("pv_midday_dip_winter"), errors="coerce") if "pv_midday_dip_winter" in df else None
    hint_solar = pd.Series("", index=df.index)
    if ds is not None and dw is not None:
        hint_solar[(ds - dw) >= sel.get("pv_hint_midday_dip", 0.15)] = "summer midday dip: solar?"
    if "share_intervals_at_or_below_zero" in df:
        z = pd.to_numeric(df["share_intervals_at_or_below_zero"], errors="coerce") > 0.01
        hint_solar[z] = (hint_solar[z] + "; exports at times").str.lstrip("; ")
    hint_cyc = pd.Series("", index=df.index)
    if "ramp_p95_over_avg" in df:
        r = pd.to_numeric(df["ramp_p95_over_avg"], errors="coerce")
        hint_cyc[r >= sel.get("cycling_hint_ramp_over_avg", 0.15)] = "large 15-min swings: on/off cycling?"
    front = {"rank": rank, "selected": selected, "filter_pass": passed, "filter_fail_reasons": reasons,
             "cumulative_share": cum, "hint_solar": hint_solar, "hint_cycling": hint_cyc}
    for i, (name, col) in enumerate(front.items()):
        df.insert(1 + i, name, col)
    # the ranking number and its secondary right after
    for i, name in enumerate(["window_above_base_winter_kw", "window_avg_frac_of_overall_avg_winter"]):
        if name in df.columns:
            colv = df.pop(name)
            df.insert(5 + i, name, colv)
    return df


def sensitivity(df, cfg):
    """
    Robustness of the selection: move each threshold by its `shift`,
    stricter and looser, one at a time and all together, and report
    whether the selected set changes. Returns a table, one row per case.
    """
    sel = cfg.get("selection") or {}
    filters = sel.get("filters", {})
    base_pass, _ = _filter(df, filters, sel.get("missing_value_fails", True))
    _, _, base_sel = _select(df, base_pass, sel)
    base_set = set(df.loc[base_sel, SITE_KEY].astype(str))

    def moved(which, sign):
        f2 = {}
        for col, spec in filters.items():
            sp = dict(spec)
            if which in (col, "all"):
                d = spec.get("shift", 0.0) * sign          # sign +1 = stricter
                if "min" in sp:
                    sp["min"] = sp["min"] + d
                if "max" in sp:
                    sp["max"] = sp["max"] - d
            f2[col] = sp
        return f2

    rows = []
    for which in list(filters) + ["all"]:
        for sign, label in ((+1, "stricter"), (-1, "looser")):
            f2 = moved(which, sign)
            p2, _ = _filter(df, f2, sel.get("missing_value_fails", True))
            _, _, s2 = _select(df, p2, sel)
            new_set = set(df.loc[s2, SITE_KEY].astype(str))
            rows.append({"threshold": which, "moved": f"{label} by {filters[which].get('shift', 0) if which != 'all' else 'each shift'}",
                         "n_pass": int(p2.sum()), "n_selected": len(new_set),
                         "selection_unchanged": new_set == base_set,
                         "dropped": ", ".join(sorted(base_set - new_set)),
                         "added": ", ".join(sorted(new_set - base_set))})
    return pd.DataFrame(rows), base_set


def _write_comparison(out, cfg):
    """Rank, sort (best first, unranked last, then by name) and write the
    comparison table atomically; returns (table, path written)."""
    path = cfg["results_csv_path"]
    sep = cfg.get("results_csv_sep", ";")
    dec = cfg.get("results_csv_decimal", ",")
    out = select_sites(out, cfg)
    out = out.assign(_k=out[SITE_KEY].astype(str).str.lower()) \
             .sort_values(["rank", "_k"], na_position="last").drop(columns="_k").reset_index(drop=True)
    folder = os.path.dirname(os.path.abspath(path))
    tmp = os.path.join(folder, f".{os.path.basename(path)}.tmp")
    out.to_csv(tmp, sep=sep, decimal=dec, index=False, encoding="utf-8-sig")
    try:
        os.replace(tmp, path)
        written = path
    except PermissionError:
        stem, ext = os.path.splitext(path)
        written = f"{stem}_{pd.Timestamp.now():%Y%m%d_%H%M%S}{ext}"
        os.replace(tmp, written)
        print(f"[site comparison] WARNING: {path} is locked (open in Excel?) "
              f"-- wrote the updated table to {written} instead. Close the "
              f"file and rename it, or rerun.")
    return out, written


def read_comparison(cfg):
    path = cfg.get("results_csv_path")
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path, sep=cfg.get("results_csv_sep", ";"),
                       decimal=cfg.get("results_csv_decimal", ","), encoding="utf-8-sig")


def rank_only(cfg=CONFIG):
    """Recompute the selection columns of the comparison file in place,
    print the three stages and the robustness check (also saved as
    selection_sensitivity.csv next to the comparison file)."""
    df = read_comparison(cfg)
    if df is None:
        print(f"[selection] no comparison file at {cfg.get('results_csv_path')!r}")
        return None
    out, written = _write_comparison(df, cfg)
    sel = cfg.get("selection") or {}
    print(f"\n[selection] {len(out)} site(s) in {written}")
    print("\n  stage 1 -- filters (pass/fail):")
    for col, spec in sel.get("filters", {}).items():
        rng = " and ".join(([f">= {spec['min']}"] if "min" in spec else []) + ([f"<= {spec['max']}"] if "max" in spec else []))
        print(f"    {col} {rng}   ({spec.get('why', '')})")
    failed = out[~out["filter_pass"].astype(bool)]
    print(f"  {int(out['filter_pass'].astype(bool).sum())} pass, {len(failed)} fail")
    for _, r in failed.iterrows():
        print(f"    FAIL  {r[SITE_KEY]}: {r['filter_fail_reasons']}")
    passed = out[out["filter_pass"].astype(bool)]
    print(f"\n  stage 2 -- passing sites ranked by {sel.get('rank_by')} (kW above base, winter window); "
          f"selected until {sel.get('select_cumulative_share', 0):.0%} cumulative share:")
    cols = [c for c in ["rank", "selected", SITE_KEY, sel.get("rank_by"), "cumulative_share",
                        "window_avg_frac_of_overall_avg_winter", "hint_solar", "hint_cycling"] if c in out.columns]
    if len(passed):
        print(passed[cols].to_string(index=False))
    else:
        print("    (no site passes the filters)")
    sens, base_set = sensitivity(out, cfg)
    folder = os.path.dirname(os.path.abspath(cfg["results_csv_path"]))
    sens.to_csv(os.path.join(folder, "selection_sensitivity.csv"), sep=cfg.get("results_csv_sep", ";"),
                decimal=cfg.get("results_csv_decimal", ","), index=False, encoding="utf-8-sig")
    n_same = int(sens["selection_unchanged"].sum())
    print(f"\n  robustness -- each threshold moved by its shift, stricter and looser: the selection "
          f"is unchanged in {n_same} of {len(sens)} cases")
    for _, r in sens[~sens["selection_unchanged"]].iterrows():
        print(f"    {r['threshold']} {r['moved']}: " + "; ".join(x for x in
              [f"drops {r['dropped']}" if r['dropped'] else "", f"adds {r['added']}" if r['added'] else ""] if x))
    print("\n  stage 3 -- choose among the selected sites by hand (solar, on/off cycling: see hint columns).")
    return out


def find_meter_file(row, cfg):
    """Where a comparison row's meter file is now: its stored full path,
    else its file name in meter_folder, the comparison file's folder or
    the working directory."""
    cands = []
    p = row.get("meter_path")
    if isinstance(p, str) and p:
        cands.append(p)
    name = row.get("meter_file")
    if isinstance(name, str) and name:
        folders = [cfg.get("meter_folder"),
                   os.path.dirname(os.path.abspath(cfg["results_csv_path"])), os.getcwd()]
        cands += [os.path.join(f, name) for f in folders if f]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def run_sites(paths, cfg=CONFIG, site_ids=None):
    """Analyse several meter files one after another without opening plot
    windows (each figure is saved as load_shape_<site>.png next to the
    comparison file). One failing file is reported and skipped; the
    others still run. Returns {site: "ok" | error text}."""
    status = {}
    folder = os.path.dirname(os.path.abspath(cfg["results_csv_path"] or "."))
    for i, path in enumerate(paths):
        c = dict(cfg)
        c["csv_path"] = path
        c["site_id"] = site_ids[i] if site_ids else cfg.get("site_id") if len(paths) == 1 else None
        sid = site_id_for(c)
        safe = "".join(ch if ch.isalnum() or ch in "-_ ." else "_" for ch in sid)
        c["output_png"] = os.path.join(folder, f"load_shape_{safe}.png")
        c["show_plot"] = False
        print(f"\n{'=' * 70}\n[{i + 1}/{len(paths)}] {sid}\n{'=' * 70}")
        try:
            main(c)
            status[sid] = "ok"
        except Exception as e:      # keep going: one bad file must not stop the batch
            status[sid] = f"{type(e).__name__}: {e}"
            print(f"[run] {sid} FAILED: {status[sid]}")
    print("\n[run] summary")
    for sid, st in status.items():
        print(f"  {'ok    ' if st == 'ok' else 'FAILED'}  {sid}" + ("" if st == "ok" else f"  -- {st}"))
    return status


def rerun_all_sites(cfg=CONFIG):
    """
    Re-analyse every site already in the comparison file with the current
    script and settings (new columns get filled, old numbers refreshed),
    then re-rank. A site whose meter file cannot be found is listed and
    left as it was -- point CONFIG["meter_folder"] at the folder holding
    the meter files.
    """
    df = read_comparison(cfg)
    if df is None:
        print(f"[rerun] no comparison file at {cfg.get('results_csv_path')!r} -- nothing to rerun")
        return {}
    paths, ids, missing = [], [], []
    for _, row in df.iterrows():
        p = find_meter_file(row, cfg)
        if p is None:
            missing.append(str(row[SITE_KEY]))
        else:
            paths.append(p)
            ids.append(str(row[SITE_KEY]))
    print(f"[rerun] {len(paths)} of {len(df)} site(s) found"
          + (f"; not found (kept unchanged): {missing}" if missing else ""))
    status = run_sites(paths, cfg, site_ids=ids)
    for m in missing:
        status[m] = "meter file not found"
    rank_only(cfg)
    return status


# ======================================================================
# 12. Main
# ======================================================================
_GRID_CACHE = {}


def main(cfg=CONFIG):
    # Fail fast on a mistyped grid-load path, before minutes of analysis.
    grid_csv_path = cfg.get("grid_consumption", {}).get("csv_path")
    if grid_csv_path and not os.path.exists(grid_csv_path):
        raise FileNotFoundError(
            f"CONFIG['grid_consumption']['csv_path'] = {grid_csv_path!r} "
            f"does not exist (working directory: {os.getcwd()}). Fix the "
            f"path, or set it to None to skip the constraint-window "
            f"metrics on purpose."
        )

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

    ramps, ramp_warn, _, _ = ramp_analysis(power, cfg)
    peak_hour = peak_timing(raw_matrix, dates, cfg)

    temp_corr = None
    temp_corr_seasonal = None
    if cfg.get("weather_csv_path"):
        weather = load_smhi_temperature(cfg)
        temp_corr = temperature_correlation(power, weather, cfg)
        temp_corr_seasonal = seasonal_temperature_correlation(power, weather, cfg)

    constraint_window = None
    grid_load = None
    if grid_csv_path:
        print()
        key = (os.path.abspath(grid_csv_path), os.path.getmtime(grid_csv_path))
        if key not in _GRID_CACHE:            # one read per batch, not per site
            _GRID_CACHE.clear()
            _GRID_CACHE[key] = load_grid_consumption(cfg)
        grid_load = _GRID_CACHE[key]
        report_grid_customer_overlap(grid_load, power)
        constraint_hours, _ = derive_constraint_window_hours(grid_load, cfg)
        constraint_window = constraint_window_metrics(power, constraint_hours, metrics, cfg)
    else:
        print("\n[constraint window] no grid consumption data configured "
              "(CONFIG['grid_consumption']['csv_path'] is empty) -- "
              "skipping constraint-window relevancy metrics. Nothing is "
              "guessed or defaulted in its place; this and anything "
              "downstream of it (e.g. the willingness-proxy beta ratio, "
              "once built) stay explicitly None/N/A until this is set.")

    # Written before the plot: plt.show() blocks until the window is
    # closed, and the comparison row should not depend on that.
    extra = winter_and_hint_metrics(power, grid_load, cfg)
    print(f"\n[selection metrics] winter load factor {extra.get('load_factor_winter')}, "
          f"winter window {extra.get('constraint_window_hours_winter')}, "
          f"window load above base {extra.get('window_above_base_winter_kw')} kW")
    site_row = build_site_row(cfg, power, metrics, ldc_table, profiles, best_k,
                              labels, ramps, ramp_warn, peak_hour, monthly,
                              temp_corr, temp_corr_seasonal, constraint_window, extra)
    comparison = upsert_site_row(site_row, cfg)

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
        "temp_corr_seasonal": temp_corr_seasonal,
        "grid_load": grid_load,
        "constraint_window": constraint_window,
        "site_row": site_row,
        "comparison": comparison,
    }


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if "--rerun-all" in args:
        rerun_all_sites(CONFIG)
    elif "--rank-only" in args:
        rank_only(CONFIG)
    elif args:
        run_sites(args, CONFIG)
    else:
        main()

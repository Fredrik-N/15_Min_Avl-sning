"""
run_all.py -- runs the whole chain, in order, from the folder that holds the scripts.

    python run_all.py                  everything
    python run_all.py --dry-run        only print what would run
    python run_all.py --from offer     start at the step whose name contains "offer"
    python run_all.py --only figures   run one step
    python run_all.py --skip unilm,ladder   leave steps out
    python run_all.py --keep-going     continue after a step fails (default: stop)

Edit STEPS / ONE_SECOND_SITES below. The settings of each script stay in that script.
"""
import subprocess
import sys
import time

PY = sys.executable

# 1 s sites: (meter file, anonymised code, output folder). Leave [] if none.
# The output folder is what reducible_power.py / resolution_ladder.py are set to read.
ONE_SECOND_SITES = [
    # ("B_1s.csv", "B", "results_B"),
]

STEPS = [
    ("lpc rerun",    [PY, "lpc_11.py", "--rerun-all"]),       # screening of every site, writes site_comparison.csv
    ("lpc proxy",    [PY, "lpc_11.py", "--export-proxy"]),    # substation proxy (needs natbelastning); remove if unused
]
STEPS += [(f"unilm {code}", [PY, "unilm_fhmm.py", meter, "--out", out, "--site", code])
          for meter, code, out in ONE_SECOND_SITES]
STEPS += [
    ("sensitivity",  [PY, "filter_sensitivity.py"]),          # 81 threshold combinations
    ("reducible",    [PY, "reducible_power.py"]),
    ("offer",        [PY, "offerable_flex.py"]),
    ("ladder",       [PY, "resolution_ladder.py"]),
    ("baseline",     [PY, "baseline_test.py"]),
    ("figures",      [PY, "figures.py"]),
    ("tables",       [PY, "thesis_tables.py"]),               # last: reads everything above, redraws robustness_vs_value
]


def arg(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def main():
    only, start, skip = arg("--only"), arg("--from"), (arg("--skip") or "").split(",")
    steps = list(STEPS)
    if only:
        steps = [s for s in steps if only in s[0]]
    if start:
        i = next((k for k, s in enumerate(steps) if start in s[0]), None)
        if i is None:
            sys.exit(f"no step contains {start!r}: {[s[0] for s in steps]}")
        steps = steps[i:]
    steps = [s for s in steps if not any(k and k in s[0] for k in skip)]
    if "--dry-run" in sys.argv:
        for name, cmd in steps:
            print(f"{name:14s} {' '.join(cmd[1:])}")
        return
    results = []
    for name, cmd in steps:
        print(f"\n{'=' * 70}\n>> {name}: {' '.join(cmd[1:])}\n{'=' * 70}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd)
        results.append((name, rc, time.time() - t0))
        if rc != 0 and "--keep-going" not in sys.argv:
            print(f"\n!! {name} failed (exit {rc}); stopping. Fix it and run  python run_all.py --from {name.split()[0]}")
            break
    print("\nsummary")
    for name, rc, sec in results:
        print(f"  {'ok    ' if rc == 0 else 'FAILED'}  {name:14s} {sec:6.0f} s")
    if any(rc != 0 for _, rc, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

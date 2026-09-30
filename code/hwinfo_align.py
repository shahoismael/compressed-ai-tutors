#!/usr/bin/env python3
"""
hwinfo_align.py  --  ET&S study, step 3b (energy alignment)

Reads the HWiNFO sensor log and integrates measured package power over each
energy cell's recorded wall-clock window, giving MEASURED joules per assessed
item. Idle power from the session's own idle windows is subtracted, so the
reported figure is the energy attributable to inference.

Run AFTER a run_energy.py session, with HWiNFO logging stopped:

  py hwinfo_align.py --list          show the power columns in the CSV

--csv may be a file or a folder. Given a folder it takes the most recently
modified .csv in it, which is where HWiNFO drops the session log.
  py hwinfo_align.py                 align every cell that is not yet aligned
  py hwinfo_align.py --force         re-align cells already done
  py hwinfo_align.py --column "CPU Package Power [W]"

Nothing is estimated. A cell whose window is not covered by the log is left
unaligned and reported, never filled with a modelled number.
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(r"D:\claude_projects\ET&S")
ENERGY = ROOT / "06-results" / "energy"
HWINFO_DIR = Path(r"D:\claude_projects\Geochemistry\01-paper2\HWiNFO")
CSV_PATH = HWINFO_DIR

CPU_PATTERNS = [r"cpu package power", r"\bpackage power\b", r"cpu power",
                r"core \(tctl.*power", r"\bppt\b"]
# \b matters: it keeps "IGPU Power" out. On an integrated-graphics machine the
# iGPU rail is already inside CPU Package Power, so counting it again would
# inflate every figure. Only a discrete GPU rail should match here.
GPU_PATTERNS = [r"\bgpu package power", r"\bgpu power", r"\bgpu ppt",
                r"total graphics power"]

ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
TIME_FORMATS = ("%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S",
                "%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S",
                "%m/%d/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")


def read_csv(path):
    for enc in ENCODINGS:
        try:
            with open(path, newline="", encoding=enc) as f:
                rows = list(csv.reader(f))
            if rows:
                return rows, enc
        except UnicodeDecodeError:
            continue
    sys.exit(f"cannot decode {path} with any of {ENCODINGS}")


def parse_time(date_s, time_s):
    s = f"{date_s.strip()} {time_s.strip()}"
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


def to_float(v):
    if v is None:
        return None
    t = str(v).strip().replace(",", ".")
    if not t or t in ("-", "N/A", "Yes", "No"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def pick_column(header, patterns):
    best, best_rank = None, len(patterns)
    for i, h in enumerate(header):
        hl = h.lower()
        if "[w]" not in hl and not hl.endswith("power"):
            continue
        for rank, pat in enumerate(patterns):
            if re.search(pat, hl) and rank < best_rank:
                best, best_rank = i, rank
    return best


def load_series(path, cpu_col=None, gpu_col=None, list_only=False):
    rows, enc = read_csv(path)
    header = [h.strip() for h in rows[0]]
    if list_only:
        print(f"{path}  ({enc})  {len(rows)-1} logged rows\n")
        for i, h in enumerate(header):
            if "[w]" in h.lower() or h.lower().endswith("power"):
                print(f"  [{i:3d}]  {h}")
        return None

    ci = header.index(cpu_col) if cpu_col else pick_column(header, CPU_PATTERNS)
    if ci is None:
        sys.exit("no CPU power column found — run --list and pass --column")
    gi = header.index(gpu_col) if gpu_col else pick_column(header, GPU_PATTERNS)

    series = []
    for r in rows[1:]:
        if len(r) < 3:
            continue
        t = parse_time(r[0], r[1])
        if t is None:
            continue                      # HWiNFO appends Min/Avg/Max footer rows
        cpu = to_float(r[ci]) if ci < len(r) else None
        gpu = to_float(r[gi]) if (gi is not None and gi < len(r)) else None
        if cpu is None:
            continue
        series.append((t, cpu, gpu))
    series.sort(key=lambda x: x[0])
    print(f"CPU column : {header[ci]}")
    print(f"GPU column : {header[gi] if gi is not None else '(none)'}")
    print(f"samples    : {len(series)}")
    if series:
        span = (series[-1][0] - series[0][0]) / 3600
        print(f"log covers : {span:.2f} h  "
              f"({datetime.fromtimestamp(series[0][0])} -> "
              f"{datetime.fromtimestamp(series[-1][0])})")
    return series


def integrate(series, t0, t1, idx):
    """Trapezoidal joules over [t0, t1], with linear endpoint interpolation."""
    pts = [(s[0], s[idx]) for s in series
           if t0 <= s[0] <= t1 and s[idx] is not None]
    # interpolate the two endpoints so a 1 Hz log does not clip the window
    before = [s for s in series if s[0] < t0 and s[idx] is not None]
    after = [s for s in series if s[0] > t1 and s[idx] is not None]
    if before and pts:
        b = before[-1]
        f = (t0 - b[0]) / (pts[0][0] - b[0]) if pts[0][0] != b[0] else 0
        pts.insert(0, (t0, b[idx] + f * (pts[0][1] - b[idx])))
    if after and pts:
        a = after[0]
        f = (t1 - pts[-1][0]) / (a[0] - pts[-1][0]) if a[0] != pts[-1][0] else 0
        pts.append((t1, pts[-1][1] + f * (a[idx] - pts[-1][1])))

    if len(pts) < 2:
        return None
    j = dur = 0.0
    for (ta, va), (tb, vb) in zip(pts, pts[1:]):
        dt = tb - ta
        if 0 < dt < 30:
            j += 0.5 * (va + vb) * dt
            dur += dt
    if dur <= 0:
        return None
    vals = [v for _, v in pts]
    return {"joules": round(j, 3), "integrated_s": round(dur, 3),
            "window_s": round(t1 - t0, 3),
            "coverage": round(dur / (t1 - t0), 4),
            "mean_w": round(j / dur, 3),
            "min_w": min(vals), "max_w": max(vals), "n_samples": len(pts)}


def session_idle(series):
    """Mean idle watts across every recorded idle window in this energy folder."""
    cpu_w, gpu_w, used = [], [], []
    for p in sorted(ENERGY.glob("session__*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        for key in ("idle_start", "idle_end"):
            w = s.get(key)
            if not w:
                continue
            c = integrate(series, w["t_start_epoch"], w["t_end_epoch"], 1)
            g = integrate(series, w["t_start_epoch"], w["t_end_epoch"], 2)
            if c:
                cpu_w.append(c["mean_w"])
                used.append(f"{p.name}:{key}")
            if g:
                gpu_w.append(g["mean_w"])
    if not cpu_w:
        return None
    return {"cpu_idle_w": round(sum(cpu_w) / len(cpu_w), 3),
            "gpu_idle_w": round(sum(gpu_w) / len(gpu_w), 3) if gpu_w else None,
            "windows_used": used,
            "cpu_idle_spread_w": round(max(cpu_w) - min(cpu_w), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_PATH))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--column", help="exact CPU power column header")
    ap.add_argument("--gpu-column", help="exact GPU power column header")
    ap.add_argument("--force", action="store_true", help="re-align finished cells")
    ap.add_argument("--min-coverage", type=float, default=0.95,
                    help="reject a cell whose window is less well covered")
    a = ap.parse_args()

    path = Path(a.csv)
    if path.is_dir():
        logs = sorted(path.glob("*.csv"), key=lambda q: q.stat().st_mtime)
        if not logs:
            sys.exit(f"no .csv in {path}\n"
                     "  start HWiNFO logging into this folder before the energy run")
        print(f"log files  : {len(logs)} in {path}")
    else:
        if not path.exists():
            sys.exit(f"HWiNFO log not found: {path}\n"
                     "  start HWiNFO logging to this path before the energy run")
        logs = [path]

    # Every log is merged, so a campaign split across several nights (one CSV
    # per night) aligns in one pass. Overlapping samples are deduplicated.
    series = []
    for q in logs:
        print(f"\n--- {q.name} ---")
        s = load_series(q, a.column, a.gpu_column, a.list)
        if a.list:
            continue
        if s:
            series.extend(s)
    if a.list:
        return
    if not series:
        sys.exit("no usable samples in the log(s)")
    series.sort(key=lambda x: x[0])
    dedup, last_t = [], None
    for s in series:
        if last_t is None or s[0] - last_t > 1e-6:
            dedup.append(s)
            last_t = s[0]
    series = dedup
    print(f"\nmerged     : {len(series)} samples across {len(logs)} log(s)")

    idle = session_idle(series)
    if idle is None:
        sys.exit("no idle window in the log — cannot subtract idle, refusing to "
                 "report a net figure")
    print(f"idle CPU   : {idle['cpu_idle_w']} W "
          f"(spread {idle['cpu_idle_spread_w']} W across "
          f"{len(idle['windows_used'])} windows)")
    print(f"idle GPU   : {idle['gpu_idle_w']} W\n")

    aligned = skipped = uncovered = 0
    for p in sorted(ENERGY.glob("*.json")):
        if p.name.startswith(("session__", "summary")):
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("energy") and not a.force:
            skipped += 1
            continue

        t0, t1 = rec["t_start_epoch"], rec["t_end_epoch"]
        cpu = integrate(series, t0, t1, 1)
        gpu = integrate(series, t0, t1, 2)
        if cpu is None or cpu["coverage"] < a.min_coverage:
            cov = f"{cpu['coverage']:.2f}" if cpu else "0.00"
            print(f"  UNCOVERED  {rec['cell']}  (coverage {cov}) — left unaligned")
            uncovered += 1
            continue

        n = max(rec["items"], 1)
        net_cpu = round(cpu["joules"] - idle["cpu_idle_w"] * cpu["integrated_s"], 3)
        net_gpu = None
        if gpu and idle["gpu_idle_w"] is not None:
            net_gpu = round(gpu["joules"] - idle["gpu_idle_w"] * gpu["integrated_s"], 3)
        net = net_cpu + (net_gpu or 0.0)
        tok = rec.get("tokens_generated") or 0

        rec["energy"] = {
            "source": path.name,
            "cpu_gross": cpu,
            "gpu_gross": gpu,
            "idle_cpu_w": idle["cpu_idle_w"],
            "idle_gpu_w": idle["gpu_idle_w"],
            "net_cpu_j": net_cpu,
            "net_gpu_j": net_gpu,
            "net_total_j": round(net, 3),
            "j_per_item": round(net / n, 4),
            "j_per_token": round(net / tok, 6) if tok else None,
            "gross_total_j": round(cpu["joules"] + (gpu["joules"] if gpu else 0), 3),
        }
        p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"  {rec['cell']:52s} {rec['energy']['j_per_item']:8.3f} J/item")
        aligned += 1

    print(f"\naligned {aligned}, already done {skipped}, uncovered {uncovered}")
    if uncovered:
        print("uncovered cells need the HWiNFO log that was running at their "
              "timestamps; nothing is estimated in their place")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
power_probe.py  --  ET&S study, energy instrumentation

Reads REAL package power from LibreHardwareMonitor's built-in web server and
integrates it to joules over a timed window. No TDP estimates anywhere: if the
power sensors are absent this module refuses to run.

Host requirements
  1. LibreHardwareMonitor running AS ADMINISTRATOR
     (admin is required for the Intel RAPL MSRs; without it no Power sensor
      appears and --check will say so)
  2. Options -> Remote Web Server -> Run   (default port 8085)

Self-test
  py power_probe.py --check
  py power_probe.py --idle 120
"""

import argparse
import json
import threading
import time
import urllib.request

LHM_URL = "http://localhost:8085/data.json"
POLL_HZ = 5.0          # samples per second
MAX_GAP_S = 5.0        # a gap longer than this is not integrated over


def fetch(url=LHM_URL, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def walk(node, path=()):
    """Yield (path_tuple, node) for every node in the LHM tree."""
    text = node.get("Text", "")
    here = path + (text,) if text else path
    yield here, node
    for c in node.get("Children", []) or []:
        yield from walk(c, here)


def parse_w(value):
    """'12.3 W' or '12,3 W' -> 12.3 ; anything else -> None."""
    if value is None:
        return None
    t = str(value).strip().replace(",", ".")
    t = t.replace("W", "").replace("w", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


def power_sensors(root):
    """Every sensor that reports watts, with its full tree path."""
    out = []
    for path, n in walk(root):
        sid = (n.get("SensorId") or "")
        is_power = "/power/" in sid.lower() or n.get("Type") == "Power"
        if not is_power:
            continue
        v = parse_w(n.get("Value"))
        if v is None:
            continue
        out.append(
            {
                "sensor_id": sid,
                "text": n.get("Text", ""),
                "path": " / ".join(path),
                "value_w": v,
            }
        )
    return out


def _score(s, kind):
    """Rank candidate sensors; highest score wins."""
    t = s["text"].lower()
    p = s["path"].lower()
    sid = s["sensor_id"].lower()
    sc = 0
    if kind == "cpu":
        if "cpu" in p or "cpu" in sid or "intelcpu" in sid or "amdcpu" in sid:
            sc += 10
        if "package" in t:
            sc += 8
        if t in ("cpu package", "package"):
            sc += 4
        if "cores" in t or "graphics" in t or "memory" in t or "uncore" in t:
            sc -= 6          # sub-domains, not the package total
        if "platform" in t or "psys" in t:
            sc += 2          # whole-SoC rail, better than package if present
    else:
        if "gpu" in p or "gpu" in sid:
            sc += 10
        if "package" in t or "total" in t or t == "gpu power":
            sc += 6
    return sc


def pick(sensors, kind):
    cands = [s for s in sensors if _score(s, kind) > 0]
    if not cands:
        return None
    return max(cands, key=lambda s: _score(s, kind))


class PowerSampler(threading.Thread):
    """Background thread: polls LHM and stores (t, cpu_w, gpu_w) samples."""

    def __init__(self, cpu_sid, gpu_sid=None, url=LHM_URL, hz=POLL_HZ):
        super().__init__(daemon=True)
        self.cpu_sid = cpu_sid
        self.gpu_sid = gpu_sid
        self.url = url
        self.period = 1.0 / hz
        self.samples = []
        self.errors = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                root = fetch(self.url)
                vals = {}
                for _, n in walk(root):
                    sid = n.get("SensorId")
                    if sid and sid in (self.cpu_sid, self.gpu_sid):
                        vals[sid] = parse_w(n.get("Value"))
                self.samples.append(
                    (t, vals.get(self.cpu_sid), vals.get(self.gpu_sid))
                )
            except Exception:
                self.errors += 1
            rest = self.period - (time.perf_counter() - t)
            if rest > 0:
                self._stop.wait(rest)

    def stop(self):
        self._stop.set()
        self.join(timeout=10)


def integrate(samples, which="cpu"):
    """Trapezoidal integration of watts over time -> joules."""
    i = 1 if which == "cpu" else 2
    j = 0.0
    dur = 0.0
    vals = []
    prev = None
    for row in samples:
        v = row[i]
        if v is None:
            prev = None
            continue
        if prev is not None:
            dt = row[0] - prev[0]
            if 0 < dt < MAX_GAP_S:
                j += 0.5 * (prev[1] + v) * dt
                dur += dt
        prev = (row[0], v)
        vals.append(v)
    return {
        "joules": round(j, 3),
        "integrated_s": round(dur, 3),
        "mean_w": round(j / dur, 3) if dur > 0 else None,
        "min_w": min(vals) if vals else None,
        "max_w": max(vals) if vals else None,
        "n_samples": len(vals),
    }


def measure_idle(cpu_sid, gpu_sid, seconds, url=LHM_URL):
    """Machine must be otherwise idle. Returns the same dict shape."""
    s = PowerSampler(cpu_sid, gpu_sid, url)
    s.start()
    time.sleep(seconds)
    s.stop()
    return {
        "seconds": seconds,
        "cpu": integrate(s.samples, "cpu"),
        "gpu": integrate(s.samples, "gpu"),
        "poll_errors": s.errors,
    }


def require_sensors(url=LHM_URL):
    """Returns (cpu_sensor, gpu_sensor_or_None). Raises if CPU power absent."""
    try:
        root = fetch(url)
    except Exception as e:
        raise SystemExit(
            f"cannot reach LibreHardwareMonitor at {url}\n"
            f"  {type(e).__name__}: {e}\n"
            "  start LHM as administrator, then Options -> Remote Web Server -> Run"
        )
    sens = power_sensors(root)
    cpu = pick(sens, "cpu")
    if cpu is None:
        raise SystemExit(
            "LHM is reachable but reports no CPU power sensor.\n"
            "  This means LHM was NOT started as administrator, so the RAPL\n"
            "  MSRs are unreadable. Close it and run it as administrator.\n"
            f"  sensors seen: {[s['text'] for s in sens] or 'none'}"
        )
    return cpu, pick(sens, "gpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=LHM_URL)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--idle", type=int, metavar="SECONDS")
    a = ap.parse_args()

    if a.check:
        root = fetch(a.url)
        sens = power_sensors(root)
        print(f"{len(sens)} power sensor(s) at {a.url}\n")
        for s in sens:
            print(f"  {s['value_w']:8.2f} W   {s['text']:<24s} {s['sensor_id']}")
            print(f"              {s['path']}")
        cpu, gpu = require_sensors(a.url)
        print(f"\nselected CPU : {cpu['text']}  [{cpu['sensor_id']}]")
        print(f"selected GPU : {gpu['text'] if gpu else '(none — CPU-only run)'}")
        print("\nOK — ready to measure.")
        return

    if a.idle:
        cpu, gpu = require_sensors(a.url)
        print(f"measuring idle for {a.idle}s — do not touch the machine ...")
        r = measure_idle(cpu["sensor_id"], gpu["sensor_id"] if gpu else None,
                         a.idle, a.url)
        print(json.dumps(r, indent=2))
        return

    ap.error("use --check or --idle SECONDS")


if __name__ == "__main__":
    main()

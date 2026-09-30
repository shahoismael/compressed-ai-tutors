#!/usr/bin/env python3
"""
build_ris.py  --  ET&S study, citation master build

Reads every per-paper citation file and writes ONE normalised RIS master:

  01-Methodology-Papers/citation/1..11   -> ETS01..ETS11
  00-Literature-Papers/citation/12..50   -> ETS12..ETS50
  output: 09-manuscript-drafts/ets.ris

Handles three source formats: .ris, .bib (BibTeX) and .txt (plain APA 7).
Abstracts are dropped; every bibliographic field is kept. Tag order is fixed
so the file imports cleanly into Mendeley and EndNote.

Run:  py build_ris.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(r"D:\claude_projects\ET&S")
SRC = [(ROOT / "01-Methodology-Papers" / "citation", range(1, 12)),
       (ROOT / "00-Literature-Papers" / "citation", range(12, 51))]
OUT = ROOT / "09-manuscript-drafts" / "ets.ris"

DROP = {"AB", "N2", "N1", "M3", "Y2", "L1", "L2", "L3", "L4", "C1", "C2",
        "DB", "TY", "SE", "JA", "AD", "CY", "ET", "Y1", "DA"}

# Fields the source export genuinely omits. Each is taken from the publisher's
# own landing page, not inferred, and is recorded here so the gap is auditable.
OVERRIDES = {
    "ETS22": {"AU": ["Qureshi, Abdul Rehman", "Zakria", "Asif, Muhammad",
                     "Khokhar, Muhammad Saddam", "Mangnejo, Safdar Hussain"],
              "TI": ["Towards trustworthy automated essay scoring: Explainable "
                     "transformers and efficient fine-tuning strategies"],
              "PY": ["2026"]},
    "ETS37": {"AU": ["Yalcin, Erva Nur Sultan", "Cubukcu Cerasi, Ceren"],
              "TI": ["Performance and energy efficiency analysis of language "
                     "models of different sizes"],
              "PY": ["2026"],
              "JO": ["Darnios Aplinkos Vystymas"],
              "KW": ["Original title: Ivairiu dydziu kalbos modeliu nasumo ir "
                     "energijos vartojimo efektyvumo analize"]},
}
JOURNAL_FIX = {
    "ITM Web Conf.": "ITM Web of Conferences",
    "Sage Open": "SAGE Open",
    "AI Open": "AI Open",
}

ORDER = ["TY", "TI", "AU", "PY", "DA", "JO", "VL", "IS", "SP", "EP",
         "PB", "SN", "DO", "UR", "KW", "LA", "ID"]

DASH = re.compile(r"[\u2010-\u2015\u2212]")


def clean(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def normalise(rec):
    """Repair the defects the publisher exports leave behind."""
    # year: '%2026/%06/%16', '2025///', '2024/02/26' -> '2026'
    for tag in ("PY", "Y1", "DA"):
        if rec.get(tag):
            y = re.search(r"(19|20)\d{2}", rec[tag][0])
            if y and tag == "PY":
                rec["PY"] = [y.group(0)]
            elif y and not rec.get("PY"):
                rec["PY"] = [y.group(0)]
    # volume: 'Volume 9 - 2024' -> '9'
    if rec.get("VL"):
        v = re.search(r"\d+", rec["VL"][0])
        rec["VL"] = [v.group(0)] if v else []
    # pages: SP '11-22' -> SP 11 / EP 22
    if rec.get("SP") and not rec.get("EP"):
        m = re.match(r"^\s*(\d+)\s*[-\u2010-\u2015]\s*(\d+)\s*$", rec["SP"][0])
        if m:
            rec["SP"], rec["EP"] = [m.group(1)], [m.group(2)]
    # DOI: fall back to M3, then to any URL that carries one
    if not rec.get("DO"):
        for tag in ("M3", "UR", "N1", "ID"):
            for v in rec.get(tag, []):
                d = re.search(r"10\.\d{4,9}/[^\s\"<>]+", v)
                if d:
                    rec["DO"] = [d.group(0).rstrip(".")]
                    break
            if rec.get("DO"):
                break
    if rec.get("DO"):
        rec["UR"] = [f"https://doi.org/{rec['DO'][0]}"]
    # DOI must be the bare identifier, not a resolver URL (Elsevier/Wiley exports)
    if rec.get("DO"):
        d = re.sub(r"^\s*(https?://(dx\.)?doi\.org/|doi:\s*)", "", rec["DO"][0], flags=re.I)
        rec["DO"] = [d.strip().rstrip(".")]
        rec["UR"] = [f"https://doi.org/{rec['DO'][0]}"]
    # authors: IEEE exports give 'P. Tikka'; RIS wants 'Tikka, P.'
    if rec.get("AU"):
        fixed = []
        for a in rec["AU"]:
            a = clean(a)
            m = re.match(r"^((?:[A-ZÀ-ÖØ-Þ]\.\s*)+)([A-ZÀ-ÖØ-Þ].*)$", a)
            fixed.append(f"{m.group(2).strip()}, {clean(m.group(1))}" if m else a)
        rec["AU"] = fixed
    # journal titles the exports abbreviate or mis-case
    if rec.get("JO"):
        rec["JO"] = [JOURNAL_FIX.get(rec["JO"][0].strip(), rec["JO"][0].strip())]
    # drop empty values
    return {k: [x for x in v if clean(x)] for k, v in rec.items()}


# ---------------------------------------------------------------- .ris

def parse_ris(text):
    rec, tag = {}, None
    for line in text.splitlines():
        m = re.match(r"^([A-Z][A-Z0-9])  - ?(.*)$", line)
        if m:
            tag, val = m.group(1), m.group(2).strip()
            if tag == "ER":
                break
            rec.setdefault(tag, []).append(val)
        elif tag and line.strip():
            rec[tag][-1] += " " + line.strip()
    # unify journal-name tags
    for alt in ("T2", "JF", "JA", "T3"):
        if alt in rec and "JO" not in rec:
            rec["JO"] = rec.pop(alt)
        rec.pop(alt, None)
    for alt in ("T1",):
        if alt in rec and "TI" not in rec:
            rec["TI"] = rec.pop(alt)
        rec.pop(alt, None)
    for alt in ("Y1",):
        if alt in rec and "PY" not in rec:
            rec["PY"] = [rec[alt][0][:4]]
        rec.pop(alt, None)
    return rec


# ---------------------------------------------------------------- .bib

def parse_bib(text):
    fields = {}
    for m in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*[,}]?\s*(?=\n\s*\w+\s*=|\s*\}\s*$)",
                         text, re.S):
        fields[m.group(1).upper()] = clean(m.group(2))
    rec = {}
    if "AUTHOR" in fields:
        rec["AU"] = [clean(a) for a in re.split(r"\s+and\s+", fields["AUTHOR"]) if a.strip()]
    for src, dst in (("TITLE", "TI"), ("JOURNAL", "JO"), ("YEAR", "PY"),
                     ("DOI", "DO"), ("ISSN", "SN"), ("PUBLISHER", "PB"),
                     ("NUMBER", "IS"), ("PAGES", "SP")):
        if src in fields:
            rec[dst] = [fields[src]]
    if "VOLUME" in fields:
        v = re.search(r"\d+", fields["VOLUME"])
        if v:
            rec["VL"] = [v.group(0)]
    if "DO" in rec:
        rec["UR"] = [f"https://doi.org/{rec['DO'][0]}"]
    return rec


# ---------------------------------------------------------------- .txt (APA 7)

AUTHOR_RE = re.compile(
    r"([A-ZÀ-ÖØ-Þ][^\s,]*(?:[- ][A-ZÀ-ÖØ-Þ][^\s,]*)*),\s*((?:[A-ZÀ-ÖØ-Þ]\.\s*)+)"
)


def parse_apa(text):
    t = clean(text.replace("Cite as:", " "))
    t = re.sub(r"\s*(Published|Submitted)\s+[A-Z][a-z]+\s+\d.*$", "", t)

    ym = re.search(r"\((\d{4})\)\.\s*", t)
    if not ym:
        raise ValueError("no (year). found")
    authors_str, rest = t[: ym.start()], t[ym.end():]
    year = ym.group(1)

    authors = [f"{s.strip()}, {clean(i)}" for s, i in AUTHOR_RE.findall(authors_str)]
    if not authors:
        raise ValueError("no authors parsed")

    rec = {"AU": authors, "PY": [year]}

    anchor = re.search(r",\s*(\d+)\s*\((\d+)\)\s*,\s*(\d+)\s*[-\u2010-\u2015]\s*(\d+)\.", rest)
    if anchor:
        rec["VL"], rec["IS"] = [anchor.group(1)], [anchor.group(2)]
        rec["SP"], rec["EP"] = [anchor.group(3)], [anchor.group(4)]
        head = rest[: anchor.start()]
    else:
        head = rest.split("https://")[0].rstrip(". ")
    i = head.rfind(". ")
    if i == -1:
        raise ValueError("cannot split title/journal")
    rec["TI"], rec["JO"] = [clean(head[:i])], [clean(head[i + 2:])]

    d = re.search(r"https?://doi\.org/(\S+?)\s*$", rest) or re.search(r"10\.\d{4,}/\S+", rest)
    if d:
        doi = d.group(1) if d.lastindex else d.group(0)
        rec["DO"] = [doi.rstrip(".")]
        rec["UR"] = [f"https://doi.org/{rec['DO'][0]}"]
    return rec


# ---------------------------------------------------------------- emit

def emit(rec, ets_id):
    rec = {k: v for k, v in rec.items() if k not in DROP and v}
    rec["ID"] = [ets_id]
    lines = ["TY  - JOUR"]
    for tag in ORDER:
        if tag == "TY":
            continue
        for v in rec.get(tag, []):
            lines.append(f"{tag}  - {clean(v)}")
    for tag in sorted(set(rec) - set(ORDER)):
        for v in rec[tag]:
            lines.append(f"{tag}  - {clean(v)}")
    lines.append("ER  - ")
    return "\n".join(lines)


def main():
    records, problems = [], []
    for folder, nums in SRC:
        for n in nums:
            hits = [p for p in folder.iterdir()
                    if p.stem == str(n) and p.suffix.lower() in (".ris", ".bib", ".txt")]
            if not hits:
                problems.append(f"{n}: FILE MISSING in {folder.name}")
                continue
            p = hits[0]
            raw = p.read_text(encoding="utf-8", errors="replace")
            try:
                if p.suffix.lower() == ".ris":
                    rec = parse_ris(raw)
                elif p.suffix.lower() == ".bib":
                    rec = parse_bib(raw)
                else:
                    rec = parse_apa(raw)
            except Exception as e:
                problems.append(f"{n} ({p.name}): PARSE FAILED - {e}")
                continue
            ets = f"ETS{n:02d}"
            rec = normalise(rec)
            if ets in OVERRIDES:
                rec.update(OVERRIDES[ets])
            for req in ("TI", "AU", "PY", "DO"):
                if not rec.get(req):
                    problems.append(f"{ets} ({p.name}): missing {req}")
            records.append((n, ets, p.name, rec))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n\n".join(emit(r, e) for _, e, _, r in records) + "\n",
                   encoding="utf-8")

    dois = {}
    for _, e, _, r in records:
        d = (r.get("DO") or [""])[0].lower()
        if d:
            dois.setdefault(d, []).append(e)
    dupes = {d: v for d, v in dois.items() if len(v) > 1}

    print(f"wrote {len(records)} records -> {OUT}")
    print(f"unique DOIs: {len(dois)}")
    if dupes:
        print("DUPLICATE DOIs:")
        for d, v in dupes.items():
            print(f"  {d}  {v}")
    if problems:
        print("\nPROBLEMS:")
        for x in problems:
            print("  " + x)
    else:
        print("no problems")
    return 1 if problems or dupes else 0


if __name__ == "__main__":
    sys.exit(main())

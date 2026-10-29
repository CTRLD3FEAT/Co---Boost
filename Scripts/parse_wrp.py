#!/usr/bin/env python3
"""
parse_wrp.py - Extract commodity prices from Sri Lanka DCS
"Open Market Weekly Average Retail Prices" PDF reports.

For every commodity row it keeps: commodity, parent, child, unit, and the price
of the report's own week (the 3rd price column, e.g. "3rd Week Dec. 2019").

Rules implemented
  * Table headers, titles and footnotes are never records.
  * Child rows (indented rows such as "No 2." under "Samba - No 1.", or
    "Imported" under "B.Onions - Local") become independent rows and get the
    parent name added:  "Samba - No 2.",  "B.Onions - Imported".
  * Every row is independent: parent rows and child rows each have their own price.
  * Anything unclear (no price printed, "n.a.", columns that do not line up,
    price outside the printed range, ...) is NOT guessed. It is excluded and
    listed in the notes printed at the end (and saved to *_excluded.csv).

Usage
  pip install pdfplumber
  python parse_wrp.py DCS-WRP-2019-12-W3.pdf DCS-WRP-2026-10-W1.pdf -o prices.csv
  python parse_wrp.py ./reports/            # a folder of PDFs also works
"""
import argparse
import calendar
import csv
import re
import sys
from pathlib import Path

import pdfplumber

# ----------------------------------------------------------------- settings
LINE_TOL = 2.5          # pt: words whose 'top' differ by less are the same line
NAME_UNIT_GAP = 8.0     # pt: a gap wider than this separates item name from unit
INDENT_TOL = 5.0        # pt: first word this far right of the margin = child row
COL_GAP = 12.0          # pt: gap between value columns when clustering x positions
EXPECTED_COLS = 7       # 3 prices, 2 % changes, price-range min and max
LEGACY_COLS = 5         # legacy layout: 3 prices + 2 % changes (no range)
PRICE_COL = 2           # 0 = year-ago, 1 = previous week, 2 = THIS week
RANGE_COLS = (5, 6)     # min / max of this week

VALUE_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?%?$")
# "n.a." or a bare dash "-" = value not printed for that column
NA_RE = re.compile(r"^(?:n\.a\.?|-)$", re.I)
TITLE_RE = re.compile(r"(\d+)(?:st|nd|rd|th)\s+Week\s+of\s+([A-Za-z]+)\s+(\d{4})", re.I)
FILE_RE = re.compile(r"(\d{4})-(\d{1,2})-W(\d)", re.I)
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
# A data row ends its left side with a unit token (kg / g / ml / Nuts / ...);
# header continuation lines end with week names, years, "%", ":" etc.
UNIT_TAIL_RE = re.compile(
    r"(?:kg|kg\.|k\.g\.|g|g\.|gm\.|ml|ml\.|each|bunch|pkt|nuts|leaves)\.?$", re.I)
NO_UNIT_WORDS = {"compared", "%", "chg.", "avg.price", "rs.cts.", "week",
                 "month", ":", "source"}


# ------------------------------------------------------------------ helpers
def clean(text):
    """Collapse whitespace and tidy '( Large )' -> '(Large)'."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\(\s+", "(", text)
    return re.sub(r"\s+\)", ")", text)


def is_value(text):
    return bool(VALUE_RE.match(text) or NA_RE.match(text))


def group_lines(words):
    lines = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(w["top"] - lines[-1]["top"]) <= LINE_TOL:
            lines[-1]["words"].append(w)
        else:
            lines.append({"top": w["top"], "words": [w]})
    for ln in lines:
        ln["words"].sort(key=lambda w: w["x0"])
    return lines


def cluster(xs):
    xs = sorted(xs)
    groups = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] > COL_GAP:
            groups.append([x])
        else:
            groups[-1].append(x)
    return [sum(g) / len(g) for g in groups]


def split_name_unit(tokens):
    """tokens (left of the numbers) -> (name, unit) using the widest big gap."""
    split_at = None
    for i in range(1, len(tokens)):
        if tokens[i]["x0"] - tokens[i - 1]["x1"] > NAME_UNIT_GAP:
            split_at = i          # last wide gap wins (units are on the right)
    if split_at is None:
        return None, None
    name = clean(" ".join(t["text"] for t in tokens[:split_at]))
    unit = clean(" ".join(t["text"] for t in tokens[split_at:]))
    return name, unit


def split_parent_child(name):
    """'Samba - No 1.' -> ('Samba','No 1.');  'Coconut (Large)' -> ('Coconut','(Large)')."""
    parts = re.split(r"\s+-\s*", name, maxsplit=1)
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0].strip(), parts[1].strip()
    m = re.match(r"^(.*?)\s*(\([^)]*\))$", name)
    if m and m.group(1):
        return m.group(1).strip(), m.group(2)
    return name, ""


def report_period(title_text, filename):
    """Canonical reporting period. Policy: the file name is the authority
    (DCS indexing scheme); Printed titles occasionally contain copy-paste
    errors (observed: DCS-WRP-2024-01-W4.pdf headlines 'January 2023') --
    such mismatches are logged, never guessed away."""
    year = month = week = None
    title_match = TITLE_RE.search(title_text or "")
    f = FILE_RE.search(filename)
    notes = []
    if f:
        year, month, week = int(f.group(1)), int(f.group(2)), int(f.group(3))
    if title_match and title_match.group(2).lower() in MONTHS:
        tw, tmo, ty = int(title_match.group(1)), MONTHS[title_match.group(2).lower()], \
            int(title_match.group(3))
        if not f:
            year, month, week = tw, tmo, ty
            notes.append("period taken from title (file name has no Wn pattern)")
        elif (ty, tmo, tw) != (year, month, week):
            notes.append(f"TITLE/FN MISMATCH: title says {ty}-{tmo:02d}-W{tw} "
                         f"but file name says {year}-{month:02d}-W{week}; file name used")
    elif not f:
        notes.append("report period could not be determined (no title match, "
                     "no file name pattern)")
    return year, month, week, notes


_ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}
_MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December"]


def canonical_title(week, month, year):
    if week is None or month is None or year is None:
        return ""
    return f"{_ORDINAL.get(week, str(week) + 'th')} Week of {_MONTH_NAMES[month - 1]} {year}"


# ------------------------------------------------------------ page parsing
def parse_page(page):
    """Return (title, rows, problems). rows = list of dicts (raw, unvalidated)."""
    words = page.extract_words(x_tolerance=1.5, y_tolerance=2)
    lines = group_lines(words)
    if not lines:
        return None, [], ["page has no text"]
    title = lines[0]["words"] and " ".join(w["text"] for w in lines[0]["words"])

    # Data starts below the table header line. Newer reports end it with
    # "this week"; legacy reports (2020 - mid 2021) use an "Item  Unit" line.
    header_idx = None
    for i, ln in enumerate(lines):
        texts = [w["text"].lower() for w in ln["words"]]
        if "this" in texts and "week" in texts:
            header_idx = i
            break
        if "item" in texts and "unit" in texts:
            header_idx = i
            break
    if header_idx is None:
        return title, [], ["table header ('... this week') not found on this page"]

    body = []
    for ln in lines[header_idx + 1:]:
        texts = [w["text"].lower() for w in ln["words"]]
        if texts and texts[0].startswith("source"):
            break                               # footer: Source / n.a. / markets list
        # Header continuation lines (legacy layout spills over) have no unit
        # on the right end and contain header keywords. Skip those only; keep
        # anything that might be a record (even one ending in "n.a.").
        tail_is_unit = texts and UNIT_TAIL_RE.match(texts[-1])
        kw_hit = any(any(k in t for k in ("week", "compared", "chg",
                                         "avg.price", "rs.cts", "item", "unit"))
                     for t in texts)
        if not tail_is_unit and kw_hit:
            continue
        body.append(ln)
    if not body:
        return title, [], ["no data rows found below the header"]

    # Split each line into name/unit tokens (left) and value tokens (right).
    # A dash "-" is a printed placeholder (column not applicable); keep it with
    # the value tokens so that a middle column does not stop the right-tail scan.
    parsed = []
    for ln in body:
        toks = ln["words"]
        k = len(toks)
        while k > 0 and (is_value(toks[k - 1]["text"]) or toks[k - 1]["text"] in ("-", "–")):
            k -= 1
        parsed.append({"text": " ".join(t["text"] for t in toks),
                       "left": toks[:k], "values": toks[k:]})

    # Column positions for this page from right-aligned numbers (n.a. is not aligned).
    xs = [v["x1"] for p in parsed for v in p["values"] if not NA_RE.match(v["text"])]
    if not xs:
        return title, [], ["no numeric values found on this page"]
    centers = cluster(xs)
    problems = []

    # --- classify this page's layout from the data itself ----------------
    # Which clusters hold values ending in "%" identifies the % columns;
    # the layout (and thus the price column) follows from that, not from
    # a preset column map.
    pct = {c: 0 for c in range(len(centers))}
    num = {c: 0 for c in range(len(centers))}
    for p in parsed:
        for v in p["values"]:
            if NA_RE.match(v["text"]):
                continue
            col = min(range(len(centers)), key=lambda i: abs(centers[i] - v["x1"]))
            if v["text"].endswith("%"):
                pct[col] += 1
            else:
                num[col] += 1
    chg_cols = {c for c in range(len(centers))
                if (pct[c] + num[c]) > 0 and pct[c] / (pct[c] + num[c]) > 0.3}

    if len(centers) == EXPECTED_COLS:
        price_col, range_cols = PRICE_COL, RANGE_COLS          # 3 prices + 2 % + range
    elif len(centers) == LEGACY_COLS:
        if chg_cols == {3, 4}:          # legacy 2020-mid2021: 3 prices + 2 %, no range
            price_col, range_cols = 2, None
        elif chg_cols == {2}:           # transitional: no prev-week column at all
            price_col, range_cols = 1, (3, 4)                  # year-ago|THIS|%|min|max
        else:
            return title, [], [f"ambiguous 5-column layout (% columns "
                               f"{sorted(chg_cols)}); page skipped to avoid wrong prices"]
    else:
        return title, [], [f"found {len(centers)} value columns instead of "
                           f"{EXPECTED_COLS} or {LEGACY_COLS}; page skipped "
                           "to avoid wrong prices"]

    margin = min(p["left"][0]["x0"] for p in parsed if p["left"])
    rows = []
    for p in parsed:
        row = {"line": p["text"], "name": None, "unit": None, "indent": False,
               "cells": {}, "error": None}
        if not p["left"]:
            row["error"] = "line has numbers but no item name"
            rows.append(row)
            continue
        row["indent"] = p["left"][0]["x0"] > margin + INDENT_TOL
        row["name"], row["unit"] = split_name_unit(p["left"])
        if row["name"] is None:
            row["error"] = "could not separate item name from unit"
            row["name"] = clean(" ".join(t["text"] for t in p["left"]))
            rows.append(row)
            continue
        cells = {}
        for v in p["values"]:                       # real values, right-aligned
            if NA_RE.match(v["text"]):
                continue
            col = min(range(len(centers)), key=lambda i: abs(centers[i] - v["x1"]))
            if col in cells:
                row["error"] = "two values fell into the same column"
            cells[col] = v["text"]
        for v in p["values"]:                       # n.a./dash placeholders
            if not NA_RE.match(v["text"]):
                continue
            pos = (v["x0"] + v["x1"]) / 2 + 11      # unaligned: use its centre
            col = min(range(len(centers)), key=lambda i: abs(centers[i] - pos))
            if col not in cells:                    # never overwrite a real value
                cells[col] = v["text"]
        # sanity: % values only in a % column; a number never in a % column
        bad = [c for c, t in cells.items()
               if (t.endswith("%") != (c in chg_cols)) and not NA_RE.match(t)]
        if bad:
            row["error"] = "values do not line up with the table columns"
        row["cells"] = cells
        row["price_col"], row["range_cols"] = price_col, range_cols
        rows.append(row)
    return title, rows, problems


# -------------------------------------------------------------- file level
def parse_pdf(path):
    records, excluded, notes = [], [], []
    path = Path(path)
    parent = None
    title_text = None
    page_data = []
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            title, rows, problems = parse_page(page)
            if title_text is None and title and TITLE_RE.search(title):
                title_text = title
            for pr in problems:
                notes.append(f"{path.name} page {pno}: {pr}")
            page_data.append((pno, rows))

    year, month, week, pnotes = report_period(title_text, path.name)
    notes += [f"{path.name}: {n}" for n in pnotes]
    if year is None:
        notes.append(f"{path.name}: report week could not be determined")
    period_label = canonical_title(week, month, year)

    def exclude(pno, row, reason):
        excluded.append({"source_file": path.name, "page": pno,
                         "line": row["line"], "reason": reason})

    for pno, rows in page_data:
        for row in rows:
            if row["error"]:
                exclude(pno, row, row["error"])
                continue
            name, unit = row["name"], row["unit"]

            # --- work out parent / child naming (done before price checks so that
            #     rows with missing prices still keep the parent chain intact)
            if row["indent"]:
                if parent is None:
                    exclude(pno, row, "indented child row but no parent row above it")
                    continue
                child = name
                commodity = f"{parent} - {child}"
            else:
                parent, child = split_parent_child(name)
                commodity = f"{parent} - {child}" if child and not child.startswith("(") \
                    else (f"{parent} {child}" if child else parent)

            cells = row["cells"]
            price_col = row["price_col"]
            if price_col not in cells:
                exclude(pno, row, "no price printed for this week")
                continue
            raw = cells[price_col]
            if NA_RE.match(raw):
                exclude(pno, row, f"price for this week is '{raw}' (not available)")
                continue
            price = float(raw.replace(",", ""))
            if price <= 0:                       # report itself printed 0.00 (n.a. in disguise)
                exclude(pno, row, f"price for this week is printed as {raw} (not available)")
                continue
            lo_t = hi_t = None
            if row["range_cols"]:
                lo_t, hi_t = (cells.get(c) for c in row["range_cols"])
            if lo_t and hi_t and not NA_RE.match(lo_t) and not NA_RE.match(hi_t):
                lo, hi = float(lo_t.replace(",", "")), float(hi_t.replace(",", ""))
                if not (lo - 0.01 <= price <= hi + 0.01):
                    exclude(pno, row, f"price {price} is outside the printed range {lo}-{hi}")
                    continue
            records.append({
                "source_file": path.name, "report_week": period_label,
                "year": year, "month": month, "week_of_month": week,
                "commodity": commodity, "parent": parent, "child": child,
                "unit": unit, "price_rs": price})
    return records, excluded, notes


# --------------------------------------------------------------------- main
FIELDS = ["source_file", "report_week", "year", "month", "week_of_month",
          "commodity", "parent", "child", "unit", "price_rs"]


def collect_pdfs(paths):
    out = []
    for p in map(Path, paths):
        out += sorted(p.glob("*.pdf")) if p.is_dir() else [p]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pdfs", nargs="+", help="PDF files or folders")
    ap.add_argument("-o", "--output", default="weekly_prices.csv")
    ap.add_argument("--excluded", help="CSV for excluded rows (default: <output>_excluded.csv)")
    args = ap.parse_args(argv)

    all_rec, all_exc, all_notes = [], [], []
    for pdf in collect_pdfs(args.pdfs):
        try:
            rec, exc, notes = parse_pdf(pdf)
        except Exception as e:                       # unreadable / not a PDF
            all_notes.append(f"{pdf}: could not be read ({e})")
            continue
        print(f"{pdf.name}: {len(rec)} price records, {len(exc)} excluded")
        all_rec += rec
        all_exc += exc
        all_notes += notes

    out = Path(args.output)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rec)
    exc_path = Path(args.excluded) if args.excluded else out.with_name(out.stem + "_excluded.csv")
    with exc_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["source_file", "page", "line", "reason"])
        w.writeheader()
        w.writerows(all_exc)

    print(f"\nSaved {len(all_rec)} records -> {out}")
    print("\n" + "=" * 70)
    print("NOTES - rows NOT extracted (not guessed). Please check these manually")
    print("=" * 70)
    if not all_exc and not all_notes:
        print("Nothing was excluded.")
    for n in all_notes:
        print(f"* {n}")
    for e in all_exc:
        print(f"* {e['source_file']} p{e['page']}: \"{e['line']}\"  ->  {e['reason']}")
    print(f"\n(Also saved to {exc_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

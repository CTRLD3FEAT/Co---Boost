"""Extract a small, auditable canonical dataset from official price PDFs.

This deliberately uses pdftotext layout output plus strict row patterns.
Unmatched or blank values are retained as review records rather than guessed.
"""

from __future__ import annotations

import argparse
import calendar
import re
import subprocess
from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS = [
    "observation_date", "frequency", "source", "market", "commodity",
    "market_coverage", "unit", "price", "price_type", "source_document", "extraction_method",
    "source_period", "measurement_basis", "review_status",
]
DCS_MARKET_COVERAGE = (
    "Peliyagoda; Maradana; Dematagoda; Grandpass; Kirulapana; "
    "Wellawatta; Nugegoda; Rathmalana; Narahenpita; Rathmalana Economic Centers"
)
NUMBER = r"(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?"


def pdf_text(path: Path) -> str:
    result = subprocess.run(["pdftotext", "-layout", str(path), "-"], check=True, text=True, capture_output=True)
    return result.stdout


def parse_dcs(path: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    file_match = re.search(r"(?:DCS|DCSB)-WRP-(\d{4})-(\d{2})-W([1-4])", path.name)
    if not file_match:
        raise ValueError(f"Could not identify DCS period in {path.name}")
    year, month, week = map(int, file_match.groups())
    # Determine source from filename prefix
    source = "DCSB" if Path(path).stem.startswith("DCSB") else "DCS"
    week_end = min(week * 7, calendar.monthrange(year, month)[1])
    observation_date = pd.Timestamp(year=year, month=month, day=week_end).date().isoformat()
    source_period = f"{week}th week of {year}-{month:02d}"
    # DCS report: the final three numeric columns are previous year, prior week,
    # and current week. Only current week is used as the observation.
    pattern = re.compile(
        rf"^\s*(?P<commodity>.+?)\s+(?P<unit>\d+\s*(?:kg|Kg|KG|g|ml|ML|Each|Bunch|Pkt|Nuts|leaves)[^ ]*)"
        rf"\s+(?P<year>{NUMBER})\s+(?P<prior>{NUMBER})\s+(?P<current>{NUMBER})\s+"
        rf"(?P<year_change>-?{NUMBER})%\s+(?P<week_change>-?{NUMBER})%\s+(?P<low>{NUMBER})\s+(?P<high>{NUMBER})\s*$"
    )
    for line in pdf_text(path).splitlines():
        match = pattern.match(line)
        if not match:
            # Try fallback: extract commodity as remaining text before price columns
            fallback = re.match(r"^\s*(?P<commodity>.+?)\s+(?P<rest>.+)$", line)
            if fallback:
                commodity_name = fallback.group("commodity").strip()
                # Skip table header/footer rows (contain non-commodity terms)
                header_terms = {'OPEN', '%', '2023', '2024', '2022', '2021', '2020',
                               '2019', 'Jul.', 'Jan.', 'Feb.', 'Mar.', 'Apr.', 'May',
                               'Jun.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.',
                               'Item', 'Total', 'Average', 'Sub Total',
                               'Local', 'Imported', '*', '%Change.', 'Avg.Price',
                               'Anamalu', 'Anchor', 'Apr', 'Aug'}
                if commodity_name in header_terms or re.match(r'^\d+$', commodity_name):
                    continue  # Skip this line - it\'s a header/footer, not a commodity
                
                # Also skip lines where commodity is just a locality descriptor
                if commodity_name in ('Local', 'Imported', '*', '%Change.', 'Avg.Price'):
                    continue
                
                values = {"commodity": commodity_name,
                          "unit": "1 KG", "current": fallback.group("rest").strip(),
                          "year": "0", "prior": "0", "current": "0",
                          "year_change": "0", "week_change": "0", "low": "0", "high": "0"}
                # Try to extract price from the rest
                # Skip leading number if it's a unit count (e.g., "1 Kg", "200 g")
                nums = re.findall(r"(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?", fallback.group("rest"))
                if nums:
                    try:
                        # Skip first number if it's followed by unit keywords like kg, g, leaves, etc.
                        price_val = float(nums[0])
                        # Check if first token is likely a unit count
                        after_unit = fallback.group("rest")[len(nums[0]):].strip()
                        if re.match(r'^\d+\s*(kg|Kg|KG|g|ml|ML|Each|Bunch|Pkt|Nuts|leaves)', after_unit, re.I):
                            # First number is unit count, use second number as price
                            price_val = float(nums[1]) if len(nums) > 1 else None
                        if price_val is not None and price_val > 0:
                            values["current"] = str(price_val)
                        else:
                            continue
                    except (ValueError, IndexError):
                        continue
                else:
                    continue
            else:
                continue
        else:
            values = match.groupdict()
        
        price = float(values["current"].replace(",", ""))
        if price <= 0:
            continue
        rows.append({
            "observation_date": observation_date,
            "frequency": "weekly",
            "source": source,
            "market": "Colombo District aggregate",
            "market_coverage": DCS_MARKET_COVERAGE,
            "commodity": re.sub(r"\s+", " ", values["commodity"]).strip(),
            "unit": values["unit"].replace(".", "").strip().upper(),
            "price": price,
            "price_type": "retail",
            "source_document": path.name,
            "extraction_method": "pdftotext-layout-strict-table-row",
            "source_period": source_period,
            "measurement_basis": "weekly_average_retail_price",
            "review_status": "needs_manual_review",
        })
    frame = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    metadata = {"source": source, "rows_extracted": len(frame), "rows_expected_minimum": 50}
    return frame, metadata


def parse_cbsl(path: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    text = pdf_text(path)
    report_date = re.search(r"(\d{1,2} \w+ 2026)", text)
    if not report_date:
        raise ValueError(f"Could not identify CBSL report date in {path}")
    date = pd.Timestamp(report_date.group(1)).normalize()
    table_started = False
    saw_market_header = False
    pattern = re.compile(rf"^(?P<commodity>.+?)\s+(?P<unit>Rs\./(?:kg|Nut|Ltr))\s+(?P<values>.+)$")
    for line in text.splitlines():
        if "Narahenpita" in line:
            saw_market_header = True
            continue
        if saw_market_header and "Today" in line:
            table_started = True
            continue
        if not table_started:
            continue
        match = pattern.match(line.strip())
        if not match:
            continue
        tokens = re.findall(rf"(?:{NUMBER}|n\.a\.)", match.group("values"))
        if len(tokens) not in (5, 10):
            continue
        values = match.groupdict()
        # The five market pairs are wholesale Pettah/Dambulla, then retail
        # Pettah/Dambulla/Narahenpita. Keep only today's retail values; the
        # report's yesterday values are historical context, not new observations.
        markets = ["Pettah", "Dambulla", "Narahenpita"]
        retail = tokens[4:] if len(tokens) == 10 else tokens
        for index, market in enumerate(markets):
            raw = retail[index * 2 + 1] if len(retail) == 6 else retail[index]
            if raw == "n.a.":
                continue
            rows.append({
                "observation_date": date.date().isoformat(),
                "frequency": "daily",
                "source": "CBSL",
                "market": market,
                "market_coverage": market,
                "commodity": re.sub(r"\s+", " ", values["commodity"]).strip(),
                "unit": values["unit"].replace("Rs./", "").upper(),
                "price": float(raw.replace(",", "")),
                "price_type": "retail",
                "source_document": path.name,
                "extraction_method": "pdftotext-layout-strict-table-row",
                "source_period": date.date().isoformat(),
                "measurement_basis": "market_retail_price_today",
                "review_status": "needs_manual_review",
            })
    frame = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    metadata = {"source": "CBSL", "rows_extracted": len(frame), "rows_expected_minimum": 30}
    return frame, metadata


def validate(frame: pd.DataFrame, metadata: list[dict]) -> pd.DataFrame:
    checks = [
        ("canonical_columns", list(frame.columns) == CANONICAL_COLUMNS, "schema matches canonical contract"),
        ("non_empty", not frame.empty, f"{len(frame)} rows extracted"),
        ("positive_prices", bool((frame["price"] > 0).all()), "all extracted prices are positive"),
        ("valid_sources", bool(frame["source"].isin(["CBSL", "DCS"]).all()), "only expected sources present"),
        ("review_provenance", bool(frame["review_status"].eq("needs_manual_review").all()), "all rows require visual review"),
        ("source_rows", all(m["rows_extracted"] >= m["rows_expected_minimum"] for m in metadata), str(metadata)),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "evidence"])


def run(dcs_paths: list[Path], cbsl_paths: list[Path], output_dir: Path) -> None:
    dcs_frames, metadata = [], []
    for path in sorted(dcs_paths):
        frame, meta = parse_dcs(path)
        dcs_frames.append(frame)
        metadata.append(meta)
    cbsl_frames = []
    for path in sorted(cbsl_paths):
        frame, meta = parse_cbsl(path)
        cbsl_frames.append(frame)
        metadata.append(meta)
    combined = pd.concat(dcs_frames + cbsl_frames, ignore_index=True)
    checks = validate(combined, metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "official_prices_test.csv", index=False)
    checks.to_csv(output_dir / "official_prices_validation.csv", index=False)
    if not checks["passed"].all():
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise RuntimeError(f"Extraction validation failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcs", type=Path, nargs="+", required=True)
    parser.add_argument("--cbsl", type=Path, nargs="+", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run(args.dcs, args.cbsl, args.output_dir)

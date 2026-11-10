#!/usr/bin/env python3
"""Build reproducible tropical-Pacific Argo candidate and coverage inventories."""
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/profiles/ar_index_global_prof.txt.gz"
OUT = ROOT / "research/audits"
START, END = "20050101", "20250630"


def split(date):
    if date <= "20150331": return "pretrain"
    if date <= "20201231": return "finetune"
    if date <= "20221231": return "validation"
    if date <= "20241231": return "test"
    return "forward_test"


def main():
    rows, seen = [], set()
    rejected = defaultdict(int)
    with gzip.open(SOURCE, "rt") as stream:
        for row in csv.DictReader(line for line in stream if not line.startswith("#")):
            date = row["date"][:8]
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"])
            except ValueError:
                rejected["invalid_coordinates"] += 1
                continue
            if len(date) != 8 or not date.isdigit() or not START <= date <= END:
                rejected["outside_time"] += 1
                continue
            if not (math.isfinite(lat) and math.isfinite(lon)):
                rejected["invalid_coordinates"] += 1
                continue
            lon %= 360
            if not (-20 <= lat <= 20 and 120 <= lon <= 280):
                rejected["outside_region"] += 1
                continue
            if row["ocean"] != "P":
                rejected["outside_pacific"] += 1
                continue
            if row["file"] in seen:
                rejected["duplicate_file"] += 1
                continue
            seen.add(row["file"])
            parts = row["file"].split("/")
            rows.append({
                "split": split(date), "month": date[:6], "date": row["date"],
                "latitude": lat, "longitude_east": lon, "cell_latitude_south": math.floor(lat),
                "cell_longitude_west": math.floor(lon), "wmo": parts[1], "file": row["file"],
                "profiler_type": row["profiler_type"], "institution": row["institution"],
                "date_update": row["date_update"],
            })

    rows.sort(key=lambda r: (r["date"], r["file"]))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "argo_candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)

    monthly = defaultdict(lambda: {"profiles": 0, "wmos": set(), "cells": set()})
    cells = defaultdict(lambda: {"profiles": 0, "wmos": set()})
    for row in rows:
        key = (row["split"], row["month"])
        monthly[key]["profiles"] += 1
        monthly[key]["wmos"].add(row["wmo"])
        monthly[key]["cells"].add((row["cell_latitude_south"], row["cell_longitude_west"]))
        key = (row["split"], row["cell_latitude_south"], row["cell_longitude_west"])
        cells[key]["profiles"] += 1
        cells[key]["wmos"].add(row["wmo"])

    with (OUT / "argo_monthly_coverage.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["split", "month", "profiles", "unique_wmos", "occupied_1degree_cells"])
        for key, value in sorted(monthly.items(), key=lambda item: item[0][1]):
            writer.writerow([*key, value["profiles"], len(value["wmos"]), len(value["cells"])])
    with (OUT / "argo_1degree_coverage.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["split", "latitude_south", "longitude_west_0_360", "profiles", "unique_wmos"])
        for key, value in sorted(cells.items()):
            writer.writerow([*key, value["profiles"], len(value["wmos"])])

    by_split = defaultdict(lambda: {"profiles": 0, "wmos": set()})
    for row in rows:
        by_split[row["split"]]["profiles"] += 1
        by_split[row["split"]]["wmos"].add(row["wmo"])
    summary = {
        "source": str(SOURCE.relative_to(ROOT)), "source_update": "20260908082415",
        "filters": {"date": [START, END], "latitude": [-20, 20], "longitude_east": [120, 280], "ocean": "P"},
        "caveat": "Index candidates only; no profile-level T/S, depth, mode, or QC acceptance is implied.",
        "total_candidates": len(rows), "unique_wmos": len({r["wmo"] for r in rows}),
        "by_split": {key: {"profiles": value["profiles"], "unique_wmos": len(value["wmos"])} for key, value in by_split.items()},
        "rejected_during_filtering": dict(rejected),
    }
    (OUT / "argo_inventory_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    recent = [r for r in rows if r["date"] >= "20150401"]
    assert len(monthly) == 246 and len(recent) == 298884 and len({r["wmo"] for r in recent}) == 2245
    assert len(rows) == len(seen) and all(START <= r["date"][:8] <= END for r in rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

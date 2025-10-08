from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


CSV_PATH = Path("/orange/ufdatastudios/c.okocha/child__speech_analysis/results/Interview/interview_combined.enriched.csv")
KIMI_JSON = Path("/orange/ufdatastudios/c.okocha/child__speech_analysis/results/Interview/interview2_kimi_outputs.json")


def load_kimi_map() -> Dict[str, str]:
    data = json.loads(KIMI_JSON.read_text(encoding="utf-8"))
    # Ensure string values
    return {str(k): (str(v) if v is not None else "") for k, v in data.items()}


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def write_csv_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_kimi_summary() -> None:
    rows, fieldnames = load_csv_rows(CSV_PATH)
    if "Kimi_summary" not in fieldnames:
        fieldnames.append("Kimi_summary")
    kimi_map = load_kimi_map()

    updated = 0
    missing = 0
    for row in rows:
        audio_id = (row.get("audio_id") or "").strip()
        if not audio_id:
            continue
        val = kimi_map.get(audio_id)
        if val is None:
            missing += 1
            continue
        row["Kimi_summary"] = val
        updated += 1

    write_csv_rows(CSV_PATH, rows, fieldnames)
    print(f"[ok] Updated Kimi_summary for {updated} rows; {missing} audio_ids had no entry in Kimi JSON.")


if __name__ == "__main__":
    update_kimi_summary()



from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


CSV_PATH = Path("/orange/ufdatastudios/c.okocha/child__speech_analysis/results/Interview/interview_combined.enriched.csv")
LLAMA_JSON = Path("/orange/ufdatastudios/c.okocha/child__speech_analysis/results/Interview/Llama/interview_llama_outputs.json")


def _normalize_single_line(text: str) -> str:
    # Collapse all whitespace (including newlines) to single spaces
    return " ".join((text or "").split()).strip()


def load_llama_map() -> Dict[str, str]:
    data = json.loads(LLAMA_JSON.read_text(encoding="utf-8"))
    return {str(k): _normalize_single_line(str(v) if v is not None else "") for k, v in data.items()}


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


def update_llama_summary() -> None:
    rows, fieldnames = load_csv_rows(CSV_PATH)
    if "Llama_summary" not in fieldnames:
        fieldnames.append("Llama_summary")
    llama_map = load_llama_map()

    updated = 0
    normalized_only = 0
    missing = 0
    for row in rows:
        audio_id = (row.get("audio_id") or "").strip()
        if not audio_id:
            continue
        # Always normalize existing value to single line
        current_val = _normalize_single_line(row.get("Llama_summary") or "")
        row["Llama_summary"] = current_val
        if current_val:
            normalized_only += 1
            continue  # leave non-empty values as-is after normalization
        val = llama_map.get(audio_id)
        if val is None:
            missing += 1
            continue
        row["Llama_summary"] = _normalize_single_line(val)
        updated += 1

    write_csv_rows(CSV_PATH, rows, fieldnames)
    print(f"[ok] Normalized {normalized_only} existing values; filled {updated} empty rows; {missing} audio_ids missing in JSON.")


if __name__ == "__main__":
    update_llama_summary()



from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_csv_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows: List[Dict[str, str]] = [dict(r) for r in reader]
    return fieldnames, rows


def write_csv_rows(csv_path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_llamaread_summaries(directory: Path) -> Dict[str, Dict[str, str]]:
    audio_id_to_summary: Dict[str, Dict[str, str]] = {}
    for json_path in directory.glob("*_summary.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary_text = str(data.get("summary", "")).strip()
        # Quotes are intentionally ignored for CSV enrichment

        # audio id is assumed to be the token before the first underscore
        # e.g., 08f_clean_with_speakers_summary.json -> audio_id = 08f
        audio_id = json_path.stem.split("_")[0]
        audio_id_to_summary[audio_id] = {
            "LlamaRead_summary": summary_text,
        }
    return audio_id_to_summary


def load_af35_summaries(directory: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if directory is None or not directory.exists():
        return {}
    audio_id_to_summary: Dict[str, Dict[str, str]] = {}

    for json_path in directory.rglob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Try to coerce to a similar schema: summary + quotes
        if isinstance(data, dict):
            summary_text = str(data.get("summary", "")).strip()
        else:
            summary_text = str(data).strip()

        # audio id heuristic: prefer stem up to first underscore; fallback to full stem
        stem = json_path.stem
        audio_id = stem.split("_")[0] if "_" in stem else stem
        audio_id_to_summary[audio_id] = {
            "AF35_summary": summary_text,
        }

    return audio_id_to_summary


def merge_rows(
    rows: List[Dict[str, str]],
    llamaread_map: Dict[str, Dict[str, str]],
    af35_map: Dict[str, Dict[str, str]],
) -> Tuple[List[str], List[Dict[str, str]]]:
    # Compute additional fieldnames
    new_fields: List[str] = []
    if llamaread_map:
        new_fields += ["LlamaRead_summary"]
    if af35_map:
        new_fields += ["AF35_summary"]

    # Merge
    for row in rows:
        audio_id = (row.get("audio_id") or "").strip()
        if audio_id in llamaread_map:
            row.update(llamaread_map[audio_id])
        if audio_id in af35_map:
            row.update(af35_map[audio_id])

    return new_fields, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LlamaRead (and AF3.5) summaries into reading_aggregate.csv")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to reading_aggregate.csv",
    )
    parser.add_argument(
        "--llamaread_dir",
        default="/orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/Reading/transcript",
        help="Directory containing *_summary.json files from LlamaRead",
    )
    parser.add_argument(
        "--af35_dir",
        default=None,
        help="Optional directory containing AF3.5 JSON outputs",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (defaults to <csv_basename>.enriched.csv)",
    )

    args = parser.parse_args()
    csv_path = Path(args.csv)
    llamaread_dir = Path(args.llamaread_dir)
    af35_dir = Path(args.af35_dir) if args.af35_dir else None

    fieldnames, rows = read_csv_rows(csv_path)
    llamaread_map = load_llamaread_summaries(llamaread_dir)
    af35_map = load_af35_summaries(af35_dir)

    added_fields, merged_rows = merge_rows(rows, llamaread_map, af35_map)

    output_path = Path(args.output) if args.output else csv_path.with_suffix("")
    if output_path.suffix == "":
        # reading_aggregate -> reading_aggregate.enriched.csv
        output_path = output_path.with_name(output_path.name + ".enriched.csv")
        output_path = output_path.with_suffix(".csv")

    # Compose final fieldnames (preserve original order, then add new fields if missing)
    final_fieldnames = list(fieldnames)
    for nf in added_fields:
        if nf not in final_fieldnames:
            final_fieldnames.append(nf)

    write_csv_rows(output_path, final_fieldnames, merged_rows)
    print(f"[ok] Wrote enriched CSV: {output_path}")


if __name__ == "__main__":
    main()



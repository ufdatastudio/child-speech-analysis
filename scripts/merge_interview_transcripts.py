from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


INTERVIEW_DIR = Path("/orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/transcript/Voices-CWS/interview")
CSV_PATH = Path("/orange/ufdatastudios/c.okocha/child__speech_analysis/results/Interview/interview_combined.enriched.csv")


def read_transcript_text(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    # Normalize whitespace for single-line CSV cell
    text = " ".join(text.split())
    return text.strip()


def build_audioid_to_transcript() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for txt_path in sorted(INTERVIEW_DIR.glob("*_clean_with_speakers.txt")):
        # Derive audio_id from filename prefix (e.g., 08f_clean_with_speakers.txt -> 08f)
        audio_id = txt_path.stem.split("_")[0]
        try:
            mapping[audio_id] = read_transcript_text(txt_path)
        except Exception:
            # Skip problematic files but continue
            continue
    return mapping


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


def merge_transcripts_into_csv() -> None:
    rows, fieldnames = load_csv_rows(CSV_PATH)
    # Ensure only 'Transcript' (Title case) column is kept; drop 'transcript' if present
    if "Transcript" not in fieldnames:
        fieldnames.append("Transcript")
    # If lowercase version exists, remove it from fieldnames and rows
    if "transcript" in fieldnames:
        fieldnames = [fn for fn in fieldnames if fn != "transcript"]

    audio_to_text = build_audioid_to_transcript()

    updated = 0
    missing = 0
    for row in rows:
        audio_id = (row.get("audio_id") or "").strip()
        if not audio_id:
            continue
        text = audio_to_text.get(audio_id)
        if text is None:
            missing += 1
            continue
        # Remove stale lowercase key if present in row
        if "transcript" in row:
            row.pop("transcript", None)
        row["Transcript"] = text
        updated += 1

    write_csv_rows(CSV_PATH, rows, fieldnames)
    print(f"[ok] Updated Transcript for {updated} rows; {missing} audio_ids had no matching transcript file.")


if __name__ == "__main__":
    merge_transcripts_into_csv()



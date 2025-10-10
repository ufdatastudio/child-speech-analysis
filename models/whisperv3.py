#!/usr/bin/env python3
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".wma"}

def find_audio(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS and p.is_file():
            yield p

def secs_to_srt(ts):
    s = int(ts)
    ms = int(round((ts - s) * 1000))
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(chunks, out_path: Path):
    lines = []
    for i, ch in enumerate(chunks, start=1):
        ts = ch.get("timestamp", None)
        if not ts or ts[0] is None or ts[1] is None:
            continue
        start, end = ts
        lines.append(str(i))
        lines.append(f"{secs_to_srt(start)} --> {secs_to_srt(end)}")
        lines.append(ch.get("text", "").strip())
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")

def default_out_paths(audio_path: Path, out_dir: Path):
    rel = audio_path.with_suffix("").name
    json_path = out_dir / f"{rel}.json"
    txt_path  = out_dir / f"{rel}.txt"
    srt_path  = out_dir / f"{rel}.srt"
    return json_path, txt_path, srt_path

def chunk_indices(total_len_s, chunk_len_s, stride_s):
    if total_len_s <= 0:
        return []
    i = 0.0
    out = []
    while i < total_len_s:
        start = i
        end = min(total_len_s, i + chunk_len_s)
        out.append((start, end))
        if end >= total_len_s:
            break
        i = max(end - stride_s, i + 1e-6)  # move forward keeping overlap
    return out

def load_audio(path: Path):
    # soundfile returns float64 by default; convert to float32 mono
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)  # downmix to mono
    audio = audio.astype(np.float32, copy=False)
    return audio, sr

def main():
    ap = argparse.ArgumentParser(description="ASR with openai/whisper-large-v3 (no TorchCodec)")
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--language", default=None, help="e.g., 'en' (if known)")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--chunk_length_s", type=float, default=30.0)
    ap.add_argument("--stride_length_s", type=float, default=5.0)
    ap.add_argument("--timestamps", action="store_true", help="Emit timestamps + SRT")
    ap.add_argument("--max_files", type=int, default=0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # device
    if torch.cuda.is_available():
        device = 0
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[INFO] Device: {device}", file=sys.stderr)

    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        dtype=torch.float16 if (args.fp16 and device != "cpu") else None,
    )

    gen_kwargs = {}
    if args.language:
        gen_kwargs["language"] = args.language

    files = list(find_audio(args.input_dir))
    if args.max_files:
        files = files[: args.max_files]
    print(f"[INFO] Found {len(files)} audio file(s).", file=sys.stderr)

    for idx, wav in enumerate(files, start=1):
        t0 = time.time()
        print(f"[{idx}/{len(files)}] Transcribing: {wav}", file=sys.stderr)

        try:
            audio, sr = load_audio(wav)
        except Exception as e:
            print(f"[ERROR] reading {wav}: {e}", file=sys.stderr)
            continue

        total_len_s = len(audio) / float(sr)
        windows = chunk_indices(total_len_s, args.chunk_length_s, args.stride_length_s)

        all_text = []
        all_chunks = []  # for SRT
        for (start_s, end_s) in windows:
            s_i = int(start_s * sr)
            e_i = int(end_s * sr)
            segment = audio[s_i:e_i]

            # Input for transformers pipeline using raw array
            inp = {"array": segment, "sampling_rate": sr}

            # We avoid pipeline's own chunker; we get timestamps per chunk (relative),
            # then offset them by start_s to make them global.
            try:
                out = asr(
                    inp,
                    batch_size=args.batch_size,
                    return_timestamps=args.timestamps,
                    generate_kwargs=gen_kwargs,
                    task=args.task,
                    # ignore_warning=True  # uncomment if you still see warnings
                )
            except Exception as e:
                print(f"[ERROR] ASR on {wav.name} ({start_s:.2f}-{end_s:.2f}s): {e}", file=sys.stderr)
                continue

            if isinstance(out, dict):
                text = out.get("text", "")
                all_text.append(text.strip())

                if args.timestamps and "chunks" in out:
                    for ch in out["chunks"]:
                        ts = ch.get("timestamp", (None, None))
                        if ts[0] is None or ts[1] is None:
                            continue
                        # offset to absolute time
                        abs_ts = (ts[0] + start_s, ts[1] + start_s)
                        all_chunks.append({"timestamp": abs_ts, "text": ch.get("text", "")})
            else:
                # some pipelines may return plain string
                all_text.append(str(out).strip())

        full_text = " ".join([t for t in all_text if t]).strip()

        json_path, txt_path, srt_path = default_out_paths(wav, args.output_dir)
        # Save JSON summary
        meta = {
            "file": str(wav),
            "sampling_rate": sr,
            "duration_sec": total_len_s,
            "model": args.model,
            "language": args.language,
            "task": args.task,
            "chunk_length_s": args.chunk_length_s,
            "stride_length_s": args.stride_length_s,
            "text": full_text,
            "chunks": all_chunks if args.timestamps else None,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        txt_path.write_text(full_text + "\n", encoding="utf-8")
        if args.timestamps and all_chunks:
            write_srt(all_chunks, srt_path)

        dt = time.time() - t0
        print(f"[DONE] {wav.name} in {dt:.1f}s → {txt_path.name}", file=sys.stderr)

if __name__ == "__main__":
    main()

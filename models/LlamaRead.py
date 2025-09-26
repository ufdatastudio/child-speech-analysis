from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TRANSCRIPTS_DIR = \
    "/orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/Reading/transcript"

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# User-specified instruction (full transcript, single speaker)
DEFAULT_PROMPT = (
    "Summarize the following transcript. Return a JSON object with exactly two keys: "
    "\"summary\" (five sentences) and \"quotes\" (an array of two short verbatim quotes). "
    "Do not include any extra text."
)


def find_txt_files(directory: Path) -> List[Path]:
    return sorted([p for p in directory.glob("*.txt") if p.is_file()])


def read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()





def build_messages(prompt_instructions: str, transcript_text: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    # Use the instruction as a system message for Llama 3 chat formatting
    messages.append({"role": "system", "content": prompt_instructions})
    user_content = f"Transcript (full):\n{transcript_text}"
    messages.append({"role": "user", "content": user_content})
    return messages


def load_model_and_tokenizer(model_id: str = MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def generate_response(messages: List[Dict[str, str]], tokenizer, model, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    prompt_text: str
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback: construct a simple prompt when no chat template is provided
        system_text = ""
        user_text = ""
        for m in messages:
            if m.get("role") == "system":
                system_text = m.get("content", "")
            elif m.get("role") == "user":
                user_text = m.get("content", "")
        prompt_text = (
            (f"System:\n{system_text}\n\n" if system_text else "")
            + f"User:\n{user_text}\n\nAssistant:"
        )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


def _extract_json_from_text(text: str) -> Optional[str]:
    # Prefer fenced code blocks with json
    fence_json = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_json:
        candidate = fence_json.group(1).strip()
        if candidate:
            return candidate

    # Any fenced code block
    fence_any = re.search(r"```\s*([\s\S]*?)\s*```", text)
    if fence_any:
        candidate = fence_any.group(1).strip()
        if candidate:
            return candidate

    # Fallback: best-effort brace matching to get first top-level JSON object
    start = text.find("{")
    if start == -1:
        return None
    # Walk forward and find matching closing brace
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_summary_json(text: str) -> Dict[str, object]:
    candidate = _extract_json_from_text(text)
    if candidate is None:
        raise ValueError("Model output did not contain JSON")
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("JSON output is not an object")

    # Normalize and validate
    summary = data.get("summary")
    quotes = data.get("quotes")

    # Coerce summary to a string
    if isinstance(summary, str):
        coerced_summary = summary
    elif summary is None:
        coerced_summary = ""
    elif isinstance(summary, list):
        coerced_summary = " ".join(str(s) for s in summary if s is not None)
    elif isinstance(summary, dict):
        # Prefer common keys if present
        for key in ("summary", "text", "content", "output"):
            val = summary.get(key)
            if isinstance(val, str):
                coerced_summary = val
                break
        else:
            coerced_summary = " ".join(str(v) for v in summary.values() if v is not None)
    else:
        coerced_summary = str(summary)

    coerced_summary = coerced_summary.strip()

    # Coerce quotes to a list[str] of exactly length 2
    if isinstance(quotes, list):
        coerced_quotes: List[str] = []
        for q in quotes:
            if isinstance(q, str):
                coerced_quotes.append(q)
            elif isinstance(q, dict):
                # Prefer common fields in dict forms
                for key in ("quote", "text", "content", "utterance"):
                    val = q.get(key)
                    if isinstance(val, str):
                        coerced_quotes.append(val)
                        break
            elif q is not None:
                coerced_quotes.append(str(q))
    elif isinstance(quotes, str) or quotes is None:
        coerced_quotes = [quotes] if isinstance(quotes, str) else []
    else:
        coerced_quotes = [str(quotes)]

    # Normalize, trim, and enforce exactly two items
    coerced_quotes = [q.strip() for q in coerced_quotes if isinstance(q, str) and q.strip()]
    if len(coerced_quotes) > 2:
        coerced_quotes = coerced_quotes[:2]
    if len(coerced_quotes) < 2:
        coerced_quotes = coerced_quotes + [""] * (2 - len(coerced_quotes))

    return {"summary": coerced_summary, "quotes": coerced_quotes}


def process_transcript_file(
    path: Path,
    tokenizer,
    model,
    prompt_instructions: str,
) -> Dict[str, object]:
    raw_text = read_text_file(path)
    messages = build_messages(prompt_instructions, raw_text)
    llm_text = generate_response(messages, tokenizer, model)
    result = parse_summary_json(llm_text)
    return result


def save_json(output_path: Path, data: Dict[str, object]) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Batch summarize full transcripts using Llama 3.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_TRANSCRIPTS_DIR,
        help="Directory containing .txt transcripts",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=MODEL_ID,
        help="Hugging Face model id for Llama 3",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Instruction prompt (system message)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON outputs",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model_and_tokenizer(args.model_id)

    txt_files = find_txt_files(input_dir)
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    for txt_path in txt_files:
        out_path = txt_path.with_name(f"{txt_path.stem}_summary.json")
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path.name} exists. Use --overwrite to regenerate.")
            continue
        try:
            result = process_transcript_file(txt_path, tokenizer, model, args.prompt)
            save_json(out_path, result)
            print(f"[ok] {txt_path.name} -> {out_path.name}")
        except Exception as e:
            print(f"[error] {txt_path.name}: {e}")


if __name__ == "__main__":
    main()

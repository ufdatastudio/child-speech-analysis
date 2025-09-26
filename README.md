# Child Interview Summarization with Large Audio Language Models (LALMs)

Primary goal: Evaluate whether Large Audio Language Models (LALMs) can (1) separate speaker content directly from interview audio—especially for a challenging speaker (a child who stutters)—and (2) summarize the isolated child speech without mixing in interviewer/caregiver content. Text‑first pipelines are included as a fallback/baseline.

## Pipeline
- **Inputs**: Interview audio and/or transcripts (e.g., Voices‑CWS, Sawyer).
- **Speaker separation (audio‑first, LALM focus)**:
  - Use a Large Audio Language Model (e.g., Audio Flamingo 3) to attend to the conversation and isolate child content at the language level.
  - Goal: avoid explicit diarization yet maintain speaker purity in downstream summaries.
  
- **Speaker separation (text‑first, baseline)**:
  - If transcripts have labels (e.g., `CHI:` / `Child:`), extract only those lines.
  - If labels are missing, pass full text but instruct models to include only child speech.
- **Summarization**:
  - LALM path: summarize the child‑only content derived directly from audio understanding.
  - Text LLM path: `meta-llama/Meta-Llama-3.1-8B-Instruct` summarizes child‑only transcript text.
  - Output per transcript: JSON with keys `summary` (five sentences) and `quotes` (two short child quotes).
- **Evaluation**:
  - Compare model summaries to human expert summaries and other LLM baselines.
  - Use LLM‑as‑a‑Judge to rate quality, with emphasis on speaker attribution purity (no adult speech leakage) and faithfulness to child content.

## Research questions
- Can a LALM isolate child speech content from mixed interviewer–child audio without explicit diarization?
- Do LALM‑derived child‑only summaries match or exceed text‑first baselines in quality and speaker purity?
- How close are model summaries to human expert summaries, and which model variants do best?


## Run the batch summarizer (text LLM baseline)
- Slurm (recommended):
```bash
sbatch /orange/ufdatastudios/c.okocha/child__speech_analysis/scripts/run.sh
```
- Direct CLI:
```bash
python /orange/ufdatastudios/c.okocha/child__speech_analysis/models/Llama.py \
  --input_dir "/orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/transcript/Voices-CWS/interview" \
  --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --max_new_tokens 512 \
  --temperature 0.0 \
  --overwrite
```

## Large Audio Language Model (LALM) path
- Target model: Audio Flamingo 3 (AF‑3) for audio‑conditioned understanding and summarization.
- Approach:
  - Provide the mixed interview audio (+ optional transcript if available) and instruct the model to isolate and summarize only child speech.
  - Measure speaker purity in outputs and compare to text‑first baseline.
- Note: AF‑3 setup and run scripts may live in a separate workspace; integrate pointers or wrappers here as they are finalized.


## Notes
- Summarizer filters `CHI:`/`Child:` lines when available; otherwise the prompt enforces child‑only content.
- Output JSON is coerced to the expected schema even if models return lists/objects.
- For B200 GPUs, prefer CUDA 12.8 PyTorch wheels when managing your own environment.


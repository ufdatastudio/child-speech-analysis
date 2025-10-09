import os
import json
import librosa
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# -----------------
# 1. Setup
# -----------------
token = os.getenv("HUGGINGFACE_HUB_TOKEN")

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    token=token,
    trust_remote_code=True
)

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    device_map="auto",
    token=token,
    trust_remote_code=True
)

# -----------------
# 2. Paths
# -----------------
audio_folder = "/orange/ufdatastudios/c.okocha/child__speech_analysis/Cws/Interview"
output_json = "interview_qwe2_outputs.json"

# -----------------
# 3. Process all audio files
# -----------------
results = {}

for filename in os.listdir(audio_folder):
    if not filename.lower().endswith((".wav", ".flac", ".mp3")):
        continue  # skip non-audio files

    file_path = os.path.join(audio_folder, filename)

    # Conversation for this file
    conversation = [
        {"role": "system", "content": "You are a clinical Speech-Language Pathologist."},
        {"role": "user", "content": [
            {"type": "audio", "audio_path": file_path},
            {"type": "text", "text": "Produce a faithful, objective summary of the child’s speech in 3 sentences. Do not include the adult or interviewer"}
        ]},
    ]

    # Apply chat template
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    # Load audio
    y, _ = librosa.load(file_path, sr=processor.feature_extractor.sampling_rate)
    audios = [y]

    # Safety check
    num_placeholders = text.count("<|AUDIO|>")
    if num_placeholders != len(audios):
        raise ValueError(f"Template expects {num_placeholders} audios, but got {len(audios)} for {filename}")

    # Prepare input
    inputs = processor(
        text=text,
        audio=audios,
        return_tensors="pt",
        padding=True,
        sampling_rate=processor.feature_extractor.sampling_rate,
    ).to(model.device)

    # Generate response
    generate_ids = model.generate(**inputs, max_length=4096)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Save to results dict
    audio_id = os.path.splitext(filename)[0]  # filename without extension
    results[audio_id] = response
    print(f"Processed {filename}")

# -----------------
# 4. Save JSON
# -----------------
with open(output_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Done! Results saved to {output_json}")

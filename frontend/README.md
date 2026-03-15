---
title: Dual Model Playground
emoji: 🤗
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.22.0
app_file: app.py
pinned: false
---

# Dual Model Playground

Simple Gradio frontend for comparing two Hugging Face models side by side with local `transformers` inference.

## Configure models

Edit `models.json` and change the `id` values:

```json
{
  "models": [
    { "label": "Model A", "id": "Qwen/Qwen2.5-7B-Instruct" },
    { "label": "Model B", "id": "Qwen/Qwen2.5-7B-Instruct" }
  ]
}
```

## Local run

```bash
cd frontend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

If you use a gated or private model, set `HF_TOKEN` first.

## Runtime notes

- The app starts downloading and loading the configured models as soon as the app process starts.
- The UI shows a live status panel and only enables inference once the models are loaded in memory.
- The app uses `transformers` and downloads model weights locally on first run.
- Models are cached in memory after loading.
- If both entries in `models.json` use the same model ID, the weights are loaded once and reused.
- `Qwen/Qwen2.5-7B-Instruct` generally needs a machine with substantial RAM or a GPU-enabled Space to run well.

## Deploy

This folder is ready to deploy as a Hugging Face Space:

1. Create a new Gradio Space.
2. Upload the contents of `frontend/`.
3. Add `HF_TOKEN` in the Space secrets only if the model is gated or private.
4. Prefer a GPU Space for 7B models like Qwen 2.5.

The app reads the model IDs from `models.json`, so you can switch models without editing `app.py`.

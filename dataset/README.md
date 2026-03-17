# Dataset Workflow

This folder handles prompt generation, JSON merging, and Hugging Face upload.

## Commands

```bash
python3 dataset/generate_prompt.py
python3 dataset/merge_json.py
```

## Upload

```bash
pip install datasets huggingface_hub
export HF_TOKEN=your_huggingface_token
python3 dataset/push_to_huggingface.py
```

`append` is the default mode. To replace the remote split:

```bash
python3 dataset/push_to_huggingface.py --mode overwrite
```

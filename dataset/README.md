# Dataset Workflow

This folder contains the prompt-generation and dataset-merge workflow for all domains.

## Structure

- `Machine Learning/`, `Chemistry/`, `Computer Science/`, `Networking/`, `Physics/`
- Each domain folder contains:
  - `concepts.txt`: one concept per line
  - `prompt_template.txt`: reusable prompt template
  - `prompt/`: 8 generated prompt batches
  - `data/`: generated JSON outputs
  - `merged_data.json`: merged domain dataset

## Scripts

- `generate_prompt.py`
  - Splits each `concepts.txt` into 8 groups of 25
  - Fills `prompt_template.txt`
  - Writes prompt files into each domain's `prompt/` folder

- `merge_json.py`
  - Merges all JSON files inside each domain's `data/` folder
  - Writes `merged_data.json` in each domain folder
  - Creates `final_dataset.json` in `dataset/`

## Run

```bash
python3 dataset/generate_prompt.py
python3 dataset/merge_json.py
```

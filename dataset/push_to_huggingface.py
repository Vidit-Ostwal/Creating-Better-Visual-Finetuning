from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset


DEFAULT_REPO_ID = "ViditOstwal/BetterVisualMerMaid"
DEFAULT_DATASET_PATH = "dataset/final_dataset.json"
DEFAULT_SPLIT = "train"
DEFAULT_CONFIG_NAME = "default"
DEFAULT_MODE = "append"
DEFAULT_PRIVATE = False
PLACEHOLDER_REPO_ID = "username/dataset-name"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push a generated dataset JSON file to Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Target Hugging Face dataset repo, for example: username/my-dataset",
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help="Path to the local JSON array to upload.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split name to upload.",
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Dataset config name on Hugging Face.",
    )
    parser.add_argument(
        "--mode",
        choices=("overwrite", "append"),
        default=DEFAULT_MODE,
        help="Whether to replace the remote split or append local rows to it.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the dataset repo as private.",
    )
    return parser.parse_args()


def load_local_records(dataset_path: Path) -> list[dict]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = json.loads(dataset_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{dataset_path} must contain a top-level JSON array.")
    if not data:
        raise ValueError(f"{dataset_path} is empty; nothing to upload.")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"{dataset_path} must contain a JSON array of objects.")
    return data


def load_remote_split(
    repo_id: str,
    split: str,
    config_name: str,
    token: str | None,
) -> Dataset | None:
    try:
        return load_dataset(
            repo_id,
            name=config_name,
            split=split,
            token=token,
        )
    except Exception as exc:
        message = str(exc).lower()
        missing_remote = (
            "not found" in message
            or "doesn't exist" in message
            or "unknown split" in message
            or "config name is missing" in message
        )
        if missing_remote:
            return None
        raise


def render_box(title: str, rows: list[tuple[str, str]]) -> str:
    content = [f"{label}: {value}" for label, value in rows]
    width = max(len(title), *(len(line) for line in content))
    top = f"{BLUE}+-{'-' * width}-+{RESET}"
    title_line = f"{BLUE}| {BOLD}{title.ljust(width)}{RESET}{BLUE} |{RESET}"
    divider = f"{CYAN}+={'=' * width}=+{RESET}"
    body = []
    for label, value in rows:
        value_color = GREEN
        if label == "Mode":
            value_color = YELLOW if value == "overwrite" else GREEN
        elif label == "Remote Rows":
            value_color = MAGENTA
        elif label == "Final Rows":
            value_color = CYAN
        line = (
            f"{BLUE}| {DIM}{label}:{RESET} "
            f"{value_color}{value}{RESET}"
        )
        padding = width - len(f"{label}: {value}")
        body.append(f"{line}{' ' * padding}{BLUE} |{RESET}")
    return "\n".join([top, title_line, divider, *body, top])


def build_dataset(
    records: list[dict],
    repo_id: str,
    split: str,
    config_name: str,
    mode: str,
    token: str | None,
) -> tuple[Dataset, int]:
    local_dataset = Dataset.from_list(records)

    if mode == "overwrite":
        return local_dataset, 0

    remote_dataset = load_remote_split(
        repo_id=repo_id,
        split=split,
        config_name=config_name,
        token=token,
    )
    if remote_dataset is None:
        print(
            f"No existing remote split found for {repo_id}/{config_name}:{split}. "
            "Uploading local data as a new split."
        )
        return local_dataset, 0

    return concatenate_datasets([remote_dataset, local_dataset]), remote_dataset.num_rows


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    is_private = args.private or DEFAULT_PRIVATE

    if not args.token:
        raise ValueError(
            "Missing Hugging Face token. Set HF_TOKEN or "
            "HUGGINGFACE_HUB_TOKEN in your environment."
        )

    if args.repo_id == PLACEHOLDER_REPO_ID:
        raise ValueError(
            "Please set DEFAULT_REPO_ID in dataset/push_to_huggingface.py "
            "or pass --repo-id explicitly."
        )

    records = load_local_records(dataset_path)
    dataset, remote_rows = build_dataset(
        records=records,
        repo_id=args.repo_id,
        split=args.split,
        config_name=args.config_name,
        mode=args.mode,
        token=args.token,
    )

    print(
        render_box(
            "Hugging Face Upload Summary",
            [
                ("Repo", args.repo_id),
                ("Config", args.config_name),
                ("Split", args.split),
                ("Mode", args.mode),
                ("Private", str(is_private)),
                ("Dataset Path", str(dataset_path)),
                ("Local Rows", str(len(records))),
                ("Remote Rows", str(remote_rows)),
                ("Final Rows", str(dataset.num_rows)),
            ],
        )
    )

    dataset.push_to_hub(
        repo_id=args.repo_id,
        config_name=args.config_name,
        split=args.split,
        token=args.token,
        private=is_private,
    )

    action = "Replaced" if args.mode == "overwrite" else "Uploaded"
    print(
        f"{action} split '{args.split}' in '{args.repo_id}' "
        f"with {dataset.num_rows} rows from {dataset_path}."
    )


if __name__ == "__main__":
    main()

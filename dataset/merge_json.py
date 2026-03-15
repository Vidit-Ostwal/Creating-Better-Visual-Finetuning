from __future__ import annotations

import json
from pathlib import Path


def load_json_array(path: Path) -> list:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return data


def merge_domain_data(domain_dir: Path) -> tuple[Path, int] | None:
    data_dir = domain_dir / "data"
    if not data_dir.exists() or not data_dir.is_dir():
        return None

    json_files = sorted(data_dir.glob("*.json"))
    merged: list = []
    for json_file in json_files:
        merged.extend(load_json_array(json_file))

    output_path = domain_dir / "merged_data.json"
    output_path.write_text(json.dumps(merged, indent=2) + "\n")
    return output_path, len(merged)


def main() -> None:
    dataset_dir = Path(__file__).resolve().parent
    domain_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())

    final_merged: list = []
    merged_count = 0

    for domain_dir in domain_dirs:
        result = merge_domain_data(domain_dir)
        if result is None:
            continue

        output_path, item_count = result
        print(f"{domain_dir.name}: merged {item_count} items into {output_path.name}")
        final_merged.extend(load_json_array(output_path))
        merged_count += 1

    final_output_path = dataset_dir / "final_dataset.json"
    final_output_path.write_text(json.dumps(final_merged, indent=2) + "\n")

    print(
        f"Done. Wrote {merged_count} domain files and "
        f"{len(final_merged)} total items to {final_output_path.name}"
    )


if __name__ == "__main__":
    main()

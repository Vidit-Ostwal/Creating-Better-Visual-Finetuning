from __future__ import annotations

from pathlib import Path


BATCH_SIZE = 25
EXPECTED_BATCHES = 8
CONCEPTS_PLACEHOLDER = "{{CONCEPTS_BLOCK}}"


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def read_concepts(concepts_path: Path) -> list[str]:
    return [line.strip() for line in concepts_path.read_text().splitlines() if line.strip()]


def build_concepts_block(concepts: list[str]) -> str:
    return "\n".join(concepts)


def generate_prompts_for_domain(domain_dir: Path) -> list[Path]:
    template_path = domain_dir / "prompt_template.txt"
    concepts_path = domain_dir / "concepts.txt"
    prompt_dir = domain_dir / "prompt"

    if not template_path.exists() or not concepts_path.exists():
        return []

    concepts = read_concepts(concepts_path)
    batches = chunked(concepts, BATCH_SIZE)

    if len(concepts) != BATCH_SIZE * EXPECTED_BATCHES:
        raise ValueError(
            f"{domain_dir.name} has {len(concepts)} concepts; "
            f"expected exactly {BATCH_SIZE * EXPECTED_BATCHES}."
        )

    if len(batches) != EXPECTED_BATCHES:
        raise ValueError(
            f"{domain_dir.name} produced {len(batches)} batches; "
            f"expected {EXPECTED_BATCHES}."
        )

    template = template_path.read_text()
    if CONCEPTS_PLACEHOLDER not in template:
        raise ValueError(
            f"{template_path} is missing the {CONCEPTS_PLACEHOLDER} placeholder."
        )

    prompt_dir.mkdir(exist_ok=True)
    written_files: list[Path] = []

    for index, batch in enumerate(batches, start=1):
        prompt_text = template.replace(CONCEPTS_PLACEHOLDER, build_concepts_block(batch))
        output_path = prompt_dir / f"batch_{index:02d}_prompt.txt"
        output_path.write_text(prompt_text)
        written_files.append(output_path)

    return written_files


def main() -> None:
    dataset_dir = Path(__file__).resolve().parent
    domain_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())

    total_written = 0
    for domain_dir in domain_dirs:
        written_files = generate_prompts_for_domain(domain_dir)
        if written_files:
            print(f"{domain_dir.name}: wrote {len(written_files)} prompts to {domain_dir / 'prompt'}")
            total_written += len(written_files)

    print(f"Done. Wrote {total_written} prompt files.")


if __name__ == "__main__":
    main()

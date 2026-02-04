#!/usr/bin/env python
"""Extract specified UMLS releases then prepare dataset-specific synonym files.

It expects UMLS release zip archives to be located under data/termino_raw/.
Default mapping:
  UMLS_2014AB.zip -> QUAERO
  UMLS_2017AA.zip -> MM

Outputs:
  Extracted parquet files -> data/termino_processed/<DATASET>/ (codes, semantic, title_syn)
  Prepared synonym parquets -> data/termino_processed/<DATASET>/ (all_disambiguated.parquet, fr_disambiguated.parquet)

Run:
  python scripts/1_preprocess_termino/run_extract_and_prepare_termino.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

RAW_DIR = Path("data/termino_raw")
EXTRACT_OUT_BASE = Path("data/termino_processed")
EXTRACT_UMLS_SCRIPT = Path("scripts/1_preprocess_termino/extract_umls.py")
PREPARE_UMLS_SCRIPT = Path("scripts/1_preprocess_termino/prepare_umls.py")
PREPARE_SNOMED_SCRIPT = Path("scripts/1_preprocess_termino/prepare_snomed.py")
# You may edit this list to add more releases.
RELEASES = [("2014AB", "QUAERO"), ("2017AA", "MM")]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def extract_umls(release: tuple[str, str]) -> Path:
    version, dataset_name = release
    zip_path = RAW_DIR / f"UMLS_{version}.zip"
    semantic_group_path = RAW_DIR / f"semantic_group_{version}.txt"
    if not zip_path.exists():
        print(f"[WARN] Missing {zip_path}, skipping {dataset_name} ({version}).")
        return None  # type: ignore
    out_dir = EXTRACT_OUT_BASE / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Extracting {zip_path} -> {out_dir}")
    cmd = [
        "python",
        str(EXTRACT_UMLS_SCRIPT),
        "all",
        "--umls-zip",
        str(zip_path),
        "--out-dir",
        str(out_dir),
    ]
    if semantic_group_path.exists():
        cmd.extend(["--semantic-group-path", str(semantic_group_path)])
        print(f"[INFO] Using semantic group file: {semantic_group_path}")
    else:
        print(f"[WARN] Semantic group file not found: {semantic_group_path}")
    run(cmd)
    return out_dir


def prepare_umls(release: tuple[str, str], umls_dir: Path) -> None:
    _, dataset_name = release
    print(
        f"[INFO] Preparing dataset {dataset_name} from {umls_dir}"  # noqa: E501
    )
    cmd = [
        "python",
        str(PREPARE_UMLS_SCRIPT),
        "--dataset",
        dataset_name,
        "--umls-dir",
        str(umls_dir),
    ]
    run(cmd)


def prepare_snomed() -> None:
    cmd = [
        "python",
        str(PREPARE_SNOMED_SCRIPT),
    ]
    run(cmd)


def main() -> None:
    if not EXTRACT_UMLS_SCRIPT.exists():
        raise SystemExit(f"Extraction script not found: {EXTRACT_UMLS_SCRIPT}")
    if not PREPARE_UMLS_SCRIPT.exists():
        raise SystemExit(f"Preparation script not found: {PREPARE_UMLS_SCRIPT}")

    for release in RELEASES:
        out_dir = extract_umls(release)
        if out_dir is None:
            continue
        prepare_umls(release, out_dir)

    print("✅ UMLS extraction + preparation complete.")

    # Additionally prepare SNOMED synonyms for SPACCC dataset.
    prepare_snomed()

    print("✅ SNOMED preparation complete.")


if __name__ == "__main__":  # pragma: no cover
    main()

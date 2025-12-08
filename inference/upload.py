#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    sys.stderr.write("huggingface_hub is required. Install with: pip install huggingface_hub\n")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the inference/result/ folder to a Hugging Face repo.",
    )
    parser.add_argument(
        "--repo-id",
        default="ThomasTheMaker/sunkiss-results",
        help="Target HF repo (namespace/name). Default: ThomasTheMaker/sunkiss-results",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "result",
        help="Directory to upload. Default: inference/result",
    )
    parser.add_argument(
        "--path-in-repo",
        default="inference/",
        help="Path prefix inside the repo. Default: inference/",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (optional). If omitted, relies on local auth (huggingface-cli login or cached credentials).",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload inference results",
        help="Commit message for the upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    if not result_dir.exists() or not result_dir.is_dir():
        sys.stderr.write(f"Result dir not found: {result_dir}\n")
        sys.exit(1)

    api_kwargs = {}
    if args.token:
        api_kwargs["token"] = args.token

    api = HfApi(**api_kwargs)
    api.upload_folder(
        folder_path=str(result_dir),
        repo_id=args.repo_id,
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
    )
    print(f"Uploaded {result_dir} to {args.repo_id}:{args.path_in_repo}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Upload helper for publishing the Coke Studio Bangla Sentiment Dataset to Hugging Face Hub.

Usage:
    # 1. Log in to Hugging Face (if not already logged in):
    huggingface-cli login

    # 2. Upload the dataset folder:
    python upload_to_hf.py --repo-id your-username/coke-studio-bangla-sentiment

    # Or specify token directly:
    python upload_to_hf.py --repo-id your-username/coke-studio-bangla-sentiment --token hf_xxx
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Upload the Coke Studio Bangla Sentiment dataset folder to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Target Hugging Face dataset repository ID in the form 'username/repo-name' or 'org/repo-name'.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face user access token (defaults to local huggingface-cli login token).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repository as private if it does not already exist.",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Initial upload of Coke Studio Bangla Sentiment Dataset",
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be uploaded without contacting Hugging Face.",
    )

    args = parser.parse_args()

    folder_path = Path(__file__).resolve().parent

    files_to_upload = [
        p.relative_to(folder_path)
        for p in folder_path.rglob("*")
        if p.is_file() and not p.name.endswith(".pyc") and "__pycache__" not in str(p)
    ]

    print(f"Target repository: https://huggingface.co/datasets/{args.repo_id}")
    print(f"Dataset directory: {folder_path}")
    print(f"Files to upload ({len(files_to_upload)} files):")
    for f in sorted(files_to_upload):
        print(f"  - {f}")

    if args.dry_run:
        print("\n[Dry run] No files were uploaded.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("\nError: huggingface_hub is required. Install it using:")
        print("    pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=args.token)

    print("\nEnsuring repository exists...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    print("Uploading folder to Hugging Face Hub...")
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
        ignore_patterns=["*.pyc", "__pycache__", ".git*"],
    )

    print("\nUpload complete! Visit your dataset at:")
    print(f"  https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

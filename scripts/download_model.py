import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Gemma model snapshot locally for offline Docker deployment."
    )
    parser.add_argument(
        "--model-id",
        default="google/gemma-2-9b-it",
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/translate-gemma",
        help="Directory where model files will be stored.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. You can also set HF_TOKEN env var.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision (branch/tag/commit).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model already exists.",
    )
    return parser.parse_args()


def model_ready(output_dir: Path) -> bool:
    return output_dir.exists() and (output_dir / "config.json").exists()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_ready(output_dir) and not args.force:
        print(f"[skip] Model already exists at: {output_dir}")
        return

    print(f"[download] Fetching {args.model_id} into {output_dir}")
    snapshot_path = snapshot_download(
        repo_id=args.model_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        token=args.hf_token,
        revision=args.revision,
        resume_download=True,
    )
    print(f"[done] Snapshot stored at: {snapshot_path}")

    metadata = {
        "model_id": args.model_id,
        "revision": args.revision,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = output_dir / ".model-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[meta] Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()


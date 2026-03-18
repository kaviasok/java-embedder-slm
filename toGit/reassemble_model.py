#!/usr/bin/env python3
"""Reassemble model.safetensors from chunks (run once after clone so no LFS is needed)."""
from pathlib import Path

MODEL_NAME = "model.safetensors"

def main():
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir / "java_embedder_stage2_merged"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / MODEL_NAME
    parts = sorted(model_dir.glob(f"{MODEL_NAME}.part-*"))
    if not parts:
        raise SystemExit(
            f"No part files found in {model_dir}. "
            "If you see model.safetensors there already, the model is ready."
        )
    with open(out_path, "wb") as out:
        for p in parts:
            print(f"Adding {p.name}...")
            out.write(p.read_bytes())
    print(f"Written {out_path} ({out_path.stat().st_size / (1024*1024):.1f} MB)")

if __name__ == "__main__":
    main()

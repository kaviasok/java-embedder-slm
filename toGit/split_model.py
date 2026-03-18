#!/usr/bin/env python3
"""Split model.safetensors into chunks under 100MB for GitHub (no LFS). Run once before push."""
from pathlib import Path

CHUNK_BYTES = 90 * 1024 * 1024  # 90 MB (under GitHub 100 MB limit)
MODEL_NAME = "model.safetensors"

def main():
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir / "java_embedder_stage2_merged"
    path = model_dir / MODEL_NAME
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, "rb") as f:
        part = 0
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            part += 1
            out = model_dir / f"{MODEL_NAME}.part-{part:02d}"
            with open(out, "wb") as w:
                w.write(chunk)
            print(f"Wrote {out.name} ({len(chunk) / (1024*1024):.1f} MB)")
    print("Done. Remove model.safetensors before commit; use reassemble_model.py after clone.")

if __name__ == "__main__":
    main()

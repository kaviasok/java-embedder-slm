# Java embedder (fine-tuned) – use without Git LFS

After cloning this repo, **run once** (if you don’t have `model.safetensors` yet):

```bash
cd toGit
python reassemble_model.py
```

Then run the demo:

```bash
python embed_use.py
```

`embed_use.py` can also run `reassemble_model.py` for you automatically if it finds only part files.

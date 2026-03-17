from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def load_model(model_dir: Path):
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Make sure you copy your fine‑tuned model folder "
            "into this directory (for example 'java_embedder_stage2_merged')."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir))
    model.eval()
    return tokenizer, model


def encode_texts(tokenizer: AutoTokenizer, model: AutoModel, texts: list[str]) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoded)
        last_hidden = outputs.last_hidden_state  # (batch, seq, hidden)

        # Simple mean pooling with attention mask
        attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (batch, seq, 1)
        masked = last_hidden * attention_mask
        sum_embeddings = masked.sum(dim=1)
        lengths = attention_mask.sum(dim=1).clamp(min=1)
        embeddings = sum_embeddings / lengths

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


def demo():
    script_dir = Path(__file__).resolve().parent

    # Name this folder to match the model you copy in.
    # For example, copy your local 'java_embedder_stage2_merged' folder here.
    model_dir = script_dir / "java_embedder_stage2_merged"

    print(f"Loading model from: {model_dir}")
    tokenizer, model = load_model(model_dir)

    corpus = [
        "Java is a high-level, class-based, object-oriented programming language.",
        "Python is an interpreted, high-level and general-purpose programming language.",
        "The JVM executes Java bytecode and provides platform independence.",
    ]

    query = "How does Java run on different operating systems?"

    print("Encoding corpus and query...")
    corpus_embeddings = encode_texts(tokenizer, model, corpus)
    query_embedding = encode_texts(tokenizer, model, [query])[0].unsqueeze(0)

    scores = torch.matmul(query_embedding, corpus_embeddings.T)[0]  # cosine similarities

    best_idx = int(torch.argmax(scores).item())
    print("\nQuery:")
    print(query)
    print("\nMost similar corpus entry:")
    print(f"- Text: {corpus[best_idx]}")
    print(f"- Score: {scores[best_idx].item():.4f}")


if __name__ == "__main__":
    demo()


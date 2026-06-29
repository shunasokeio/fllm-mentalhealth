"""
Load MentalChat16K from Hugging Face, embed each data point with Llama-3.1-8B-Instruct
(last hidden state, mean-pooled over non-padding tokens), and save the embedded dataset.
"""
import os
import argparse
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DATASET = "ShenLab/MentalChat16K"
DEFAULT_OUTPUT_DIR = "embedded_data"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_LENGTH = 512


def get_text_to_embed(example, input_col="input"):
    """Embed only the input field (no instruction, no output)."""
    return (example.get(input_col, "") or "").strip() or " "


def mean_pool_last_hidden_state(last_hidden_state, attention_mask):
    """Mean-pool over non-padding tokens. Shapes: (batch, seq, dim), (batch, seq)."""
    mask = attention_mask.unsqueeze(-1).float()
    sum_h = (last_hidden_state * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return (sum_h / count).float()


def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if device and str(device) != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def embed_dataset(
    dataset,
    model,
    tokenizer,
    device,
    batch_size=DEFAULT_BATCH_SIZE,
    max_length=DEFAULT_MAX_LENGTH,
    input_col="input",
):
    """One embedding per row: mean-pooled last hidden state of the input text only (no instruction/output)."""
    all_embeddings = []
    n = len(dataset)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_examples = [dataset[i] for i in range(start, end)]
        texts = [get_text_to_embed(ex, input_col=input_col) for ex in batch_examples]
        if not any(texts):
            texts = [" " for _ in texts]

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        last_hidden = out.hidden_states[-1]
        pooled = mean_pool_last_hidden_state(last_hidden, enc["attention_mask"])
        all_embeddings.append(pooled.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Embed MentalChat16K with Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HuggingFace dataset name")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model for embeddings")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save embedded dataset")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for forward pass")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Max token length per example")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading full dataset {args.dataset} (all splits)...")
    full = load_dataset(args.dataset)
    dataset = concatenate_datasets([full[s] for s in full])
    print(f"Loaded {len(dataset)} examples.")

    print(f"Loading model and tokenizer: {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    print("Embedding...")
    embeddings = embed_dataset(
        dataset,
        model,
        tokenizer,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "embeddings.npy"), embeddings)
    embedding_list = [embeddings[i].tolist() for i in range(len(embeddings))]
    dataset_with_emb = dataset.add_column("embedding", embedding_list)
    dataset_with_emb.save_to_disk(os.path.join(args.output_dir, "mentalchat16k_embedded"))
    print(f"Saved to {args.output_dir}/mentalchat16k_embedded and embeddings.npy.")


if __name__ == "__main__":
    main()

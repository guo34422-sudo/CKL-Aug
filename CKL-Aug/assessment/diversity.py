import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--COL_ALL", type=str, required=True)
parser.add_argument("--EMBEDDING_MODEL", type=str, required=True)
parser.add_argument("--OUTPUT_TXT", type=str, required=True)
parser.add_argument("--BATCH_SIZE", type=int, default=8)
parser.add_argument("--DEVICE", type=str, default="cuda")
parser.add_argument("--NORMALIZE_EMBEDDINGS", action="store_true")

args = parser.parse_args()

DATA_PATH = args.data_path
COL_ALL = [col.strip() for col in args.COL_ALL.split(",")]
EMBEDDING_MODEL = args.EMBEDDING_MODEL
OUTPUT_TXT = args.OUTPUT_TXT
BATCH_SIZE = args.BATCH_SIZE
DEVICE = torch.device(args.DEVICE if torch.cuda.is_available() else "cpu")
NORMALIZE_EMBEDDINGS = args.NORMALIZE_EMBEDDINGS


def load_data(data_path, cols):
    df = pd.read_excel(data_path)
    df = df[cols].copy()

    for col in cols:
        df[col] = df[col].astype(str)

    return df


def load_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True
    )

    model.to(device)
    model.eval()

    return tokenizer, model


def encode_texts(texts, tokenizer, model, device, batch_size, max_length, normalize_embeddings=True):
    all_embeddings = []

    with torch.no_grad():
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx + batch_size]

            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            model_output = model(**encoded_input)

            embeddings = model_output.last_hidden_state[:, 0]

            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def calculate_homogeneity(embeddings):
    n = embeddings.size(0)
    h = embeddings.size(1)

    if n <= 1:
        return 1.0

    entropy = torch.tensor(0.0, device=embeddings.device)
    upper_bound = torch.log(torch.tensor(float(n - 1), device=embeddings.device) + 1e-10)

    if upper_bound.item() <= 0:
        return 1.0

    for embed in embeddings:
        diff_ij = embed.unsqueeze(0) - embeddings
        squared_dist_ij = torch.square(diff_ij).sum(dim=1)

        weights_ij = torch.sqrt(squared_dist_ij + 1e-12) ** torch.log(
            torch.tensor(float(h), device=embeddings.device)
        )

        weights_ij = weights_ij + 1e-12
        sum_weights = weights_ij.sum()
        prob_trans_ij = weights_ij / sum_weights
        log_prob_trans_ij = torch.log(prob_trans_ij + 1e-10)

        v = 1.0 / prob_trans_ij.size(0)
        entropy_ij = -torch.sum(v * prob_trans_ij * log_prob_trans_ij)
        entropy += entropy_ij

    return (entropy / upper_bound).item()


def calc_row_embedding_metrics(row_embeddings):
    row_embeddings = row_embeddings.float()
    n = row_embeddings.size(0)

    if n < 2:
        return {
            "avg_distance": 0.0,
            "avg_dissimilarity": 0.0,
            "embed_std": 0.0,
            "homogeneity": 1.0
        }

    pairwise_dist = torch.cdist(row_embeddings, row_embeddings, p=2)
    upper_mask = torch.triu(torch.ones_like(pairwise_dist, dtype=torch.bool), diagonal=1)

    avg_distance = pairwise_dist[upper_mask].mean().item()

    sims = torch.matmul(row_embeddings, row_embeddings.T)
    avg_similarity = sims[upper_mask].mean().item()
    avg_dissimilarity = 1.0 - avg_similarity

    embed_std = torch.std(row_embeddings, dim=0, unbiased=False).mean().item()
    homogeneity = calculate_homogeneity(row_embeddings)

    return {
        "avg_distance": avg_distance,
        "avg_dissimilarity": avg_dissimilarity,
        "embed_std": embed_std,
        "homogeneity": homogeneity
    }


def main():
    df = load_data(DATA_PATH, COL_ALL)
    tokenizer, model = load_model(EMBEDDING_MODEL, DEVICE)

    all_texts = []

    for _, row in df.iterrows():
        for col in COL_ALL:
            all_texts.append(row[col])

    embeddings = encode_texts(
        texts=all_texts,
        tokenizer=tokenizer,
        model=model,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        max_length=512,
        normalize_embeddings=NORMALIZE_EMBEDDINGS
    )

    result_rows = []
    n_cols = len(COL_ALL)

    for idx in range(len(df)):
        base = idx * n_cols
        row_embeddings = embeddings[base:base + n_cols].to(DEVICE)

        metric_result = calc_row_embedding_metrics(row_embeddings)

        row_result = {"id": idx}

        for col in COL_ALL:
            row_result[col] = df.loc[idx, col]

        row_result.update(metric_result)
        result_rows.append(row_result)

    result_df = pd.DataFrame(result_rows)

    dataset_avg_distance = float(result_df["avg_distance"].mean())
    dataset_avg_dissimilarity = float(result_df["avg_dissimilarity"].mean())
    dataset_embed_std = float(result_df["embed_std"].mean())
    dataset_homogeneity = float(result_df["homogeneity"].mean())

    with open(OUTPUT_TXT, "a", encoding="utf-8") as f:
        f.write(f"Average Distance:      {dataset_avg_distance:.6f}\n")
        f.write(f"Average Dissimilarity: {dataset_avg_dissimilarity:.6f}\n")
        f.write(f"Average Embed Std:     {dataset_embed_std:.6f}\n")
        f.write(f"Average Homogeneity:   {dataset_homogeneity:.6f}\n")


if __name__ == "__main__":
    main()

import json
import re

import numpy as np
from opencc import OpenCC

cc = OpenCC("t2s") 

from text2vec import SentenceModel

CHUNK_SIZE = 2_000_000  

import sqlite3

conn = sqlite3.connect("conceptnet_zh.db")
cur = conn.cursor()


def get_triples_sqlite(term):
    rows = cur.execute(
        "SELECT rel,s,e,dj FROM edges WHERE term=?",
        (term,)
    ).fetchall()
    return rows


def get_neighbors(term):
    neighbors = set()

    rows = get_triples_sqlite(term)

    for rel, s, e, dj in rows:
        # s -> e
        if s and s != term:
            neighbors.add(s)
        if e and e != term:
            neighbors.add(e)

    if neighbors:
        neighbors = [cc.convert(node) for node in neighbors]
        neighbors = single_word(neighbors, term)

    return neighbors


def cosine_scores(query_emb, cand_embs):
    # encode(normalize_embeddings=True)
    return (cand_embs @ query_emb).tolist()


def clean_word(word: str) -> str:
    if not word:
        return ""
    word = re.sub(r"[^\u4e00-\u9fff]", "", word)
    return word.strip()


def single_word(neighbor_nodes, term):
    seen = set()
    result = []
    for w in neighbor_nodes:
        w_clean = clean_word(w)
        if w_clean == term:
            continue
        if w_clean in seen:
            continue
        seen.add(w_clean)
        result.append(w_clean)
    return result


def get_scores(
        neighbor_nodes,
        sentence,
        model
):
    if not sentence or not neighbor_nodes:
        return [], []

    if neighbor_nodes:
        s_emb = model.encode([sentence], normalize_embeddings=True)[0]
        node_embs = model.encode(neighbor_nodes, normalize_embeddings=True)
        node_scores = cosine_scores(s_emb, node_embs)  
    return neighbor_nodes, node_scores

def main():
    out = []
    with open(JSON_TABLE, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = SentenceModel(model_name)

    index = 0

    for idx, item in enumerate(data[:], start=1):
        sentence = item.get("sent", None)
        source_domain = item.get("vehicle", None)

        neighbors = get_neighbors(source_domain)

        if len(neighbors) > 2:
            index = index + 1
            neighbor_nodes, node_scores = get_scores(neighbor_nodes=list(neighbors), sentence=sentence, model=model)

            scored_nodes = list(zip(neighbor_nodes, node_scores))
            scored_nodes.sort(key=lambda x: x[1], reverse=True)

            neighbors_sorted = [node for node, score in scored_nodes]

            append_data = {
                "origin_id": idx,
                "id": index,
                "sentence": sentence,
                "source_domain": source_domain,
                "neighbors_1": neighbors_sorted[0],
                "neighbors_2": neighbors_sorted[1],
                "neighbors_3": neighbors_sorted[2],
            }

            out.append(append_data)
            print(append_data)

            if len(out) % 50 == 0:
                with open(OUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

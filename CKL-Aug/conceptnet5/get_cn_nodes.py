import re
import pandas as pd
from opencc import OpenCC
from romote.auto_annotation.main.get_en_nodes import get_en_node
from translate import translate_zh_to_en, translate_en_to_zh

cc = OpenCC("t2s")  

from text2vec import SentenceModel



CHUNK_SIZE = 2_000_000  

import sqlite3

conn = sqlite3.connect("./conceptnet_zh.db")
cur = conn.cursor()
conn_en = sqlite3.connect("./conceptnet_en.db")
cur_en = conn_en.cursor()


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

def main(csv_file_path, output_file_path, start_index=0, end_index=None):
    data = pd.read_csv(csv_file_path)
    output_data = []

    model = SentenceModel(model_name)
    model_en = SentenceModel(model_en_name)


    if end_index is None:
        end_index = len(data)

    for index, row in data.iloc[start_index:end_index].iterrows():
        if 'id' in row:
            sentence = row['Sentence']
            source_domain = row['source_domain']

            neighbors = get_neighbors(source_domain)
            neighbor_nodes, node_scores = get_scores(neighbor_nodes=list(neighbors), sentence=sentence, model=model)

            scored_nodes = list(zip(neighbor_nodes, node_scores))
            scored_nodes.sort(key=lambda x: x[1], reverse=True)

            neighbors_sorted = [node for node, score in scored_nodes]

            if len(neighbors_sorted)<3:
                sentence_en = translate_zh_to_en(sentence,4,64)
                sd_en = translate_zh_to_en(source_domain,1,4)
                neighbors_sorted_en = get_en_node(model_en, sentence_en, sd_en, cur_en)
                neighbors_sorted_zh = [translate_en_to_zh(x) if x else None for x in neighbors_sorted_en]

                for x in neighbors_sorted_zh:
                    neighbors_sorted.append(x)

                neighbors_sorted = single_word(neighbors_sorted, source_domain)

                neighbors_sorted = neighbors_sorted[:3] + [None] * (3 - len(neighbors_sorted))

                row["neighbors_1"] = neighbors_sorted[0]
                row["neighbors_2"] = neighbors_sorted[1]
                row["neighbors_3"] = neighbors_sorted[2]
                row["sentence_en"] = sentence_en
                row["sd_en"] = sd_en
                row["neighbors_sorted_en"] = neighbors_sorted_en
                row["neighbors_sorted_zh"] = neighbors_sorted_zh

            else:
                row["neighbors_1"] = neighbors_sorted[0]
                row["neighbors_2"] = neighbors_sorted[1]
                row["neighbors_3"] = neighbors_sorted[2]

            output_data.append(row)
            print(f"Processed {index + 1}/{len(data)}: {neighbors_sorted} ")

            df = pd.DataFrame(output_data)
            df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    main(csv_path,out_path)

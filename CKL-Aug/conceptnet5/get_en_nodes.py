import re

def get_triples_sqlite_en(term_en, cur_en):
    rows = cur_en.execute(
        "SELECT rel,s,e,dj FROM edges WHERE term=?",
        (term_en,)
    ).fetchall()
    return rows

def normalize_en_term(term: str) -> str:
    """
    把英文 term 归一化为你入库时的 key 形式（一般为小写、空格->下划线、去掉 pos tag）
    例如: "Ice Cream" -> "ice_cream"
         "run/v" -> "run"
    """
    if not term:
        return ""
    term = term.strip().lower()
    term = term.replace(" ", "_")
    # ConceptNet 里可能有 run/v, bank/n 这种
    term = term.split("/")[0]
    # 只保留英文字符、数字、下划线、连字符
    term = re.sub(r"[^a-z0-9_\-]", "", term)
    return term

def clean_en_word(w: str) -> str:
    """清洗邻居节点：小写 + 去 pos tag + 只保留英文/下划线/连字符"""
    if not w:
        return ""
    w = w.strip().lower()
    w = w.split("/")[0]
    w = re.sub(r"[^a-z0-9_\- ]", "", w)
    w = w.replace(" ", "_")
    return w

def single_word_en(neighbor_nodes, term_en):
    seen = set()
    result = []
    for w in neighbor_nodes:
        w_clean = clean_en_word(w)
        if not w_clean:
            continue
        if w_clean == term_en:
            continue
        if w_clean in seen:
            continue
        seen.add(w_clean)
        result.append(w_clean)
    return result

def get_neighbors_en(term_en: str, cur_en):
    """
    从英文 ConceptNet DB 中取邻居节点（1-hop），并做清洗去重
    """
    term_en = normalize_en_term(term_en)
    if not term_en:
        return []

    rows = get_triples_sqlite_en(term_en,cur_en)

    neighbors = set()
    for rel, s, e, dj in rows:
        # s -> e
        if s and s != term_en:
            neighbors.add(s)
        if e and e != term_en:
            neighbors.add(e)

    return single_word_en(list(neighbors), term_en) if neighbors else []

def cosine_scores(query_emb, cand_embs):
    return (cand_embs @ query_emb).tolist()

def get_scores(neighbor_nodes, sentence, model):
    if not sentence or not neighbor_nodes:
        return [], []
    s_emb = model.encode([sentence], normalize_embeddings=True)[0]
    node_embs = model.encode(neighbor_nodes, normalize_embeddings=True)
    node_scores = cosine_scores(s_emb, node_embs)
    return neighbor_nodes, node_scores

def get_en_node(model,sentence,source_domain,cur_en):
    neighbors = get_neighbors_en(source_domain,cur_en)
    neighbor_nodes, node_scores = get_scores(neighbor_nodes=list(neighbors), sentence=sentence, model=model)

    scored_nodes = list(zip(neighbor_nodes, node_scores))
    scored_nodes.sort(key=lambda x: x[1], reverse=True)

    neighbors_sorted = [node for node, score in scored_nodes]
    # 只保留 3 个，不足补 None
    neighbors_sorted = neighbors_sorted[:3] + [None] * (3 - len(neighbors_sorted))

    return neighbors_sorted



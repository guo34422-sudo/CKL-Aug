import re


def normalize_en_term(term: str) -> str:
    if not term:
        return ""

    term = term.strip().lower()

    if term.startswith("/c/en/"):
        term = term.replace("/c/en/", "", 1)

    term = term.replace(" ", "_")
    term = term.split("/")[0]
    term = re.sub(r"[^a-z0-9_\-]", "", term)

    return term


def clean_en_word(word: str) -> str:
    if not word:
        return ""

    word = word.strip().lower()

    if word.startswith("/c/en/"):
        word = word.replace("/c/en/", "", 1)

    word = word.replace(" ", "_")
    word = word.split("/")[0]
    word = re.sub(r"[^a-z0-9_\-]", "", word)

    return word


def get_triples_sqlite_en(term_en, cur_en):
    return cur_en.execute(
        "SELECT rel, s, e, dj FROM edges WHERE term=?",
        (term_en,)
    ).fetchall()


def single_word_en(neighbor_nodes, term_en):
    term_en = normalize_en_term(term_en)
    seen = set()
    result = []

    for word in neighbor_nodes:
        word_clean = clean_en_word(word)

        if not word_clean:
            continue

        if word_clean == term_en:
            continue

        if word_clean in seen:
            continue

        seen.add(word_clean)
        result.append(word_clean)

    return result


def get_neighbors_en(term_en, cur_en):
    term_en = normalize_en_term(term_en)

    if not term_en:
        return []

    rows = get_triples_sqlite_en(term_en, cur_en)
    neighbors = set()

    for _, s, e, _ in rows:
        s = clean_en_word(s)
        e = clean_en_word(e)

        if s and s != term_en:
            neighbors.add(s)

        if e and e != term_en:
            neighbors.add(e)

    return single_word_en(list(neighbors), term_en)


def cosine_scores(query_emb, cand_embs):
    return (cand_embs @ query_emb).tolist()


def get_scores(neighbor_nodes, sentence, model):
    if not sentence or not neighbor_nodes:
        return [], []

    sentence_embedding = model.encode(
        [sentence],
        normalize_embeddings=True
    )[0]

    node_embeddings = model.encode(
        neighbor_nodes,
        normalize_embeddings=True
    )

    node_scores = cosine_scores(sentence_embedding, node_embeddings)

    return neighbor_nodes, node_scores


def get_en_node(model, sentence, source_domain, cur_en, top_k=3):
    neighbors = get_neighbors_en(source_domain, cur_en)
    neighbor_nodes, node_scores = get_scores(
        neighbor_nodes=list(neighbors),
        sentence=sentence,
        model=model
    )

    scored_nodes = list(zip(neighbor_nodes, node_scores))
    scored_nodes.sort(key=lambda x: x[1], reverse=True)

    neighbors_sorted = [node for node, _ in scored_nodes]
    neighbors_sorted = neighbors_sorted[:top_k] + [None] * (top_k - len(neighbors_sorted))

    return neighbors_sorted

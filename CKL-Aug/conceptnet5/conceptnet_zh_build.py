import sqlite3
import pandas as pd

TSV = "./conceptnet_zh.tsv"
DB  = "./conceptnet_zh.db"
CHUNK_SIZE = 2_000_000

def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS edges(
      term TEXT,
      rel  TEXT,
      s    TEXT,
      e    TEXT,
      dj   TEXT
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_term ON edges(term)")
    conn.commit()

    cols = ["assertion_uri", "rel", "start", "end", "data_json"]
    reader = pd.read_csv(
        TSV, sep="\t", header=None, names=cols, dtype=str,
        chunksize=CHUNK_SIZE, quoting=3, on_bad_lines="skip", engine="c"
    )

    buf = []
    n = 0
    for chunk in reader:
        chunk = chunk[chunk["start"].str.startswith("/c/zh/", na=False) &
                      chunk["end"].str.startswith("/c/zh/", na=False)]

        chunk["rel"] = chunk["rel"].str.replace(r"^/r/", "", regex=True)
        chunk["start"] = chunk["start"].str.replace(r"^/c/zh/", "", regex=True)
        chunk["end"] = chunk["end"].str.replace(r"^/c/zh/", "", regex=True)

        for rel, s, e, dj in chunk[["rel","start","end","data_json"]].itertuples(index=False, name=None):
            buf.append((s, rel, s, e, dj))
            buf.append((e, rel, s, e, dj))

        if len(buf) >= 200000:  
            cur.executemany("INSERT INTO edges(term,rel,s,e,dj) VALUES (?,?,?,?,?)", buf)
            conn.commit()
            n += len(buf)
            print("inserted", n)
            buf.clear()

    if buf:
        cur.executemany("INSERT INTO edges(term,rel,s,e,dj) VALUES (?,?,?,?,?)", buf)
        conn.commit()
        n += len(buf)
        print("inserted", n)

    conn.close()
    print("done, db =", DB)

if __name__ == "__main__":
    main()

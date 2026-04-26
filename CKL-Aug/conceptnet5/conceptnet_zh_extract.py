import pandas as pd
from pathlib import Path


SEP = "\t"         
CHUNKSIZE = 200_000

START_COL = 2      
END_COL   = 3      

def filter_zh_zh_rows():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    first = True
    total_in = total_out = 0

    for i, chunk in enumerate(pd.read_csv(
            IN_FILE,
            sep=SEP,
            header=None,
            dtype=str,
            chunksize=CHUNKSIZE,
            compression="infer",
            on_bad_lines="skip",
    ), start=1):
        total_in += len(chunk)

        m = chunk[START_COL].str.startswith("/c/zh/") & chunk[END_COL].str.startswith("/c/zh/")
        zh_rows = chunk.loc[m]
        print(f"chunk {i}: in={len(chunk)} kept={len(zh_rows)}")  

        total_out += len(zh_rows)

        zh_rows.to_csv(
            OUT_FILE,
            sep="\t",
            index=False,
            header=False,              
            mode="w" if first else "a",
            encoding="utf-8",
        )
        first = False

    print(f"Done. total_in={total_in:,}, kept_zh_zh={total_out:,}")
    print(f"Output: {OUT_FILE}")

if __name__ == "__main__":
    filter_zh_zh_rows()

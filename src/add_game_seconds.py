# src/add_game_seconds.py
import argparse
import pandas as pd
import numpy as np

def mmss_to_sec(s: str) -> int:
    
    m, ss = str(s).split(":")
    return int(m) * 60 + int(ss)

def compute_game_seconds(df: pd.DataFrame,
                         game_col="game_id",
                         period_col="period",
                         time_col="periodTime",
                         out_col="game_seconds") -> pd.Series:
    
    # 1) time convert to secs
    if time_col not in df.columns:
        raise ValueError(f"missing: {time_col}")
    tsec = df[time_col].astype(str).map(mmss_to_sec)

    # 2) maximum seconds
    tmp = df.copy()
    tmp["_tsec_"] = tsec
    per_len = (tmp
               .groupby([game_col, period_col], as_index=False)["_tsec_"]
               .max()
               .rename(columns={"_tsec_": "_period_len_"}))

    # 3) 
    per_len["_period_offset_"] = (per_len
                                  .groupby(game_col)["_period_len_"]
                                  .cumsum()
                                  .shift(fill_value=0))

    # 4)
    merged = df[[game_col, period_col]].merge(
        per_len[[game_col, period_col, "_period_offset_"]],
        on=[game_col, period_col], how="left"
    )
    offsets = merged["_period_offset_"].astype(int)

    # 5) game_seconds 
    game_seconds = offsets.values + tsec.values
    return pd.Series(game_seconds, index=df.index, name=out_col)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="input：CSV（ game_id/period/periodTime）")
    ap.add_argument("--out_csv", required=True, help="output：write back to newly added 'game_seconds' CSV")
    ap.add_argument("--game_col", default="game_id")
    ap.add_argument("--period_col", default="period")
    ap.add_argument("--time_col", default="periodTime")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    df["game_seconds"] = compute_game_seconds(
        df, game_col=args.game_col, period_col=args.period_col, time_col=args.time_col
    )
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] ：{args.out_csv}（added game_seconds）")

if __name__ == "__main__":
    main()




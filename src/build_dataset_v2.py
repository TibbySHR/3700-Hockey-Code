# src/build_dataset_v2.py
import argparse
from pathlib import Path
import pandas as pd
from src.features_v2 import (
    ensure_game_seconds, attach_shot_geofeatures, build_last_event_features, SHOT_TYPES
)
from src.pp_utils import compute_powerplay_features

import pandas as pd
import numpy as np

from src.features_basic import (compute_shot_distance,compute_shot_angle)

def build_basic_features(df):
    df = df.copy()

    # Distance & angle
    df["shotDistance"] = compute_shot_distance(df)
    df["shotAngle"] = compute_shot_angle(df)

    # Event-based features
    df["shotType"] = df["eventType"]
    df["isGoal"] = (df["eventType"] == "GOAL").astype(int)

    # Normalize empty_net if missing
    df["empty_net"] = df["empty_net"].fillna(0).astype(int)

    return df



def load_events(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)


    
    assert {"game_id","event_idx","period","x","y","eventType"}.issubset(df.columns), "missing base columns"
    if "game_seconds" not in df.columns:
        df["game_seconds"] = ensure_game_seconds(df, time_col="periodTime", period_col="period")
    # shot_type might be misssing ,replace with "unknown"
    if "shotType" not in df.columns:
        df["shotType"] = "Unknown"
    # empty_net 
    if "emptyNet" not in df.columns:
        df["emptyNet"] = 0
    return df

def filter_seasons(df: pd.DataFrame, season_from=2016, season_to=2019) -> pd.DataFrame:
    
    if "season" in df.columns:
        return df[(df["season"] >= season_from) & (df["season"] <= season_to)].copy()
    
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events_csv", type=str, required=True,
                    help="Step1 cleaned CSV")
    ap.add_argument("--out_csv", type=str, default="data/processed/shots_train_fixed.csv")
    ap.add_argument("--with_powerplay", action="store_true", help="calcul pp related（bonus）")
    args = ap.parse_args()

    df = load_events(args.events_csv)
    df = filter_seasons(df, 2016, 2019)

    
    if args.with_powerplay:
        df = compute_powerplay_features(df)

   
    df = attach_shot_geofeatures(df)


    df = build_last_event_features(df)

 
    shots = df[df["eventType"].isin({"SHOT", "GOAL"})].copy()

    
    shots["is_goal"] = (shots["eventType"] == "GOAL").astype(int)

    keep_cols = [
        "game_id","event_idx","period","game_seconds",
        "x","y","shotDistance","shotAngle","shotType",
        "isGoal","empty_net"
    ]

    if args.with_powerplay:
        keep_cols += ["home_skaters","away_skaters","pp_elapsed"]

    shots = shots[keep_cols].reset_index(drop=True)

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    shots.to_csv(outp, index=False)
    print(f"[OK] Saved shots dataset with FE v2 → {outp} | rows={len(shots)}")

if __name__ == "__main__":
    main()



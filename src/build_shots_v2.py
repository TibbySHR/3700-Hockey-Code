# src/build_shots_v2.py   modificationo 20260102cd
import argparse
import pandas as pd
import numpy as np
import glob
import json
from src import features_v2


# Approximate net location (standardized attacking side)
NET_X, NET_Y = 89, 0

def compute_shot_distance(df):
    return np.hypot(df["x"] - NET_X, df["y"] - NET_Y)

def compute_shot_angle(df):
    return np.degrees(
        np.arctan2(np.abs(df["y"] - NET_Y), np.abs(df["x"] - NET_X))
    )

def parse_mmss(x):
    if not isinstance(x, str) or ":" not in x:
        return None
    m, s = x.split(":")
    try:
        return int(m) * 60 + int(s)
    except Exception:
        return None

def load_events(raw_glob):
    rows = []

    for fp in glob.glob(raw_glob):
        with open(fp, "r") as f:
            game = json.load(f)

        # -------- Case 1: new gamecenter format --------
        if "plays" in game and isinstance(game["plays"], list):
            game_id = game.get("id")
            plays = game["plays"]

        # -------- Case 2: old StatsAPI format --------
        elif "liveData" in game:
            game_id = game.get("gamePk")
            plays = game.get("liveData", {}).get("plays", {}).get("allPlays", [])

        else:
            continue  # unknown format

        for p in plays:

            period = (p.get("periodDescriptor") or {}).get("number")
            period_time = p.get("timeInPeriod")              # "MM:SS"
            event_idx = p.get("sortOrder")                   # order in game (good for last-event)

            details = p.get("details", {}) or {}
            coords = p.get("coordinates", {}) or {}

            x = details.get("xCoord", details.get("x", coords.get("x")))
            y = details.get("yCoord", details.get("y", coords.get("y")))
            shot_type = details.get("shotType", details.get("shotTypeDesc"))


            rows.append({
                "game_id": game_id,
                "event_idx": event_idx,
                "period": period,
                "period_time": period_time,
                "eventType": p.get("typeDescKey"),
                "x": x,
                "y": y,
                "empty_net": details.get("emptyNet", 0),  # new format often stores this in details when present
                "raw_shot_type": details.get("shotType", details.get("shotTypeDesc")),
        })


    cols = [
        "game_id","event_idx","period","period_time",
        "eventType","x","y","empty_net","raw_shot_type"
    ]
    return pd.DataFrame(rows, columns=cols)


def compute_game_seconds(df):
    secs = pd.to_numeric(
        df["period_time"].apply(parse_mmss),
        errors="coerce"
    ).fillna(0).astype(int)

    periods = pd.to_numeric(
        df["period"],
        errors="coerce"
    ).fillna(0).astype(int)

    return (periods - 1) * 20 * 60 + secs
"""

def add_last_event_features(events_df):
    
    df = events_df.sort_values(
        ["game_id", "event_idx"]
    ).copy()

    g = df.groupby("game_id", sort=False)

    df["last_eventType"] = g["eventType"].shift(1)
    df["last_x"] = g["x"].shift(1)
    df["last_y"] = g["y"].shift(1)
    df["last_game_seconds"] = g["game_seconds"].shift(1)

    df["time_since_last_event"] = (
        df["game_seconds"] - df["last_game_seconds"]
    )
    df["dist_from_last_event"] = np.sqrt(
        (df["x"] - df["last_x"]) ** 2 +
        (df["y"] - df["last_y"]) ** 2
    )

    return df



def add_rebound_speed(shots_df):
    df = shots_df.copy()

    # Rebound if previous event was a shot or goal
    df["isRebound"] = df["last_eventType"].isin(
        ["shot-on-goal", "goal"]
    ).astype(int)

    last_angle = np.degrees(
        np.arctan2(
            np.abs(df["last_y"] - NET_Y),
            np.abs(df["last_x"] - NET_X)
        )
    )

    df["angle_change_on_rebound"] = np.abs(
        df["shotAngle"] - last_angle
    )
    df.loc[df["isRebound"] == 0, "angle_change_on_rebound"] = 0

    # Speed = distance / time (avoid divide-by-zero)
    denom = df["time_since_last_event"].replace(0, np.nan)
    df["play_speed"] = df["dist_from_last_event"] / denom

    # Clean infinities / NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df

"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", default="data/raw")
    parser.add_argument(
        "--out_csv",
        default="data/processed/shots_train_v2.csv"
    )
    parser.add_argument("--start", type=int, default=2016)
    parser.add_argument("--end", type=int, default=2019)
    parser.add_argument(
        "--mirror_x",
        action="store_true",
        help="Mirror x to same attacking side using abs(x)"
    )
    parser.add_argument(
        "--upload_wandb_subset",
        action="store_true",
        help="Upload required single-game dataset to WandB"
    )
    parser.add_argument(
        "--wandb_project",
        default="IFT6758.2025-A-3700"
    )
    args = parser.parse_args()

    # ---------- Load all events ----------
    dfs = []
   
    
    for year in range(args.start, args.end + 1):
        print(f"→ Processing season {year}...")

        pattern = f"{args.raw_root}/{year}/type-02/*.json"
        fps = glob.glob(pattern)
        print(f"   pattern={pattern} -> files={len(fps)}")

        df = load_events(pattern)
        print(f"   loaded rows (events)={len(df)}")

        if df.empty:
            print(f"   ⚠️ empty season, skipping {year}")
            continue

        df["game_seconds"] = compute_game_seconds(df)
        dfs.append(df)


    events = pd.concat(dfs, ignore_index=True)


    print("events rows:", len(events))



    # Numeric cleaning
    for col in ["x", "y"]:
        events[col] = pd.to_numeric(events[col], errors="coerce")
    
    print("after numeric cast: rows=", len(events),
      " non-null x=", events["x"].notna().sum(),
      " non-null y=", events["y"].notna().sum())


    events["empty_net"] = (
        pd.to_numeric(events["empty_net"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Optional coordinate standardization
    if args.mirror_x:
        events["x"] = events["x"].abs()

    # ---------- Last-event features on ALL events ----------
    events = features_v2.add_last_event_features(events)

    print("after last-event: rows=", len(events))


    # ---------- Shot / goal subset ----------
    shots = events[
        events["eventType"].isin(["goal", "shot-on-goal"])
    ].copy()

    print("shots rows (goal+sog):", len(shots),
      " goals=", (shots["eventType"]=="goal").sum(),
      " sog=", (shots["eventType"]=="shot-on-goal").sum())
    print("shots non-null x=", shots["x"].notna().sum(), " non-null y=", shots["y"].notna().sum())


    # Drop rows without coordinates (important for Part 2 plots)
    shots = shots.dropna(subset=["x", "y"]).copy()

    print("after dropna x,y: shots rows=", len(shots))


    shots["shotDistance"] = features_v2.compute_shot_distance(shots)
    shots["shotAngle"] = features_v2.compute_shot_angle(shots)

    # Part4: 真正的 shotType（Wrist/Slap/Backhand...）
    shots = features_v2.normalize_shot_type_col(shots)

    shots["isGoal"] = (shots["eventType"] == "goal").astype(int)

    # Part4 rebound/speed features
    shots = features_v2.add_rebound_speed(shots)


    print("final shots rows before save:", len(shots))

    shots.to_csv(args.out_csv, index=False)
    print(f"✅ Saved: {args.out_csv} (rows={len(shots)})")

    # ---------- Part 4 Q5: upload dataset artifact ----------
    if args.upload_wandb_subset:
        import wandb

        GAME_ID_TARGET = 2017021065
        subset = shots[shots["game_id"] == GAME_ID_TARGET].copy()

        print(
            f"→ Uploading subset game {GAME_ID_TARGET} "
            f"(rows={len(subset)})"
        )

        run = wandb.init(
            project=args.wandb_project,
            name="Feature Engineering",
            job_type="dataset-upload"
        )

        artifact_name = "wpg_v_wsh_2017021065"
        artifact = wandb.Artifact(
            artifact_name,
            type="dataset"
        )

        table = wandb.Table(dataframe=subset)
        artifact.add(table, artifact_name)

        run.log_artifact(artifact)
        run.finish()

        print(f"✅ Logged W&B artifact: {artifact_name}")

if __name__ == "__main__":
    main()

# src/build_shots_v2.py
import argparse
import pandas as pd
import numpy as np
import glob
import json
import os

NET_X, NET_Y = 89, 0  

def compute_shot_distance(df):
    return np.hypot(df["x"] - NET_X, df["y"] - NET_Y)

def compute_shot_angle(df):
    return np.degrees(np.arctan2(np.abs(df["y"] - NET_Y), np.abs(df["x"] - NET_X)))

def parse_mmss(x):
    if not isinstance(x, str) or ":" not in x:
        return None
    m, s = x.split(":")
    return int(m) * 60 + int(s)


def load_events(raw_glob):
    rows = []
    for fp in glob.glob(raw_glob):
        with open(fp, "r") as f:
            game = json.load(f)

        game_id = game["id"]  
        plays = game["plays"]

        for p in plays:
            about = p.get("about", {})
            coords = p.get("coordinates", {})
            result = p.get("result", {})

            rows.append({
                "game_id": game_id,
                "event_idx": p.get("eventIdx"),
                "period": about.get("period"),
                "period_time": about.get("periodTime"),
                "eventType": p.get("typeDescKey"),
                "x": coords.get("x"),
                "y": coords.get("y"),
                "empty_net": result.get("emptyNet", 0),
            })
    return pd.DataFrame(rows)


def compute_game_seconds(df):
    # 秒数：用 parse_mmss 解析后，强制转数值，无法解析为 NaN，再填 0，最后转 int
    secs = pd.to_numeric(df["period_time"].apply(parse_mmss), errors="coerce").fillna(0).astype(int)
    # 节：同样确保是整数
    periods = pd.to_numeric(df["period"], errors="coerce").fillna(0).astype(int)
    return (periods - 1) * 20 * 60 + secs



def add_last_event_features(df):
    df = df.sort_values(["game_id","event_idx"])
    df["last_eventType"] = df.groupby("game_id")["eventType"].shift(1)
    df["last_x"] = df.groupby("game_id")["x"].shift(1)
    df["last_y"] = df.groupby("game_id")["y"].shift(1)
    df["last_game_seconds"] = df.groupby("game_id")["game_seconds"].shift(1)

    df["time_since_last_event"] = df["game_seconds"] - df["last_game_seconds"]
    df["dist_from_last_event"] = np.sqrt((df["x"]-df["last_x"])**2 + (df["y"]-df["last_y"])**2)
    return df

def add_rebound_speed(df):
    # Rebound：上一事件为射门或进球
    df["isRebound"] = df["last_eventType"].isin(["shot-on-goal", "goal"]).astype(int)

    last_angle = np.degrees(np.arctan2(np.abs(df["last_y"] - NET_Y), np.abs(df["last_x"] - NET_X)))
    df["angle_change_on_rebound"] = np.abs(df["shotAngle"] - last_angle)
    df.loc[df["isRebound"] == 0, "angle_change_on_rebound"] = 0

    # 速度 = 位移 / 时间；避免除零
    denom = df["time_since_last_event"].replace(0, np.nan)
    df["play_speed"] = df["dist_from_last_event"] / denom

    # 清理无穷/NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", default="data/raw")
    parser.add_argument("--out_csv", default="data/processed/shots_train_v2.csv")
    parser.add_argument("--start", type=int, default=2016)
    parser.add_argument("--end", type=int, default=2019)
    args = parser.parse_args()

    dfs = []
    for year in range(args.start, args.end+1):
        print(f"→ Processing season {year}...")
        df = load_events(f"{args.raw_root}/{year}/type-02/*.json")
        df["game_seconds"] = compute_game_seconds(df)
        dfs.append(df)

    events = pd.concat(dfs, ignore_index=True)

    
    for col in ["x", "y"]:
        events[col] = pd.to_numeric(events[col], errors="coerce")
    
    events["empty_net"] = pd.to_numeric(events["empty_net"], errors="coerce").fillna(0).astype(int)


    shots = events[events["eventType"].isin(["goal","shot-on-goal"])].copy()

    shots["shotDistance"] = compute_shot_distance(shots)
    shots["shotAngle"] = compute_shot_angle(shots)
    shots["shotType"] = shots["eventType"]
    shots["isGoal"] = (shots["eventType"] == "goal").astype(int)

    shots = add_last_event_features(shots)
    shots = add_rebound_speed(shots)

    shots.to_csv(args.out_csv, index=False)
    print(f"✅ Saved: {args.out_csv}")

    # ------- 第5题：过滤并上传 WandB Artifact -------
    import wandb

    GAME_ID_TARGET = 2017021065
    subset = shots[shots["game_id"] == GAME_ID_TARGET].copy()

    # 可选：本地也存一份，便于核对
    subset_out = "data/processed/wpg_v_wsh_2017021065.csv"
    subset.to_csv(subset_out, index=False)
    print(f"✅ Saved subset: {subset_out} (rows={len(subset)})")

    # 上传到 WandB（把 project/name 换成你的）
    run = wandb.init(project="ift6758-stage2", job_type="dataset-upload")

    artifact_name = "wpg_v_wsh_2017021065"
    artifact = wandb.Artifact(artifact_name, type="dataset")

    # 用 Table 包装 DataFrame
    table = wandb.Table(dataframe=subset)
    artifact.add(table, artifact_name)

    run.log_artifact(artifact)
    run.finish()
    print(f"✅ Logged W&B artifact: {artifact_name}")


if __name__ == "__main__":
    main()

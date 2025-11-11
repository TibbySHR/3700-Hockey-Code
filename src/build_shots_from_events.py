import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# 球门中心
NET_X, NET_Y = 89, 0

def compute_shot_distance(x, y):
    return np.sqrt((x - NET_X)**2 + (y - NET_Y)**2)

def compute_shot_angle(x, y):
    return np.degrees(np.arctan2(abs(y - NET_Y), abs(x - NET_X)))

SHOT_LIKE = {"shot-on-goal", "missed-shot", "goal"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # 统一列检查
    need = {"eventType", "x", "y"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"缺少必要列: {miss} in {in_path}")

    # 只保留射门相关（包括进球本身）
    shots = df[df["eventType"].isin(SHOT_LIKE)].copy()

    # isGoal: goal=1，其余=0
    shots["isGoal"] = (shots["eventType"] == "goal").astype(int)

    # 计算距离与角度
    shots["shotDistance"] = compute_shot_distance(shots["x"], shots["y"])
    shots["shotAngle"] = compute_shot_angle(shots["x"], shots["y"])

    # shotType：这里先用 eventType 占位
    shots["shotType"] = shots["eventType"]

    # empty_net：没有就补0
    if "empty_net" not in shots.columns:
        shots["empty_net"] = 0
    shots["empty_net"] = shots["empty_net"].fillna(0).astype(int)

    # 选列（有则保留）
    keep_cols = [
        "game_id","event_idx","period","game_seconds",
        "x","y","shotDistance","shotAngle","shotType",
        "isGoal","empty_net","eventType"
    ]
    keep_cols = [c for c in keep_cols if c in shots.columns]  # 只保留存在的
    shots = shots[keep_cols]

    shots.to_csv(out_path, index=False)
    print(f"✅ Saved shots: {out_path}")
    print(shots["isGoal"].value_counts())

if __name__ == "__main__":
    main()

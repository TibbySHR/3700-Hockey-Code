import numpy as np
import pandas as pd

# 1) 计算射门距离
def compute_shot_distance(df):
    # NHL 标准球门中心位置 (89,0)
    net_x, net_y = 89, 0
    return np.sqrt((df["x"] - net_x)**2 + (df["y"] - net_y)**2)

# 2) 计算射门角度（绝对值 + 弧度转角度制）
def compute_shot_angle(df):
    net_x, net_y = 89, 0
    return np.degrees(np.arctan2(np.abs(df["y"] - net_y), np.abs(df["x"] - net_x)))

# 3) 构建基础特征
def build_basic_features(df):
    df = df.copy()

    df["shotDistance"] = compute_shot_distance(df)
    df["shotAngle"] = compute_shot_angle(df)

    # shotType 用 eventType 代替（后面有更好方式）
    df["shotType"] = df["eventType"]

    # GOAL → 1，其他 → 0
    df["isGoal"] = (df["eventType"] == "GOAL").astype(int)

    # empty_net 填缺失
    if "empty_net" in df.columns:
        df["empty_net"] = df["empty_net"].fillna(0).astype(int)

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    df = build_basic_features(df)

    df.to_csv(args.out_csv, index=False)
    print(f"✅ Saved: {args.out_csv}")


if __name__ == "__main__":
    main()

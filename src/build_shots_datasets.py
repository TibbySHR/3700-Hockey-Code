# src/build_shots_datasets.py

import pandas as pd
import numpy as np
from pathlib import Path


SEASON_FILES = {
    2016: "data/clean/shots_goals_regular_2016_2017.csv",
    2017: "data/clean/shots_goals_regular_2017_2018.csv",
    2018: "data/clean/shots_goals_regular_2018_2019.csv",
    2019: "data/clean/shots_goals_regular_2019_2020.csv",
    2020: "data/clean/shots_goals_regular_2020_2021.csv",
}

# 2) 冰球门的位置（这个在你的作业讲义里应该有）
#   这里给一个常用近似：进攻方向统一朝 +x，球门在 (89, 0)
NET_X = 89.0
NET_Y = 0.0

# 3) 输出文件路径
SHOTS_TRAIN_PATH = Path("data/processed/shots_train.csv")
SHOTS_TEST_PATH = Path("data/processed/shots_test.csv")
# ======================================


def load_all_seasons(season_files: dict[int, str]) -> pd.DataFrame:
    """读取多个赛季的 pbp 文件并拼起来。"""
    dfs = []
    for season_start, path in season_files.items():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        df = pd.read_csv(path)
        # 如果文件里没有 season_start_year 列，可以在这里补上
        if "season_start_year" not in df.columns:
            df["season_start_year"] = season_start
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


def preprocess_pbp(df: pd.DataFrame) -> pd.DataFrame:
    """只保留我们需要的列 + 只要常规赛."""
    # 只用常规赛
    if "game_type" in df.columns:
        df = df[df["game_type"] == "regular"].copy()

    # 只保留我们后面要用的列
    cols_keep = [
        "game_id",
        "season_start_year",
        "period",
        "periodTime",
        "eventType",
        "team",
        "x",
        "y",
        "shotType",
        "emptyNet",
        "strength",
    ]
    cols_exist = [c for c in cols_keep if c in df.columns]
    return df[cols_exist].copy()


def build_shot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    从事件级别 pbp 中抽取 SHOT/GOAL，
    并计算 distance / angle / is_goal / empty_net。
    """

    # 只要 SHOT 和 GOAL
    df = df[df["eventType"].isin(["SHOT", "GOAL"])].copy()

    # 有些事件可能缺坐标，直接丢掉
    df = df.dropna(subset=["x", "y"])

    # 统一进攻方向：假设主攻方向是 +x，
    # 把所有朝 -x 的射门镜像到 +x 这边（常见做法）
    df["x_coord"] = df["x"]
    df["y_coord"] = df["y"]

    # 如果你的 Step1 已经有“调整后坐标”，可以直接用那个
    # 这里的简单处理：x < 0 的镜像到正半场
    mask_left = df["x_coord"] < 0
    df.loc[mask_left, "x_coord"] = -df.loc[mask_left, "x_coord"]
    df.loc[mask_left, "y_coord"] = -df.loc[mask_left, "y_coord"]

    dx = NET_X - df["x_coord"]
    dy = NET_Y - df["y_coord"]

    df["shot_distance"] = np.sqrt(dx**2 + dy**2)
    df["shot_angle"] = np.degrees(np.arctan2(dy, dx))  # 角度，0 在中线方向

    # 是否进球
    df["is_goal"] = (df["eventType"] == "GOAL").astype(int)

    # 空网标记：NaN 当做 False
    if "emptyNet" in df.columns:
        df["empty_net"] = df["emptyNet"].fillna(False).astype(int)
    else:
        df["empty_net"] = 0

    # 返回一个“精简版”的射门表，后面 train.py 会用到这些列
    shot_cols = [
        "game_id",
        "season_start_year",
        "period",
        "periodTime",
        "team",
        "shotType",
        "strength",
        "shot_distance",
        "shot_angle",
        "is_goal",
        "empty_net",
    ]
    shot_cols = [c for c in shot_cols if c in df.columns]
    return df[shot_cols].copy()


def main():
    # 1) 读入所有赛季的 pbp
    all_pbp = load_all_seasons(SEASON_FILES)
    all_pbp = preprocess_pbp(all_pbp)

    # 2) 按 season_start_year 切训练/测试
    train_val_pbp = all_pbp[all_pbp["season_start_year"].between(2016, 2019)].copy()
    test_pbp = all_pbp[all_pbp["season_start_year"] == 2020].copy()

    # 3) 构建射门级别数据集
    shots_train = build_shot_dataframe(train_val_pbp)
    shots_test = build_shot_dataframe(test_pbp)

    SHOTS_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHOTS_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    shots_train.to_csv(SHOTS_TRAIN_PATH, index=False)
    shots_test.to_csv(SHOTS_TEST_PATH, index=False)

    print(f"训练射门数据保存到: {SHOTS_TRAIN_PATH}，共有 {len(shots_train)} 行")
    print(f"测试射门数据保存到: {SHOTS_TEST_PATH}，共有 {len(shots_test)} 行")


if __name__ == "__main__":
    main()

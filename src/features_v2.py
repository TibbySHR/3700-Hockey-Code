# src/features_v2.py
import numpy as np
import pandas as pd
from typing import Tuple


X_NET, Y_NET = 89.0, 0.0

EPS = 1e-6

SHOT_TYPES = {"Wrist Shot", "Slap Shot", "Snap Shot", "Backhand", "Tip-In", "Deflected", "Wrap-around"}

def ensure_game_seconds(df: pd.DataFrame, time_col: str = "period_time", period_col: str = "period") -> pd.Series:

    if "game_seconds" in df.columns:
        return df["game_seconds"].astype(float)
    # convert to seconds
    def mmss_to_sec(s: str) -> int:
        m, s = s.split(":")
        return int(m) * 60 + int(s)
    t = df[time_col].astype(str).apply(mmss_to_sec)
    
    return t + (df[period_col].astype(int) - 1) * 20 * 60

def shot_distance_angle(x: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
    dx, dy = x - X_NET, y - Y_NET
    dist = np.sqrt(dx * dx + dy * dy)
    angle = np.degrees(np.arctan2(dy, dx))  # [-180, 180]
    return dist, angle

def distance(a_x, a_y, b_x, b_y) -> pd.Series:
    dx, dy = (a_x - b_x), (a_y - b_y)
    return np.sqrt(dx * dx + dy * dy)

def build_last_event_features(df):
    # 自动适配列名：eventType 或 event_type
    evt_col = None
    if "event_type" in df.columns:
        evt_col = "event_type"
    elif "eventType" in df.columns:
        evt_col = "eventType"
    else:
        raise KeyError("CSV 缺少列 eventType / event_type，请检查输入数据")

    df = df.sort_values(["game_id", "event_idx"]).copy()

    df["last_event_type"] = df.groupby("game_id")[evt_col].shift(1)
    df["last_event_x"]    = df.groupby("game_id")["x"].shift(1)
    df["last_event_y"]    = df.groupby("game_id")["y"].shift(1)

    df["time_since_last_event"] = df.groupby("game_id")["game_seconds"].diff(1)
    df["dist_from_last_event"] = ((df["x"] - df["last_event_x"])**2 +
                                  (df["y"] - df["last_event_y"])**2)**0.5

    return df


def attach_shot_geofeatures(df: pd.DataFrame) -> pd.DataFrame:

    dist, angle = shot_distance_angle(df["x"].astype(float), df["y"].astype(float))
    df["shot_distance"] = dist
    df["shot_angle"] = angle
    return df

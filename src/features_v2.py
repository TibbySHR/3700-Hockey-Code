# src/features_v2.py modification 20260103
import numpy as np
import pandas as pd

NET_X, NET_Y = 89, 0

# Part4 rebound shot family
SHOT_FAMILY = {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}


def compute_shot_distance(df: pd.DataFrame) -> pd.Series:
    return np.hypot(df["x"] - NET_X, df["y"] - NET_Y)


def compute_shot_angle(df: pd.DataFrame) -> pd.Series:
    
    return np.degrees(np.arctan2(np.abs(df["y"] - NET_Y), np.abs(df["x"] - NET_X)))


def add_last_event_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Part4 Q2:
    IMPORTANT: has to be based on ALL events, then filter shots
    depend colums：game_id, event_idx, eventType, x, y, game_seconds
    output colums：
      last_eventType, last_x, last_y, last_game_seconds,
      time_since_last_event, dist_from_last_event
    """
    df = events_df.sort_values(["game_id", "event_idx"]).copy()
    g = df.groupby("game_id", sort=False)

    df["last_eventType"] = g["eventType"].shift(1)
    df["last_x"] = g["x"].shift(1)
    df["last_y"] = g["y"].shift(1)
    df["last_game_seconds"] = g["game_seconds"].shift(1)

    df["time_since_last_event"] = df["game_seconds"] - df["last_game_seconds"]
    df["dist_from_last_event"] = np.sqrt((df["x"] - df["last_x"]) ** 2 + (df["y"] - df["last_y"]) ** 2)

    return df


def add_rebound_speed(shots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Part4 Q3:
    - isRebound：last event 也是 shot family（shot-on-goal/goal/missed/blocked）
    - angle_change_on_rebound：only count when rebound, or 0
    - play_speed：dist_from_last_event / time_since_last_event（0 exclude）
    dependence：
      last_eventType, last_x, last_y, time_since_last_event, dist_from_last_event, shotAngle
    """
    df = shots_df.copy()

    df["isRebound"] = df["last_eventType"].isin(SHOT_FAMILY).astype(int)

    last_angle = np.degrees(
        np.arctan2(np.abs(df["last_y"] - NET_Y), np.abs(df["last_x"] - NET_X))
    )

    df["angle_change_on_rebound"] = np.abs(df["shotAngle"] - last_angle)
    df.loc[df["isRebound"] == 0, "angle_change_on_rebound"] = 0

    denom = df["time_since_last_event"].replace(0, np.nan)
    df["play_speed"] = df["dist_from_last_event"] / denom

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


def normalize_shot_type_col(df: pd.DataFrame) -> pd.DataFrame:
    
    out = df.copy()
    if "raw_shot_type" in out.columns:
        out["shotType"] = out["raw_shot_type"].fillna("Unknown").astype(str)
    else:
        # if no shotType，use "Unknown"，avoid taking eventType as shot type
        out["shotType"] = "Unknown"
    return out

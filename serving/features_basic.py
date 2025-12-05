import numpy as np
import pandas as pd


# 1) calculate shooting distance
def compute_shot_distance(df):
    
    net_x, net_y = 89, 0
    return np.sqrt((df["x"] - net_x) ** 2 + (df["y"] - net_y) ** 2)


# 2) shooting angle
def compute_shot_angle(df):
    
    net_x, net_y = 89, 0
    return np.degrees(np.arctan2(np.abs(df["y"] - net_y), np.abs(df["x"] - net_x)))


# 3) situationCode + eventOwnerTeamId for identifying empty net
def compute_empty_net(df):
    
    df = df.copy()

    if "situationCode" not in df.columns or "eventOwnerTeamId" not in df.columns:
        df["empty_net"] = 0
        return df

    sc = df["situationCode"].astype(str).str.zfill(4)
    df["away_goalie_on"] = sc.str[0].astype(int)
    df["home_goalie_on"] = sc.str[3].astype(int)

    #  home/away ID
    if "homeTeamId" in df.columns and "awayTeamId" in df.columns:
        df["is_home_owner"] = (df["eventOwnerTeamId"] == df["homeTeamId"]).astype(int)
        df["empty_net"] = np.where(
            df["is_home_owner"] == 1,
            1 - df["home_goalie_on"],  # home team, no goalie
            1 - df["away_goalie_on"],  # away team, no goalie
        )
    else:
        # no home/away ID, 0
        df["empty_net"] = 0

    df["empty_net"] = df["empty_net"].fillna(0).astype(int)
    return df


# 4) build basic features
def build_basic_features(df):
    df = df.copy()

    df["shot_distance"] = compute_shot_distance(df)
    df["shot_angle"] = compute_shot_angle(df)

    
    df["shot_type"] = df["secondaryType"]

    # GOAL → 1，other → 0
    df["is_goal"] = (df["event"].str.lower() == "goal").astype(int)

    # empty_net
    if "empty_net" in df.columns:
        df["empty_net"] = df["empty_net"].fillna(0).astype(int)
    else:
        df = compute_empty_net(df)

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
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()

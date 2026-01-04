# for feature engineer II

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


DATA_PATH = Path("data/processed/shots_train_v2.csv")
FIG_DIR = Path("figures")

sns.set(style="whitegrid")


def load_data():
    df = pd.read_csv(DATA_PATH)

    # rename columns to expected names for plotting
    df = df.rename(columns={
        "isGoal": "is_goal",
        "shotDistance": "shot_distance",
        "shotAngle": "shot_angle",
    })

    df["is_goal"] = df["is_goal"].astype(int)
    df["empty_net"] = df["empty_net"].astype(int)

    return df



# ---------- Q1:  histograms ----------

def plot_hist_by_distance(df):
    
    plt.figure(figsize=(8, 5))
    # goals
    sns.histplot(
        data=df,
        x="shot_distance",
        bins=40,
        hue="is_goal",
        stat="count",
        multiple="layer",  
        palette={0: "tab:blue", 1: "tab:orange"}
    )
    plt.xlabel("Shot distance")
    plt.ylabel("Number of shots")
    plt.title("Shot counts by distance (goals vs non-goals)")
    plt.legend(title="is_goal", labels=["Non-goals (0)", "Goals (1)"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outpath = FIG_DIR / "hist_distance_goals_vs_nongoals.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")


def plot_hist_by_angle(df):
    """histogram: by angles, differentiete goals or not"""
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="shot_angle",
        bins=40,
        hue="is_goal",
        stat="count",
        multiple="layer",
        palette={0: "tab:blue", 1: "tab:orange"}
    )
    plt.xlabel("Shot angle (degrees)")
    plt.ylabel("Number of shots")
    plt.title("Shot counts by angle (goals vs non-goals)")
    plt.legend(title="is_goal", labels=["Non-goals (0)", "Goals (1)"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outpath = FIG_DIR / "hist_angle_goals_vs_nongoals.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")


def plot_2d_hist_distance_angle(df):
    """2D histogram：dont different goals or not"""
    plt.figure(figsize=(7, 6))
    #  plt.hist2d or seaborn.jointplot
    plt.hist2d(
        df["shot_distance"],
        df["shot_angle"],
        bins=[40, 40],
    )
    plt.colorbar(label="Number of shots")
    plt.xlabel("Shot distance")
    plt.ylabel("Shot angle (degrees)")
    plt.title("2D histogram of shots (distance vs angle)")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outpath = FIG_DIR / "hist2d_distance_angle.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")


# ---------- Q2: goal rate vs distance / angle ----------

def compute_goal_rate_by_bin(values, is_goal, bins):
    
    cats = pd.cut(values, bins=bins, include_lowest=True)
    grouped = pd.DataFrame({"bin": cats, "is_goal": is_goal}).groupby("bin")
    counts = grouped["is_goal"].count()
    goals = grouped["is_goal"].sum()
    goal_rate = goals / counts.replace(0, np.nan)
   
    bin_centers = [interval.mid for interval in goal_rate.index.categories]
    return pd.DataFrame({
        "bin_center": bin_centers,
        "goal_rate": goal_rate.values,
        "count": counts.values,
    })


def plot_goal_rate_vs_distance(df):
    """goal rate (#goals / total shots) vs distance"""
    
    max_dist = df["shot_distance"].quantile(0.99)  # delete extreme vals
    bins = np.linspace(0, max_dist, 30)

    gr_df = compute_goal_rate_by_bin(df["shot_distance"], df["is_goal"], bins)

    plt.figure(figsize=(8, 5))
    plt.plot(gr_df["bin_center"], gr_df["goal_rate"], marker="o")
    plt.xlabel("Shot distance")
    plt.ylabel("Goal rate")
    plt.title("Goal rate vs distance")
    plt.ylim(0, 1)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outpath = FIG_DIR / "goal_rate_vs_distance.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")


def plot_goal_rate_vs_angle(df):
    """goal rate vs shot angle"""
   
    angle_min = df["shot_angle"].quantile(0.01)
    angle_max = df["shot_angle"].quantile(0.99)
    bins = np.linspace(angle_min, angle_max, 30)

    gr_df = compute_goal_rate_by_bin(df["shot_angle"], df["is_goal"], bins)

    plt.figure(figsize=(8, 5))
    plt.plot(gr_df["bin_center"], gr_df["goal_rate"], marker="o")
    plt.xlabel("Shot angle (degrees)")
    plt.ylabel("Goal rate")
    plt.title("Goal rate vs angle")
    plt.ylim(0, 1)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outpath = FIG_DIR / "goal_rate_vs_angle.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")


# ---------- Q3: Only goals，bin by distance + empty net seperation ----------

def plot_goal_hist_distance_by_empty_net(df):
    
    goals = df[df["is_goal"] == 1].copy()

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=goals,
        x="shot_distance",
        bins=40,
        hue="empty_net",
        stat="count",
        multiple="layer",
        palette={0: "tab:blue", 1: "tab:orange"}
    )
    plt.xlabel("Shot distance (goals only)")
    plt.ylabel("Number of goals")
    plt.title("Goals by distance (empty net vs non-empty net)")
    plt.legend(title="empty_net", labels=["Non-empty net (0)", "Empty net (1)"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outpath = FIG_DIR / "goals_by_distance_empty_vs_nonempty.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")

    # extra: get"suspicious events"
    # exemple：non-empty net & very long-distance goal（set threhold manually）
    suspicious = goals[(goals["empty_net"] == 0) & (goals["shot_distance"] > 80)]
    suspicious_path = FIG_DIR / "suspicious_long_distance_non_empty_goals.csv"
    suspicious.to_csv(suspicious_path, index=False)
    print(f"Saved potential suspicious events to: {suspicious_path}")


def main():
    df = load_data()
    print(f"Loaded {len(df)} shots from {DATA_PATH}")

    
    plot_hist_by_distance(df)
    plot_hist_by_angle(df)
    plot_2d_hist_distance_angle(df)

    # Q2: goal rate vs distance & angle
    plot_goal_rate_vs_distance(df)
    plot_goal_rate_vs_angle(df)

    # Q3: goals only, distance hist, empty net vs not
    plot_goal_hist_distance_by_empty_net(df)


if __name__ == "__main__":
    main()




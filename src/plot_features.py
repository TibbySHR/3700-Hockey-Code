# src/eda_part2_figures.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IN_CSV = "data/processed/shots_train_v2.csv"   # keep v2 as requested
OUT_DIR = "figures/part2_feature_eng1"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_CSV)

# Coerce numeric columns if present
for c in ["x","y","shotDistance","shotAngle","isGoal","empty_net","isRebound","play_speed"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

shots = df.copy()
shots = shots.dropna(subset=["shotDistance", "shotAngle"])  # required for Part 2 plots

# Common bins
dist_max = np.nanpercentile(shots["shotDistance"], 99)
ang_max  = np.nanpercentile(shots["shotAngle"], 99)

dist_bins = np.linspace(0, dist_max if dist_max > 0 else 1, 40)
ang_bins  = np.linspace(0, ang_max  if ang_max  > 0 else 1, 40)

# ---------- Part 2 Required Figures ----------

# (1) Histogram: number of shots binned by distance (goals vs non-goals)
fig = plt.figure()
plt.hist(shots.loc[shots["isGoal"]==1, "shotDistance"].dropna(), bins=dist_bins, alpha=0.6, label="Buts")
plt.hist(shots.loc[shots["isGoal"]==0, "shotDistance"].dropna(), bins=dist_bins, alpha=0.6, label="Non-buts")
plt.xlabel("Distance du tir")
plt.ylabel("Nombre de tirs")
plt.title("Histogramme des tirs par distance (buts vs non-buts)")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "P2_F1_hist_distance_goal_vs_nongoal.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# (2) Histogram: number of shots binned by angle (goals vs non-goals)
fig = plt.figure()
plt.hist(shots.loc[shots["isGoal"]==1, "shotAngle"].dropna(), bins=ang_bins, alpha=0.6, label="Buts")
plt.hist(shots.loc[shots["isGoal"]==0, "shotAngle"].dropna(), bins=ang_bins, alpha=0.6, label="Non-buts")
plt.xlabel("Angle du tir (degrés)")
plt.ylabel("Nombre de tirs")
plt.title("Histogramme des tirs par angle (buts vs non-buts)")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "P2_F2_hist_angle_goal_vs_nongoal.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# (3) 2D histogram: distance vs angle (all shots)
fig = plt.figure()
plt.hist2d(shots["shotDistance"], shots["shotAngle"], bins=[dist_bins, ang_bins])
plt.xlabel("Distance du tir")
plt.ylabel("Angle du tir (degrés)")
plt.title("Histogramme 2D: distance vs angle (tous les tirs)")
plt.colorbar(label="Comptes")
plt.savefig(os.path.join(OUT_DIR, "P2_F3_hist2d_distance_angle.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# Helper to compute binned goal rate
def binned_goal_rate(x, y_is_goal, bins):
    x = np.asarray(x)
    y = np.asarray(y_is_goal)
    idx = np.digitize(x, bins) - 1  # bin index
    centers = 0.5 * (bins[:-1] + bins[1:])
    rate = np.full(len(centers), np.nan, dtype=float)
    for i in range(len(centers)):
        mask = idx == i
        if mask.sum() > 0:
            rate[i] = y[mask].mean()
    return centers, rate

# (4) Goal rate vs distance
fig = plt.figure()
centers, rate = binned_goal_rate(shots["shotDistance"], shots["isGoal"].fillna(0).astype(int), dist_bins)
plt.plot(centers, rate, marker="o", linewidth=1)
plt.xlabel("Distance du tir")
plt.ylabel("Taux de buts (#buts / total)")
plt.title("Taux de buts en fonction de la distance (binned)")
plt.ylim(0, np.nanmax(rate) * 1.1 if np.isfinite(np.nanmax(rate)) else 1)
plt.savefig(os.path.join(OUT_DIR, "P2_F4_goal_rate_vs_distance.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# (5) Goal rate vs angle
fig = plt.figure()
centers, rate = binned_goal_rate(shots["shotAngle"], shots["isGoal"].fillna(0).astype(int), ang_bins)
plt.plot(centers, rate, marker="o", linewidth=1)
plt.xlabel("Angle du tir (degrés)")
plt.ylabel("Taux de buts (#buts / total)")
plt.title("Taux de buts en fonction de l'angle (binned)")
plt.ylim(0, np.nanmax(rate) * 1.1 if np.isfinite(np.nanmax(rate)) else 1)
plt.savefig(os.path.join(OUT_DIR, "P2_F5_goal_rate_vs_angle.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# (6) Sanity check: goals only, histogram by distance, empty-net vs non-empty-net
# (Assume NaN empty_net = 0 already in your pipeline; here we coerce)
if "empty_net" in shots.columns:
    goals = shots[shots["isGoal"] == 1].copy()
    goals["empty_net"] = goals["empty_net"].fillna(0).astype(int)

    fig = plt.figure()
    plt.hist(goals.loc[goals["empty_net"]==1, "shotDistance"].dropna(), bins=dist_bins, alpha=0.6, label="Filet vide")
    plt.hist(goals.loc[goals["empty_net"]==0, "shotDistance"].dropna(), bins=dist_bins, alpha=0.6, label="Filet non vide")
    plt.xlabel("Distance du tir (buts uniquement)")
    plt.ylabel("Nombre de buts")
    plt.title("Sanity check: buts par distance (filet vide vs non vide)")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "P2_F6_goals_hist_distance_empty_vs_nonempty.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

# ---------- Extra (your original plots, kept) ----------

# Extra A: scatter angle vs distance colored by isGoal
fig = plt.figure()
c = shots["isGoal"].fillna(0).astype(int)
plt.scatter(shots["shotDistance"], shots["shotAngle"], c=c, s=6)
plt.xlabel("Distance du tir")
plt.ylabel("Angle du tir (degrés)")
plt.title("Extra: angle vs distance (couleur = isGoal)")
plt.savefig(os.path.join(OUT_DIR, "EXTRA_scatter_angle_distance.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# Extra B: x–y hexbin shot density
if {"x","y"}.issubset(shots.columns):
    fig = plt.figure()
    xy = shots[["x","y"]].dropna()
    if len(xy) > 0:
        plt.hexbin(xy["x"], xy["y"], gridsize=40)
    plt.xlabel("x (glace)")
    plt.ylabel("y (glace)")
    plt.title("Extra: densité des tirs (hexbin x-y)")
    plt.savefig(os.path.join(OUT_DIR, "EXTRA_hexbin_xy.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

# Extra C: play_speed histogram (if columns exist)
if {"play_speed","isRebound"}.issubset(shots.columns):
    fig = plt.figure()
    ps = shots["play_speed"].replace([np.inf,-np.inf], np.nan).dropna()
    clip_v = np.nanpercentile(ps, 99) if len(ps) else 1.0
    reb = shots.loc[shots["isRebound"]==1, "play_speed"].replace([np.inf,-np.inf], np.nan).dropna().clip(upper=clip_v)
    non = shots.loc[shots["isRebound"]==0, "play_speed"].replace([np.inf,-np.inf], np.nan).dropna().clip(upper=clip_v)
    bins = np.linspace(0, clip_v if clip_v > 0 else 1, 40)
    plt.hist(reb, bins=bins, alpha=0.6, label="Rebonds")
    plt.hist(non, bins=bins, alpha=0.6, label="Non-rebonds")
    plt.xlabel("Vitesse du jeu (distance / temps)")
    plt.ylabel("Fréquence")
    plt.title("Extra: distribution de la vitesse (rebond vs non-rebond)")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "EXTRA_hist_play_speed.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

print("Figures saved to", OUT_DIR)


import wandb
from pathlib import Path

def wandb_log_figures(project: str, run_name: str, figures_dir: str):
    run = wandb.init(project=project, name=run_name, job_type="eda-part2")
    figs = sorted(Path(figures_dir).glob("*.png"))

   
    for fp in figs:
        run.log({fp.stem: wandb.Image(str(fp))})

    
    art = wandb.Artifact("part2_figures", type="figures")
    for fp in figs:
        art.add_file(str(fp), name=fp.name)
    run.log_artifact(art)

    run.finish()

# (project/run_name）
wandb_log_figures(
    project="IFT6758.2025-A-3700",
    run_name="Feature Engineering",
    figures_dir=OUT_DIR,
)

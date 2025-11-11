import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

IN_CSV = "data/processed/shots_train_v2.csv"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_CSV)
for c in ["x","y","shotDistance","shotAngle","isGoal","isRebound","play_speed"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

shots = df.copy()

# 1) 距离直方图（按是否进球）
fig = plt.figure()
bins = np.linspace(0, np.nanpercentile(shots["shotDistance"], 99), 40)
plt.hist(shots.loc[shots["isGoal"]==1, "shotDistance"].dropna(), bins=bins, alpha=0.6, label="Buts")
plt.hist(shots.loc[shots["isGoal"]==0, "shotDistance"].dropna(), bins=bins, alpha=0.6, label="Non-buts")
plt.xlabel("Distance du tir"); plt.ylabel("Fréquence")
plt.title("Distribution de la distance des tirs (buts vs non-buts)")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "fig1_hist_shot_distance.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

# 2) 角度×距离散点（颜色=是否进球）
fig = plt.figure()
c = shots["isGoal"].fillna(0).astype(int)
plt.scatter(shots["shotDistance"], shots["shotAngle"], c=c, s=6)
plt.xlabel("Distance du tir"); plt.ylabel("Angle du tir (degrés)")
plt.title("Angle vs Distance (couleur = isGoal)")
plt.savefig(os.path.join(OUT_DIR, "fig2_scatter_angle_distance.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

# 3) x–y hexbin
fig = plt.figure()
xy = shots[["x","y"]].dropna()
if len(xy) > 0:
    plt.hexbin(xy["x"], xy["y"], gridsize=40)
plt.xlabel("x (glace)"); plt.ylabel("y (glace)")
plt.title("Densité des tirs (hexbin)")
plt.savefig(os.path.join(OUT_DIR, "fig3_hexbin_xy.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

# 4) play_speed 直方图（按 isRebound）
fig = plt.figure()
ps = shots["play_speed"].replace([np.inf,-np.inf], np.nan).dropna()
clip_v = np.nanpercentile(ps, 99) if len(ps) else 1.0
reb = shots.loc[shots["isRebound"]==1, "play_speed"].replace([np.inf,-np.inf], np.nan).dropna().clip(upper=clip_v)
non = shots.loc[shots["isRebound"]==0, "play_speed"].replace([np.inf,-np.inf], np.nan).dropna().clip(upper=clip_v)
bins = np.linspace(0, clip_v if clip_v>0 else 1, 40)
plt.hist(reb, bins=bins, alpha=0.6, label="Rebonds")
plt.hist(non, bins=bins, alpha=0.6, label="Non-rebonds")
plt.xlabel("Vitesse du jeu (distance / temps)"); plt.ylabel("Fréquence")
plt.title("Distribution de la vitesse du jeu (rebond vs non-rebond)")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "fig4_hist_play_speed.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

print("Figures saved to", OUT_DIR)

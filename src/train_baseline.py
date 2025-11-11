import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import wandb

df = pd.read_csv("data/processed/shots_train_fixed.csv")
df = df[["shotDistance", "shotAngle", "isGoal"]].dropna()


X = df[["shotDistance", "shotAngle"]]
y = df["isGoal"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

def train_and_predict(features, label):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train[features], y_train)
    proba = model.predict_proba(X_val[features])[:, 1]
    auc = roc_auc_score(y_val, proba)
    return proba, auc, label

# Train 3 models
p_dist, auc_dist, label_dist = train_and_predict(["shotDistance"], "distance only")
p_ang, auc_ang, label_ang = train_and_predict(["shotAngle"], "angle only")
p_both, auc_both, label_both = train_and_predict(["shotDistance", "shotAngle"], "distance+angle")

# Random baseline
rand = np.random.rand(len(y_val))

# --- ROC Plot ---
plt.figure(figsize=(7,6))

for p, lab, auc in [(p_dist,label_dist,auc_dist),(p_ang,label_ang,auc_ang),(p_both,label_both,auc_both)]:
    fpr, tpr, _ = roc_curve(y_val, p)
    plt.plot(fpr, tpr, label=f"{lab} (AUC={auc:.3f})")

fpr, tpr, _ = roc_curve(y_val, rand)
plt.plot(fpr, tpr, "--", label="random baseline")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Comparison - Baseline Models")
plt.legend()
plt.savefig("figures/roc_baseline.png", dpi=300)
print("Saved ROC: figures/roc_baseline.png")

# --- Upload to WandB ---
run = wandb.init(project="IFT6758.2025-A-3700", name="baseline_models")
wandb.log({"roc_baseline": wandb.Image("figures/roc_baseline.png")})
run.finish()



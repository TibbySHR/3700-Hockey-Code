import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
import wandb


df = pd.read_csv("data/processed/shots_train_fixed.csv")
df = df[["shotDistance", "shotAngle", "isGoal"]].dropna()

X = df[["shotDistance", "shotAngle"]]
y = df["isGoal"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)


proba = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, proba)
print(f"AUC (distance + angle) : {auc:.3f}")


fpr, tpr, _ = roc_curve(y_val, proba)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"XGBoost (AUC={auc:.3f})")

rand = np.random.rand(len(y_val))
fpr_r, tpr_r, _ = roc_curve(y_val, rand)
plt.plot(fpr_r, tpr_r, "--", label="random baseline")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC - XGBoost (distance + angle)")
plt.legend()
plt.savefig("figures/roc_xgboost_baseline.png", dpi=300)
print("Saved ROC: figures/roc_xgboost_baseline.png")

run = wandb.init(project="IFT6758.2025-A-3700", name="xgboost_distance_angle")
wandb.log({"roc_xgboost_baseline": wandb.Image("figures/roc_xgboost_baseline.png")})
run.finish()

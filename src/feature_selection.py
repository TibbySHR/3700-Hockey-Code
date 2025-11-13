import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import wandb
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# creation des dossiers pour sauvegarder les figures et modeles
os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

# chargement des donnees
df = pd.read_csv("data/processed/shots_train_fixed.csv").dropna()
if "isGoal" not in df.columns:
    raise ValueError("La colonne 'isGoal' n'existe pas.")

X = df.drop(columns=["isGoal"])
y = df["isGoal"]

# conversion des colonnes categoricielles
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    X[col] = X[col].astype('category')

# separation train / validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# demarrage de Wandb
run = wandb.init(
    project="IFT6758.2025-A-3700",
    name="xgboost_feature_selection"
)

# modele de base pour la selection
xgb_base = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    enable_categorical=True,
    random_state=42
)

# selection des features importantes avec XGBoost
print("Selection des caracteristiques importantes...")
xgb_base.fit(X_train, y_train)

sfm = SelectFromModel(xgb_base, threshold="median")
support = sfm.get_support()
selected_features = X_train.columns[support]

# garder les DataFrames avec les categories pour XGBoost
X_train_sel = X_train[selected_features]
X_val_sel = X_val[selected_features]

print("Features retenues :", list(selected_features))

# entrainement avec hyperparam tuning rapide
param_grid = {
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 200],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

grid = GridSearchCV(
    estimator=XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        enable_categorical=True,
        random_state=42
    ),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    verbose=1
)

print("Entrainement du modele sur les features selectionnees...")
grid.fit(X_train_sel, y_train)
best_model = grid.best_estimator_

# prediction et evaluation
proba = best_model.predict_proba(X_val_sel)[:, 1]
auc = roc_auc_score(y_val, proba)
print(f"AUC sur validation: {auc:.3f}")

# sauvegarde des figures ROC
fpr, tpr, _ = roc_curve(y_val, proba)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}", linewidth=2)
plt.plot([0,1], [0,1], "--", label="baseline aleatoire", alpha=0.5)
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("ROC - XGBoost (features selectionnees)")
plt.legend()
plt.grid(True, alpha=0.3)
roc_path = "figures/roc_xgboost_features_sel.png"
plt.savefig(roc_path, dpi=300)
plt.close()
print("ROC sauvegardee :", roc_path)

# log sur Wandb
wandb.config.update({
    "features_sel": list(selected_features),
    "auc_validation": auc,
    "best_params": grid.best_params_
})

artifact = wandb.Artifact("xgboost_features_sel_model", type="model")
model_path = "models/xgboost_features_sel_model.json"
best_model.save_model(model_path)
artifact.add_file(model_path)
artifact.add_file(roc_path)
wandb.log_artifact(artifact)

run.finish()

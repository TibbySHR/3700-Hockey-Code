import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
import wandb
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Creation des dossiers pour sauvegarder les figures
os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Chargement des donnees
df = pd.read_csv("data/processed/shots_train_fixed.csv")
df = df.dropna()

if "isGoal" not in df.columns:
    raise ValueError("La colonne 'isGoal' n'existe pas.")

X = df.drop(columns=["isGoal"])
y = df["isGoal"]

# Conversion des colonnes categorielles
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    X[col] = X[col].astype('category')

# Separation train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Initialisation Wandb
run = wandb.init(
    project="IFT6758.2025-A-3700",
    name="xgboost_all_features_tuning"
)

# Grille d'hyperparametres pour le tuning
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 300],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

# Modele XGBoost de base
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    enable_categorical=True,  
    random_state=42
)

# Recherche par grille avec validation croisee
grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    verbose=1
)

print("Debut de l'entrainement avec GridSearchCV...")
grid.fit(X_train, y_train)

print("Meilleurs parametres:", grid.best_params_)
print("Best AUC (validation croisee):", grid.best_score_)

# Sauvegarde des parametres dans Wandb
wandb.config.update(grid.best_params_)

# Modele final avec les meilleurs parametres
best_model = grid.best_estimator_
proba = best_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, proba)
print(f"AUC sur validation (toutes features, tuned): {auc:.3f}")

# Fonction pour le taux de buts vs percentile
def plot_goal_rate_vs_percentile(y_true, y_pred, model_name):
    df_plot = pd.DataFrame({'true': y_true, 'pred': y_pred})
    df_plot['percentile'] = pd.qcut(df_plot['pred'], 100, labels=False, duplicates='drop')
    
    goal_rates = df_plot.groupby('percentile')['true'].mean()
    
    plt.figure(figsize=(8, 6))
    plt.plot(goal_rates.index, goal_rates.values, marker='o', markersize=2, linewidth=1)
    plt.xlabel('Percentile de probabilite')
    plt.ylabel('Taux de buts')
    plt.title(f'Taux de buts vs Percentile - {model_name}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"figures/goal_rate_percentile_{model_name}.png", dpi=300)
    plt.close()

# Fonction pour proportion cumulee de buts
def plot_cumulative_goals_vs_percentile(y_true, y_pred, model_name):
    df_plot = pd.DataFrame({'true': y_true, 'pred': y_pred})
    df_plot = df_plot.sort_values('pred', ascending=False)
    df_plot['cumulative_goals'] = df_plot['true'].cumsum()
    df_plot['cumulative_proportion'] = df_plot['cumulative_goals'] / df_plot['true'].sum()
    df_plot['percentile'] = range(1, len(df_plot) + 1)
    df_plot['percentile'] = (df_plot['percentile'] / len(df_plot)) * 100
    
    plt.figure(figsize=(8, 6))
    plt.plot(df_plot['percentile'], df_plot['cumulative_proportion'], linewidth=2)
    plt.plot([0, 100], [0, 1], 'k--', label='Ligne de base', alpha=0.5)
    plt.xlabel('Percentile de probabilite')
    plt.ylabel('Proportion cumulee de buts')
    plt.title(f'Proportion cumulee de buts - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"figures/cumulative_goals_{model_name}.png", dpi=300)
    plt.close()

# Fonction pour courbe de calibration
def plot_calibration_curve(y_true, y_pred, model_name):
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, 's-', label=model_name, linewidth=2, markersize=6)
    plt.plot([0, 1], [0, 1], 'k--', label='Parfaitement calibre', alpha=0.5)
    plt.xlabel('Probabilite predite (moyenne par bin)')
    plt.ylabel('Probabilite vraie')
    plt.title(f'Courbe de calibration - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"figures/calibration_{model_name}.png", dpi=300)
    plt.close()

# Generation des 4 figures requises

# 1. Courbe ROC
fpr, tpr, _ = roc_curve(y_val, proba)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"XGBoost Tune (AUC={auc:.3f})", linewidth=2)

rand = np.random.rand(len(y_val))
fpr_r, tpr_r, _ = roc_curve(y_val, rand)
plt.plot(fpr_r, tpr_r, "--", label="random baseline", alpha=0.7)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC - XGBoost (toutes les features)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("figures/roc_xgboost_tuned.png", dpi=300)
print("Saved ROC: figures/roc_xgboost_tuned.png")

# 2. Taux de buts vs percentile
plot_goal_rate_vs_percentile(y_val, proba, "XGBoost_tuned")
print("Saved Goal Rate vs Percentile: figures/goal_rate_percentile_XGBoost_tuned.png")

# 3. Proportion cumulee de buts
plot_cumulative_goals_vs_percentile(y_val, proba, "XGBoost_tuned")
print("Saved Cumulative Goals: figures/cumulative_goals_XGBoost_tuned.png")

# 4. Courbe de calibration
plot_calibration_curve(y_val, proba, "XGBoost_tuned")
print("Saved Calibration Curve: figures/calibration_XGBoost_tuned.png")

# Log toutes les figures dans Wandb
wandb.log({
    "roc_xgboost_tuned": wandb.Image("figures/roc_xgboost_tuned.png"),
    "goal_rate_percentile": wandb.Image("figures/goal_rate_percentile_XGBoost_tuned.png"),
    "cumulative_goals": wandb.Image("figures/cumulative_goals_XGBoost_tuned.png"),
    "calibration_curve": wandb.Image("figures/calibration_XGBoost_tuned.png"),
    "auc_validation": auc,
    "best_cv_score": grid.best_score_
})

# Sauvegarde du modele dans le registre Wandb
model_path = "models/xgboost_tuned_model.json"
best_model.save_model(model_path)

artifact = wandb.Artifact("xgboost_tuned_model", type="model")
artifact.add_file(model_path)  
artifact.add_file("figures/roc_xgboost_tuned.png")  
artifact.add_file("figures/goal_rate_percentile_XGBoost_tuned.png")
artifact.add_file("figures/cumulative_goals_XGBoost_tuned.png")
artifact.add_file("figures/calibration_XGBoost_tuned.png")

wandb.log_artifact(artifact)
print("Modele enregistre dans le registre Wandb")

run.finish()
print("Experiment Wandb termine")

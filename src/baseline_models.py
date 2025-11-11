# src/baseline_models.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve
)
from sklearn.calibration import CalibrationDisplay
import joblib
import wandb


COL_Y = "is_goal"
COL_DIST = "shot_distance"
COL_ANGLE = "shot_angle"
# ========================

FIG_DIR = Path("figures")
MODEL_DIR = Path("models")
WANDB_PROJECT = "IFT6758.2025-A-3700"   # WanDb project name

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if COL_Y not in df.columns:
        raise ValueError(f"Label column '{COL_Y}' not found.")
    return df

def split_train_val(df: pd.DataFrame, val_size=0.2, seed=42):
    y = df[COL_Y].astype(int)
    X = df[[COL_DIST, COL_ANGLE]].copy()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )
    return X_train, X_val, y_train, y_val

def fit_lr(X_train, y_train, features):
    """features is name of colums，exemple [COL_DIST] or[COL_ANGLE] 或 [COL_DIST, COL_ANGLE]"""
    clf = LogisticRegression()  
    clf.fit(X_train[features], y_train)
    return clf

def goal_rate_by_percentile(y_true, proba, n_bins=20):
    """(b)goal rate"""
    # percentile: proba
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(proba, qs)
    # avoid repeated exges
    edges = np.unique(edges)
    bins = np.digitize(proba, edges[1:-1], right=True)

    rates, counts, centers = [], [], []
    for b in range(len(edges) - 1):
        mask = (bins == b)
        if mask.sum() == 0:
            rates.append(np.nan); counts.append(0)
            centers.append((qs[b] + qs[b+1]) / 2)
            continue
        num_goals = y_true[mask].sum()
        total = mask.sum()
        rates.append(num_goals / total)
        counts.append(total)
        centers.append((qs[b] + qs[b+1]) / 2)
    return np.array(centers), np.array(rates), np.array(counts)

def cumulative_goals_by_percentile(y_true, proba, n_bins=100):
    """(c) cumulative goals vs percentile"""
    order = np.argsort(proba)
    y_sorted = y_true[order]
    cum_goals = np.cumsum(y_sorted)
    total_goals = y_true.sum()
    x_percentile = (np.arange(1, len(proba) + 1)) / len(proba)  # [0,1]
    frac = cum_goals / total_goals if total_goals > 0 else np.zeros_like(cum_goals)
    # n_bins 
    idx = np.linspace(0, len(proba) - 1, n_bins).astype(int)
    return x_percentile[idx], frac[idx]

def random_baseline_proba(n):
    return np.random.RandomState(0).rand(n)

def plot_roc(models, X_val, y_val, outpath):
    plt.figure(figsize=(7,5))
    # random baselines
    rand = random_baseline_proba(len(y_val))
    fpr, tpr, _ = roc_curve(y_val, rand)
    auc_rand = roc_auc_score(y_val, rand)
    plt.plot(fpr, tpr, linestyle="--", label=f"Basale aléatoire (AUC={auc_rand:.3f})")

    for label, clf, feats, color in models:
        proba = clf.predict_proba(X_val[feats])[:,1]
        fpr, tpr, _ = roc_curve(y_val, proba)
        auc = roc_auc_score(y_val, proba)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")

    plt.plot([0,1], [0,1], color="gray", linewidth=1)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("Courbes ROC (validation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150); plt.close()

def plot_goal_rate_percentile(models, X_val, y_val, outpath):
    plt.figure(figsize=(7,5))
    # 随机基线
    rand = random_baseline_proba(len(y_val))
    x, r, _ = goal_rate_by_percentile(y_val.values, rand, n_bins=20)
    plt.plot(x*100, r, linestyle="--", label="Basale aléatoire")

    for label, clf, feats, color in models:
        proba = clf.predict_proba(X_val[feats])[:,1]
        x, r, _ = goal_rate_by_percentile(y_val.values, proba, n_bins=20)
        plt.plot(x*100, r, label=label)

    plt.xlabel("Centile de probabilité (%)")
    plt.ylabel("Taux de but")
    plt.title("Taux de but vs centile de probabilité (validation)")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150); plt.close()

def plot_cumulative_goals(models, X_val, y_val, outpath):
    plt.figure(figsize=(7,5))
    # 随机基线
    rand = random_baseline_proba(len(y_val))
    x, frac = cumulative_goals_by_percentile(y_val.values, rand, n_bins=200)
    plt.plot(x*100, frac, linestyle="--", label="Basale aléatoire")

    for label, clf, feats, color in models:
        proba = clf.predict_proba(X_val[feats])[:,1]
        x, frac = cumulative_goals_by_percentile(y_val.values, proba, n_bins=200)
        plt.plot(x*100, frac, label=label)

    plt.xlabel("Centile de probabilité (%)")
    plt.ylabel("Proportion cumulée de buts")
    plt.title("Buts cumulés vs centile de probabilité (validation)")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150); plt.close()

def plot_calibration(models, X_val, y_val, outpath, n_bins=10):
    plt.figure(figsize=(7,5))
    
    rand = random_baseline_proba(len(y_val))
    CalibrationDisplay.from_predictions(y_val, rand, n_bins=n_bins, name="Basale aléatoire")

    for label, clf, feats, color in models:
        proba = clf.predict_proba(X_val[feats])[:,1]
        CalibrationDisplay.from_predictions(y_val, proba, n_bins=n_bins, name=label)

    plt.title("Diagrammes de fiabilité (validation)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150); plt.close()

def log_wandb_run(run_name, config, clf, X_val, y_val, feats):
    """each model WandB run：记录 accuracy / AUC，save model to artifact"""

    run = wandb.init(project=WANDB_PROJECT, job_type="train", name=run_name, config=config, tags=["baseline","logreg"])
    # metrics
    y_pred = clf.predict(X_val[feats])
    y_proba = clf.predict_proba(X_val[feats])[:,1]
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    pos_rate = y_val.mean()
    wandb.log({"val_accuracy": acc, "val_auc": auc, "val_pos_rate": pos_rate})

    # save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{run_name}.pkl"
    joblib.dump(clf, model_path)

    # artifact
    artifact = wandb.Artifact(name=run_name, type="model", description="Baseline logistic regression")
    artifact.add_file(str(model_path))
    run.log_artifact(artifact)
    run.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/processed/shots_train.csv")
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true", help="不记录 WandB（本地调试）")
    args = parser.parse_args()

    df = load_data(args.train_path)
    X_train, X_val, y_train, y_val = split_train_val(df, val_size=args.val_size, seed=args.seed)

    # 1) single model：distance
    clf_dist = fit_lr(X_train, y_train, [COL_DIST])
    # 2) single model：angle
    clf_angle = fit_lr(X_train, y_train, [COL_ANGLE])
    # 3) dual models：distance + angle
    clf_both = fit_lr(X_train, y_train, [COL_DIST, COL_ANGLE])

    # record“accuracy”（Q1）
    for name, clf, feats in [
        ("LR_distance", clf_dist, [COL_DIST]),
        ("LR_angle", clf_angle, [COL_ANGLE]),
        ("LR_both", clf_both, [COL_DIST, COL_ANGLE]),
    ]:
        y_pred = clf.predict(X_val[feats])
        acc = accuracy_score(y_val, y_pred)
        print(f"[{name}] val accuracy = {acc:.4f} | positive rate = {y_val.mean():.4f}")

    # WandB
    if not args.no_wandb:
        log_wandb_run("LR_distance_only", {"features":"distance"}, clf_dist, X_val, y_val, [COL_DIST])
        log_wandb_run("LR_angle_only", {"features":"angle"}, clf_angle, X_val, y_val, [COL_ANGLE])
        log_wandb_run("LR_distance_angle", {"features":"distance+angle"}, clf_both, X_val, y_val, [COL_DIST, COL_ANGLE])

    # draw 4 imgs（each 3models and baselines）
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    models = [
        ("LR distance", clf_dist, [COL_DIST], "C0"),
        ("LR angle", clf_angle, [COL_ANGLE], "C1"),
        ("LR distance+angle", clf_both, [COL_DIST, COL_ANGLE], "C2"),
    ]

    plot_roc(models, X_val, y_val, FIG_DIR / "baseline_roc.png")
    plot_goal_rate_percentile(models, X_val, y_val, FIG_DIR / "baseline_goal_rate_vs_percentile.png")
    plot_cumulative_goals(models, X_val, y_val, FIG_DIR / "baseline_cumulative_goals_vs_percentile.png")
    plot_calibration(models, X_val, y_val, FIG_DIR / "baseline_calibration.png")

    print("Figures saved in reports/figures:")
    for f in ["baseline_roc.png", "baseline_goal_rate_vs_percentile.png",
              "baseline_cumulative_goals_vs_percentile.png", "baseline_calibration.png"]:
        print(" -", FIG_DIR / f)

if __name__ == "__main__":
    main()

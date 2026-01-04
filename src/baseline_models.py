# src/baseline_models.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibrationDisplay

import joblib
import wandb


# ========================
# Column names in shots_train_v2.csv (from your build_shots_v2 pipeline)
COL_Y = "isGoal"
COL_DIST = "shotDistance"
COL_ANGLE = "shotAngle"
# ========================

FIG_DIR = Path("figures/part3_baselines")
MODEL_DIR = Path("models")
WANDB_PROJECT = "IFT6758.2025-A-3700"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [COL_Y, COL_DIST, COL_ANGLE] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}")
    # Coerce numeric
    for c in [COL_Y, COL_DIST, COL_ANGLE]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop rows with missing inputs/label
    df = df.dropna(subset=[COL_Y, COL_DIST, COL_ANGLE]).copy()
    df[COL_Y] = df[COL_Y].astype(int)
    return df


def split_train_val(df: pd.DataFrame, val_size=0.2, seed=42):
    y = df[COL_Y].astype(int)
    X = df[[COL_DIST, COL_ANGLE]].copy()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )
    return X_train, X_val, y_train, y_val


def fit_lr(X_train: pd.DataFrame, y_train: pd.Series, features):
    """
    features: list of column names, e.g. [COL_DIST] or [COL_ANGLE] or [COL_DIST, COL_ANGLE]
    """
    clf = LogisticRegression()  # default params as required
    clf.fit(X_train[features], y_train)
    return clf


def random_baseline_proba(n: int, seed: int = 0):
    # As required: y_hat ~ U(0, 1)
    return np.random.RandomState(seed).rand(n)


def goal_rate_by_percentile(y_true, proba, n_bins=20):
    """
    (b) Goal rate vs probability percentile.
    Robust implementation: sort by proba and split into equal-sized bins.
    Returns:
      centers in [0,1], rates, counts
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    order = np.argsort(proba)
    y_sorted = y_true[order]

    n = len(proba)
    edges = np.linspace(0, n, n_bins + 1).astype(int)

    centers, rates, counts = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        centers.append((i + 0.5) / n_bins)  # percentile center in [0,1]
        if hi <= lo:
            rates.append(np.nan)
            counts.append(0)
            continue
        ys = y_sorted[lo:hi]
        rates.append(float(np.mean(ys)))
        counts.append(int(len(ys)))

    return np.array(centers), np.array(rates), np.array(counts)


def cumulative_goals_by_percentile(y_true, proba, n_bins=200):
    """
    (c) Cumulative proportion of goals vs probability percentile.
    Sort by proba ascending; compute cumulative goals fraction.
    Return downsampled curve (n_bins points).
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    order = np.argsort(proba)
    y_sorted = y_true[order]

    cum_goals = np.cumsum(y_sorted)
    total_goals = np.sum(y_true)
    frac = cum_goals / total_goals if total_goals > 0 else np.zeros_like(cum_goals, dtype=float)

    x_percentile = (np.arange(1, len(proba) + 1)) / len(proba)  # (0,1]
    idx = np.linspace(0, len(proba) - 1, n_bins).astype(int)
    return x_percentile[idx], frac[idx]


def plot_roc(models, X_val, y_val, outpath):
    plt.figure(figsize=(7, 5))

    # Random baseline (U[0,1])
    rand = random_baseline_proba(len(y_val), seed=0)
    fpr, tpr, _ = roc_curve(y_val, rand)
    auc_rand = roc_auc_score(y_val, rand)
    plt.plot(fpr, tpr, linestyle="--", label=f"Baseline aléatoire (AUC={auc_rand:.3f})")

    for label, clf, feats in models:
        proba = clf.predict_proba(X_val[feats])[:, 1]
        fpr, tpr, _ = roc_curve(y_val, proba)
        auc = roc_auc_score(y_val, proba)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", linewidth=1)
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR)")
    plt.title("Courbes ROC (validation)")
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_goal_rate_percentile(models, X_val, y_val, outpath):
    plt.figure(figsize=(7, 5))

    # Random baseline
    rand = random_baseline_proba(len(y_val), seed=0)
    x, r, _ = goal_rate_by_percentile(y_val.values, rand, n_bins=20)
    plt.plot(x * 100, r, linestyle="--", label="Baseline aléatoire")

    for label, clf, feats in models:
        proba = clf.predict_proba(X_val[feats])[:, 1]
        x, r, _ = goal_rate_by_percentile(y_val.values, proba, n_bins=20)
        plt.plot(x * 100, r, label=label)

    plt.xlabel("Centile de probabilité (%)")
    plt.ylabel("Taux de but")
    plt.title("Taux de but vs centile de probabilité (validation)")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_cumulative_goals(models, X_val, y_val, outpath):
    plt.figure(figsize=(7, 5))

    # Random baseline
    rand = random_baseline_proba(len(y_val), seed=0)
    x, frac = cumulative_goals_by_percentile(y_val.values, rand, n_bins=200)
    plt.plot(x * 100, frac, linestyle="--", label="Baseline aléatoire")

    for label, clf, feats in models:
        proba = clf.predict_proba(X_val[feats])[:, 1]
        x, frac = cumulative_goals_by_percentile(y_val.values, proba, n_bins=200)
        plt.plot(x * 100, frac, label=label)

    plt.xlabel("Centile de probabilité (%)")
    plt.ylabel("Proportion cumulée de buts")
    plt.title("Buts cumulés vs centile de probabilité (validation)")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_calibration(models, X_val, y_val, outpath, n_bins=10):
    plt.figure(figsize=(7, 5))

    # Random baseline calibration
    rand = random_baseline_proba(len(y_val), seed=0)
    CalibrationDisplay.from_predictions(y_val, rand, n_bins=n_bins, name="Baseline aléatoire")

    for label, clf, feats in models:
        proba = clf.predict_proba(X_val[feats])[:, 1]
        CalibrationDisplay.from_predictions(y_val, proba, n_bins=n_bins, name=label)

    plt.title("Diagramme de fiabilité (calibration) — validation")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def log_wandb_run(run_name, config, clf, X_val, y_val, feats, fig_paths):
    """
    One model per W&B run:
      - logs val_accuracy, val_auc, class imbalance stats
      - saves model to local disk and logs as artifact
      - logs figures (the 4 Part 3 figures)
    """
    run = wandb.init(
        project=WANDB_PROJECT,
        job_type="train",
        name=run_name,
        config=config,
        tags=["baseline", "logreg"]
    )

    y_pred = clf.predict(X_val[feats])
    y_proba = clf.predict_proba(X_val[feats])[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    true_pos_rate = float(np.mean(y_val))
    pred_pos_rate = float(np.mean(y_pred))

    wandb.log({
        "val_accuracy": acc,
        "val_auc": auc,
        "val_true_pos_rate": true_pos_rate,
        "val_pred_pos_rate": pred_pos_rate,
    })

    # Save model to disk
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{run_name}.pkl"
    joblib.dump(clf, model_path)

    # Log artifact (model file)
    artifact = wandb.Artifact(
        name=f"model_{run_name}",
        type="model",
        description="Baseline LogisticRegression (scikit-learn)"
    )
    artifact.add_file(str(model_path))
    run.log_artifact(artifact)

    # Log figures (images)
    for key, fp in fig_paths.items():
        run.log({key: wandb.Image(str(fp))})

    run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/processed/shots_train_v2.csv")
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    args = parser.parse_args()

    df = load_data(args.train_path)
    X_train, X_val, y_train, y_val = split_train_val(df, val_size=args.val_size, seed=args.seed)

    # Train 3 baseline LRs
    clf_dist = fit_lr(X_train, y_train, [COL_DIST])
    clf_angle = fit_lr(X_train, y_train, [COL_ANGLE])
    clf_both = fit_lr(X_train, y_train, [COL_DIST, COL_ANGLE])

    # Q1: Accuracy check (distance-only, but we print for all 3 as useful diagnostics)
    for name, clf, feats in [
        ("LR_distance", clf_dist, [COL_DIST]),
        ("LR_angle", clf_angle, [COL_ANGLE]),
        ("LR_both", clf_both, [COL_DIST, COL_ANGLE]),
    ]:
        y_pred = clf.predict(X_val[feats])
        acc = accuracy_score(y_val, y_pred)
        true_pos_rate = float(np.mean(y_val))
        pred_pos_rate = float(np.mean(y_pred))
        print(f"[{name}] val acc={acc:.4f} | true_pos_rate={true_pos_rate:.4f} | pred_pos_rate={pred_pos_rate:.4f}")

    # Build model list for plotting (3 models)
    models = [
        ("LR distance", clf_dist, [COL_DIST]),
        ("LR angle", clf_angle, [COL_ANGLE]),
        ("LR distance+angle", clf_both, [COL_DIST, COL_ANGLE]),
    ]

    # Produce the 4 required figures (on validation set)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_paths = {
        "fig_roc": FIG_DIR / "baseline_roc.png",
        "fig_goal_rate": FIG_DIR / "baseline_goal_rate_vs_percentile.png",
        "fig_cum_goals": FIG_DIR / "baseline_cumulative_goals_vs_percentile.png",
        "fig_calibration": FIG_DIR / "baseline_calibration.png",
    }

    plot_roc(models, X_val, y_val, fig_paths["fig_roc"])
    plot_goal_rate_percentile(models, X_val, y_val, fig_paths["fig_goal_rate"])
    plot_cumulative_goals(models, X_val, y_val, fig_paths["fig_cum_goals"])
    plot_calibration(models, X_val, y_val, fig_paths["fig_calibration"])

    print(f"Figures saved in: {FIG_DIR}")
    for k, fp in fig_paths.items():
        print(" -", fp)

    # W&B: one run per model (as required for Part 3 Q4)
    if not args.no_wandb:
        log_wandb_run(
            run_name="part3_LR_distance_only",
            config={"features": [COL_DIST], "val_size": args.val_size, "seed": args.seed},
            clf=clf_dist, X_val=X_val, y_val=y_val, feats=[COL_DIST],
            fig_paths=fig_paths
        )
        log_wandb_run(
            run_name="part3_LR_angle_only",
            config={"features": [COL_ANGLE], "val_size": args.val_size, "seed": args.seed},
            clf=clf_angle, X_val=X_val, y_val=y_val, feats=[COL_ANGLE],
            fig_paths=fig_paths
        )
        log_wandb_run(
            run_name="part3_LR_distance_angle",
            config={"features": [COL_DIST, COL_ANGLE], "val_size": args.val_size, "seed": args.seed},
            clf=clf_both, X_val=X_val, y_val=y_val, feats=[COL_DIST, COL_ANGLE],
            fig_paths=fig_paths
        )

        print("✅ Logged 3 W&B runs (one per baseline LR model).")

    else:
        print("W&B logging disabled (--no_wandb).")


if __name__ == "__main__":
    main()

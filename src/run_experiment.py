# src/run_experiments.py
import subprocess
from src.experiments import EXPERIMENTS

TRAIN_PATH = "data/processed/shots_train.csv"  # real file path

for exp in EXPERIMENTS:
    cmd = [
        "python", "-m", "src.train",
        "--train_path", TRAIN_PATH,
        "--model_type", exp["model_type"],
        "--feature_set", exp["feature_set"],
        "--C", str(exp["C"]),
        "--seed", str(exp["seed"]),
        "--artifact_name", exp["name"],
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

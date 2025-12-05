# serving/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Global state
current_model = None
current_model_name = None
logs = []


def log(message: str) -> None:
    """Append log message with timestamp to in-memory log list."""
    ts = datetime.utcnow().isoformat()
    full_msg = f"[{ts}] {message}"
    print(full_msg)
    logs.append(full_msg)


def load_local_model(path: str):
    """Load a local sklearn model from disk."""
    model = joblib.load(path)
    return model


# Load a default model at startup

# DEFAULT_MODEL_PATH = "../models/LR_distance_only.pkl"

# DEFAULT_MODEL_PATH = "../models/LR_angle_only.pkl"

DEFAULT_MODEL_PATH = "../models/LR_distance_angle.pkl"


try:
    current_model = load_local_model(DEFAULT_MODEL_PATH)
    current_model_name = "logreg_distance"
    log(f"Loaded default model from {DEFAULT_MODEL_PATH}")
except Exception as e:
    log(f"Failed to load default model: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict goal probability for given input features."""
    global current_model, current_model_name

    if current_model is None:
        return jsonify({"error": "No model loaded"}), 500

    try:
        payload = request.get_json()

        # Build DataFrame from incoming JSON
        if isinstance(payload, dict):
            X = pd.DataFrame.from_dict(payload)
        elif isinstance(payload, list):
            X = pd.DataFrame(payload)
        else:
            return jsonify({"error": "Invalid JSON format"}), 400

        # Map camelCase columns to snake_case
        X.rename(columns={
            "shotDistance": "shot_distance",
            "shotAngle": "shot_angle",
        }, inplace=True)

        # Determine which features this model actually uses
        feature_names = getattr(current_model, "feature_names_in_", None)

        if feature_names is None:
            # Fallback: assume basic features
            feature_names = [c for c in ["shot_distance", "shot_angle"] if c in X.columns]

        used_cols = [c for c in feature_names if c in X.columns]

        if not used_cols:
            return jsonify({"error": "No valid feature columns found"}), 400

        # Select only the columns the model was trained on
        X_use = X[used_cols].astype(float)

        # Drop rows with NaN in any of these features
        before = len(X_use)
        X_use = X_use.fillna(X_use.mean())
        dropped = before - len(X_use)

        if len(X_use) == 0:
            return jsonify({"error": "All rows contain NaN in used features"}), 400

        proba = current_model.predict_proba(X_use)[:, 1]
        log(
            f"/predict called with {len(X_use)} rows "
            f"(dropped {dropped} rows with NaN), features={used_cols}"
        )

        return jsonify({
            "model": current_model_name,
            "n_samples": int(len(X_use)),
            "features": used_cols,
            "predictions": proba.tolist(),
        })

    except Exception as e:
        log(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500




@app.route("/logs", methods=["GET"])
def get_logs():
    """Return application logs."""
    return jsonify({"logs": logs})


######## Download wandb model

import wandb
import os
from pathlib import Path

WANDB_ENTITY = "haoran-sun-universite-de-montreal-"      
WANDB_PROJECT = "IFT6758.2025-A-3700"   
MODEL_DIR = Path("models")

def ensure_wandb_initialized():
    """Ensure there's an active wandb run before using artifacts."""
    if wandb.run is None:
        wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode="online", 
    )


def download_model_from_registry(model_name: str):
    """
    Download a model from Weights & Biases model registry.
    model_name example: "LR_distance_only:latest"
   or：
    "haoran-sun-universite-de-montreal-/IFT6758.2025-A-3700/LR_distance_only:latest"
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

   
    ensure_wandb_initialized()

  
    if "/" in model_name:
        full_name = model_name
    else:
        full_name = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{model_name}"

    log(f"Using WandB artifact: {full_name}")

    artifact = wandb.use_artifact(full_name, type="model")
    local_path = artifact.download(root=str(MODEL_DIR))


    for p in Path(local_path).glob("*.pkl"):
        log(f"Artifact downloaded to {p}")
        return p

    raise FileNotFoundError("No .pkl file found in artifact")



@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Download a model from WandB registry and switch current model if success.
    Expected JSON body: {"model_name": "REGISTRY_MODEL_NAME"}
    """
    global current_model, current_model_name

    try:
        data = request.get_json() or {}
        model_name = data.get("model_name")
        if not model_name:
            return jsonify({"error": "model_name is required"}), 400

        log(f"Requested model download: {model_name}")

        # Check if local cached model already exists
        local_cached = MODEL_DIR / (model_name.replace(":", "_") + ".pkl")
        if local_cached.exists():
            # Load from local cache
            new_model = load_local_model(str(local_cached))
            current_model = new_model
            current_model_name = model_name
            log(f"Switched to cached model {model_name}")
            return jsonify({"status": "ok", "source": "cache", "model": model_name})

        # Try to download from WandB
        try:
            p = download_model_from_registry(model_name)
            # Optionally rename to cache path
            os.replace(p, local_cached)
            new_model = load_local_model(str(local_cached))
            current_model = new_model
            current_model_name = model_name
            log(f"Downloaded and switched to model {model_name}")
            return jsonify({"status": "ok", "source": "wandb", "model": model_name})
        except Exception as e:
            log(f"Failed to download model {model_name} from WandB: {e}")
            # Keep current model alive
            return jsonify({
                "status": "failed",
                "error": str(e),
                "model_in_use": current_model_name
            }), 500

    except Exception as e:
        log(f"Error in /download_registry_model: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os

    port = int(os.getenv("SERVING_PORT", "8080"))
    
    app.run(host="0.0.0.0", port=port, debug=True)


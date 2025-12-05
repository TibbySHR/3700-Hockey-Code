import requests
import json
import pandas as pd
from typing import Any, Dict, List, Optional


class ServingClient:
    """
    A simple client wrapper to interact with the Flask prediction service.
    Provides convenient methods to call /predict, /logs, and /download_registry_model
    without manually writing HTTP requests every time.
    """

    def __init__(self, host: str = None, port: int = None):
        """
        Initialize the client with a host and port where the Flask service is running.

        By default:
        - outside Docker: uses 127.0.0.1:8080
        - inside docker-compose: you can set SERVING_HOST=serving, SERVING_PORT=8080
        """
        host = host or os.getenv("SERVING_HOST", "127.0.0.1")
        port = port or int(os.getenv("SERVING_PORT", "8080"))
        self.base_url = f"http://{host}:{port}"

    # -------------------------------------------------------------------------
    # /predict
    # -------------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Send input features to the Flask /predict endpoint.

        Args:
            X (pd.DataFrame): Input features (columns must match model expectations).

        Returns:
            Dict[str, Any]: JSON response from Flask containing predictions.
        """
        # Convert DataFrame to list-of-dict format for JSON serialization
        payload = json.loads(X.to_json(orient="records"))

        url = f"{self.base_url}/predict"
        resp = requests.post(url, json=payload)
        resp.raise_for_status()  # Raise error if HTTP status >= 400

        return resp.json()

    # -------------------------------------------------------------------------
    # /logs
    # -------------------------------------------------------------------------
    def logs(self) -> List[str]:
        """
        Retrieve all logs currently stored by the Flask app.

        Returns:
            List[str]: List of log entries (with timestamps).
        """
        url = f"{self.base_url}/logs"
        resp = requests.get(url)
        resp.raise_for_status()

        data = resp.json()
        return data.get("logs", [])

    # -------------------------------------------------------------------------
    # /download_registry_model
    # -------------------------------------------------------------------------
    def download_registry_model(self, model_name: str) -> Dict[str, Any]:
        """
        Trigger model download and switch from WandB registry via Flask endpoint.

        Args:
            model_name (str): Name of the model to download, e.g. "LR_distance_only:latest"

        Returns:
            Dict[str, Any]: JSON response with download status and model info.
        """
        url = f"{self.base_url}/download_registry_model"
        payload = {"model_name": model_name}

        resp = requests.post(url, json=payload)

        # Try to safely parse the response even if not valid JSON
        try:
            data = resp.json()
        except Exception:
            data = {"raw_text": resp.text}

        data["status_code"] = resp.status_code
        return data


# -------------------------------------------------------------------------
# Example usage (manual test)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Example test usage for local debugging
    client = ServingClient(host="127.0.0.1", port=8080)

    # Example DataFrame (replace with your real input)
    df = pd.DataFrame([
        {"shot_distance": 25.0, "shot_angle": 35.0},
        {"shot_distance": 10.0, "shot_angle": 5.0},
    ])

    print("== Predict ==")
    print(client.predict(df))

    print("\n== Logs ==")
    print(client.logs())

    print("\n== Download model ==")
    print(client.download_registry_model("LR_distance_only:latest"))

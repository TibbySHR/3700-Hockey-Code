import requests
import pandas as pd
import time
from typing import List, Dict, Any, Optional
from serving_client import ServingClient
from features_basic import build_basic_features


class GameClient:
    """
    Client for retrieving live NHL play-by-play data and sending new events
    to the Flask prediction service for xG (expected goals) predictions.
    """

    def __init__(self, game_id: str, serving_client: ServingClient):
        """
        Args:
            game_id (str): NHL game ID (e.g., "2022030411")
            serving_client (ServingClient): Client for interacting with Flask service
        """
        self.game_id = game_id
        self.serving_client = serving_client
        self.last_event_idx = -1   # Tracker for last processed event index
        self.cached_events: List[Dict[str, Any]] = []  # Store previously seen events

    # -------------------------------------------------------------------------
    # Fetch all events for a given game
    # -------------------------------------------------------------------------
    def fetch_all_events(self) -> List[Dict[str, Any]]:
        """
        Fetch play-by-play data for the given game_id from the NHL public API.

        Returns:
            List[Dict[str, Any]]: List of event dictionaries
        """
        url = f"https://api-web.nhle.com/v1/gamecenter/{self.game_id}/play-by-play"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        # Depending on the NHL API, the event list is typically stored in "plays"
        events = data.get("plays", [])
        return events

    # -------------------------------------------------------------------------
    # Get new events (not processed before)
    # -------------------------------------------------------------------------
    def get_new_events(self, all_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter and return only events that were not processed before.
        """
        start_idx = self.last_event_idx + 1
        if start_idx >= len(all_events):
            return []
        new_events = all_events[start_idx:]
        return new_events

    # -------------------------------------------------------------------------
    # Process new events and send for prediction
    # -------------------------------------------------------------------------
    def process_new_events(self) -> pd.DataFrame:
        """
        Fetch the latest play-by-play data, extract new events, compute features,
        and send predictions to the Flask service.
        """
        all_events = self.fetch_all_events()
        new_events = self.get_new_events(all_events)

        if not new_events:
            print("No new events since last check.")
            return pd.DataFrame()

        df_new = pd.DataFrame(new_events)

        # --- Filtering relevant shot events (compatible with all versions) ---
        if "event" in df_new.columns:
            df_new = df_new[df_new["event"].isin(["Shot", "Goal"])]
        elif "typeDescKey" in df_new.columns:
            df_new = df_new[df_new["typeDescKey"].isin(["shot-on-goal", "goal"])]
        elif "typeCode" in df_new.columns:
            df_new = df_new[df_new["typeCode"].isin(["SHOT", "GOAL"])]
        else:
            print("No recognizable event column found in API response.")
            self.last_event_idx = len(all_events) - 1
            return pd.DataFrame()

        if df_new.empty:
            print("No new shot events to process.")
            self.last_event_idx = len(all_events) - 1
            return pd.DataFrame()

        # --- Extract coordinates, team info, and shot type from "details" dict ---
        if "details" in df_new.columns:
            df_new["x"] = df_new["details"].apply(
                lambda d: d.get("xCoord") if isinstance(d, dict) else None
            )
            df_new["y"] = df_new["details"].apply(
                lambda d: d.get("yCoord") if isinstance(d, dict) else None
            )
            df_new["eventOwnerTeamId"] = df_new["details"].apply(
                lambda d: d.get("eventOwnerTeamId") if isinstance(d, dict) else None
            )
            df_new["secondaryType"] = df_new["details"].apply(
                lambda d: d.get("shotType") if isinstance(d, dict) else None
            )
        
        # --- Normalize event column for compatibility ---
        if "event" not in df_new.columns:
            if "typeDescKey" in df_new.columns:
                df_new["event"] = df_new["typeDescKey"]
            elif "typeCode" in df_new.columns:
                df_new["event"] = df_new["typeCode"]
            else:
                df_new["event"] = None
        
        # --- Build features ---
        df_features = build_basic_features(df_new)


        # --- Select only required columns for the model ---
        model_features = [c for c in ["shot_distance", "shot_angle"] if c in df_features.columns]
        if not model_features:
            print("No valid model features found.")
            return pd.DataFrame()

        X = df_features[model_features]

        # --- Call Flask prediction service ---
        try:
            pred_result = self.serving_client.predict(X)
            probs = pred_result.get("predictions", [])
        except Exception as e:
            print(f"Prediction service error: {e}")
            probs = [None] * len(X)

        # --- Combine predictions with event info ---
        df_features["goal_probability"] = probs

        # --- Update last processed index ---
        self.last_event_idx = len(all_events) - 1

        print(f"Processed {len(df_features)} new shot events.")
        cols = [c for c in ["event", "typeDescKey", "x", "y", "shot_distance", "shot_angle", "goal_probability"] if c in df_features.columns]
        return df_features[cols]

    # -------------------------------------------------------------------------
    # Continuous monitoring loop (optional)
    # -------------------------------------------------------------------------
    def run_live_tracking(self, interval_sec: int = 10):
        """
        Continuously poll the NHL API and update predictions for new events.
        """
        print(f"Starting live tracking for game {self.game_id} ...")
        while True:
            try:
                df = self.process_new_events()
                if not df.empty:
                    print(df.tail())
            except Exception as e:
                print(f"Error during live tracking: {e}")

            time.sleep(interval_sec)


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize Flask serving client (ensure your Flask app is running)
    serving_client = ServingClient(host="127.0.0.1", port=8080)

    # Example: any valid NHL game ID, e.g., "2022030411"
    game_client = GameClient(game_id="2022030411", serving_client=serving_client)

    # Run one-time update (no loop)
    df_latest = game_client.process_new_events()
    print(df_latest)

    # Or run continuous live polling
    # game_client.run_live_tracking(interval_sec=15)

import json
import requests
import pandas as pd
import streamlit as st

from serving_client import ServingClient
from game_client import GameClient



from serving_client import ServingClient


# -----------------------------------------------------------------------------
# init session_state
# -----------------------------------------------------------------------------
def init_session_state():
    

    if "serving_client" not in st.session_state:
        
        st.session_state.serving_client = ServingClient()

    if "game_id" not in st.session_state:
        st.session_state.game_id = None

    if "game_client" not in st.session_state:
        st.session_state.game_client = None

    # all predicted shot（include goal_probability）
    if "events_df" not in st.session_state:
        st.session_state.events_df = pd.DataFrame()

    # model selection
    if "workspace" not in st.session_state:
        st.session_state.workspace = ""
    if "model_name" not in st.session_state:
        st.session_state.model_name = ""
    if "version" not in st.session_state:
        st.session_state.version = ""


def model_sidebar():
    st.sidebar.header("Model selection & logs")

   
    st.sidebar.text_input(
        "W&B workspace (entity/project)",
        value="haoran-sun-universite-de-montreal-/IFT6758.2025-A-3700",
        disabled=True,
    )

    model_options = ["LR_distance_only", "LR_angle_only", "LR_distance_angle"]
    selected_models = st.sidebar.multiselect(
        "Select models to download",
        options=model_options,
        default=["LR_distance_angle"],
        help="You can select one or more models to fetch from the registry.",
    )

    version = st.sidebar.text_input(
        "Version",
        value="latest",
        help="Example: v1 or latest",
    )

    if st.sidebar.button("Download & switch selected models"):
        client: ServingClient = st.session_state.serving_client
        for model_name in selected_models:
            try:
                model_id = f"{model_name}:{version}"
                result = client.download_registry_model(model_id)
                status_code = result.get("status_code", "?")
                st.sidebar.success(
                    f"Downloaded {model_id} (status {status_code})"
                )
            except Exception as e:
                st.sidebar.error(f"Error downloading {model_name}: {e}")

    if st.sidebar.button("Show service logs"):
        try:
            logs = st.session_state.serving_client.logs()
            st.sidebar.text_area("Logs", value="\n".join(logs), height=200)
        except Exception as e:
            st.sidebar.error(f"Error fetching logs: {e}")


# -----------------------------------------------------------------------------
def fetch_game_meta(game_id: str) -> dict:
    """
   
    - home_team_name / away_team_name
    - home_team_id / away_team_id
    - home_score / away_score
    - period / time_remaining
    """
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    meta = {}

    # 1) analyse team
    home = data.get("homeTeam", {}) or data.get("home_team", {})
    away = data.get("awayTeam", {}) or data.get("away_team", {})

  
    def _get_team_name(team_obj):
        if not isinstance(team_obj, dict):
            return None
        name = team_obj.get("name", {})
        place = team_obj.get("placeName", {})
        if isinstance(name, dict):
            return name.get("default") or name.get("full") or name.get("short")
        if isinstance(place, dict):
            return place.get("default")
       
        return team_obj.get("name") or team_obj.get("teamName")

    meta["home_team_name"] = _get_team_name(home) or "Home"
    meta["away_team_name"] = _get_team_name(away) or "Away"
    meta["home_team_id"] = home.get("id")
    meta["away_team_id"] = away.get("id")

    # 2) analyse scores
    meta["home_score"] = home.get("score") or data.get("homeTeamScore") or 0
    meta["away_score"] = away.get("score") or data.get("awayTeamScore") or 0

    # 3) period & time remaining
    clock = data.get("clock", {}) or data.get("gameClock", {})
    period = None
    time_remaining = None

    if isinstance(clock, dict):
        # period in period or periodDescriptor
        period = clock.get("period")
        if isinstance(clock.get("periodDescriptor"), dict):
            period = (
                clock["periodDescriptor"].get("number")
                or clock["periodDescriptor"].get("ordinalNum")
                or period
            )
        # time remaining
        time_remaining = (
            clock.get("timeRemaining")
            or clock.get("timeRemainingFormatted")
            or clock.get("timeInPeriod")
        )

    meta["period"] = period or "N/A"
    meta["time_remaining"] = time_remaining or "N/A"

    return meta


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def display_summary_and_table(meta: dict, events_df: pd.DataFrame):
    st.subheader(f"{meta['home_team_name']} vs {meta['away_team_name']}")

    # period / time remaining / score
    col1, col2, col3 = st.columns(3)
    col1.metric("Période", meta["period"])
    col2.metric("Temps restant", meta["time_remaining"])
    col3.metric("Score actuel", f"{meta['home_score']} - {meta['away_score']}")

    
    home_xg = None
    away_xg = None
    if "eventOwnerTeamId" in events_df.columns:
        home_id = meta["home_team_id"]
        away_id = meta["away_team_id"]

       
        home_xg = events_df.loc[
            events_df["eventOwnerTeamId"] == home_id, "goal_probability"
        ].sum()
        away_xg = events_df.loc[
            events_df["eventOwnerTeamId"] == away_id, "goal_probability"
        ].sum()
    else:
        
        home_xg = events_df["goal_probability"].sum()
        away_xg = 0.0

    
    home_diff = home_xg - meta["home_score"]
    away_diff = away_xg - meta["away_score"]

    summary_df = pd.DataFrame(
        {
            "Team": [meta["home_team_name"], meta["away_team_name"]],
            "Goals": [meta["home_score"], meta["away_score"]],
            "xG sum": [home_xg, away_xg],
            "xG - Goals": [home_diff, away_diff],
        }
    )
    st.table(summary_df)

    st.subheader("Shot events with predicted goal probability")
    st.dataframe(events_df)


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def game_section():
    st.header("Live xG Dashboard")

    game_id_input = st.text_input(
        "Game ID (ex: 2022030411)",
        value=st.session_state.game_id or "",
    )

    if st.button("Requête jeu"):
        if not game_id_input:
            st.warning("Veuillez entrer un identifiant de jeu.")
            return

      
        if st.session_state.game_id != game_id_input:
            st.session_state.game_id = game_id_input
            st.session_state.events_df = pd.DataFrame()
            st.session_state.game_client = GameClient(
                game_id=game_id_input,
                serving_client=st.session_state.serving_client,
            )
        elif st.session_state.game_client is None:
            
            st.session_state.game_client = GameClient(
                game_id=game_id_input,
                serving_client=st.session_state.serving_client,
            )

        game_client: GameClient = st.session_state.game_client

    
        try:
            df_new = game_client.process_new_events()
        except Exception as e:
            st.error(f"Error when processing new events: {e}")
            return

        if df_new.empty:
            st.info("Aucun nouveau tir depuis la dernière requête.")
        else:
            
            if st.session_state.events_df.empty:
                st.session_state.events_df = df_new
            else:
                combined = pd.concat(
                    [st.session_state.events_df, df_new], ignore_index=True
                )
              
                for key in ["eventIndex", "eventId", "idx"]:
                    if key in combined.columns:
                        combined = combined.drop_duplicates(subset=[key])
                        break
                st.session_state.events_df = combined

      
        if not st.session_state.events_df.empty:
            try:
                meta = fetch_game_meta(st.session_state.game_id)
                display_summary_and_table(meta, st.session_state.events_df)
            except Exception as e:
                st.error(f"Error fetching game meta or displaying summary: {e}")
                st.dataframe(st.session_state.events_df)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    init_session_state()
    model_sidebar()
    game_section()


if __name__ == "__main__":
    main()

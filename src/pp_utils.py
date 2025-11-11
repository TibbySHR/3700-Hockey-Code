# src/pp_utils.py
import pandas as pd
import numpy as np


def compute_powerplay_features(events: pd.DataFrame) -> pd.DataFrame:
    
    df = events.sort_values(["game_id", "event_idx"]).copy()
    # staart：5v5
    df["home_skaters"] = 5
    df["away_skaters"] = 5
    df["pp_elapsed"] = 0.0

    
    out_rows = []
    for gid, g in df.groupby("game_id", sort=False):
        g = g.copy()
        # track penalties：[(team, expire_time)]
        active_pens = []
        last_time = 0.0
        pp_timer = 0.0      # count seconds for adv teaams
        adv_team = None     # 

        # adv team names
        home = g["home_team"].iloc[0]
        away = g["away_team"].iloc[0]

        # current skaters
        home_sk, away_sk = 5, 5

        for i, row in g.iterrows():
            t = float(row["game_seconds"])
            # clean outdated penalties first
            active_pens = [(tm, exp) for (tm, exp) in active_pens if exp > t]

            # based on active_pens recalculate on-ice skaters
            home_sk = 5 - sum(1 for (tm, exp) in active_pens if tm == home)
            away_sk = 5 - sum(1 for (tm, exp) in active_pens if tm == away)
            home_sk = max(home_sk, 3)  # 不少于3（合理边界）
            away_sk = max(away_sk, 3)

            # calculate pp_elapsed cumulation
            if adv_team is None:
                
                if home_sk > away_sk:
                    adv_team = home
                elif away_sk > home_sk:
                    adv_team = away
            dt = t - last_time
            if adv_team is not None and dt > 0:
                pp_timer += dt
            last_time = t

           
            g.at[i, "home_skaters"] = home_sk
            g.at[i, "away_skaters"] = away_sk
            g.at[i, "pp_elapsed"] = pp_timer if adv_team is not None else 0.0

            # if it's penalty, add to active_pens
            evt_col = "eventType" if "eventType" in df.columns else "event_type"
            if row[evt_col] == "PENALTY":

                minutes = row.get("penaltyMinutes", 2)
                expire = t + int(minutes) * 60
                # in penalty: row["team"]
                pen_team = row["team"]
                active_pens.append((pen_team, expire))

            
            if home_sk > away_sk:
                adv_team = home
            elif away_sk > home_sk:
                adv_team = away
            else:
                adv_team = None
                pp_timer = 0.0  # clear adv timings

        out_rows.append(g)

    return pd.concat(out_rows, axis=0)






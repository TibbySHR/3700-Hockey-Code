import argparse, json, glob
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- 解析与抽取 ----------

def extract_event(play, game_id, home_team=None, away_team=None):
    # 事件类型（兼容 gamecenter / statsapi）
    event_type = (
        play.get("typeDescKey")
        or play.get("result", {}).get("eventTypeId")
        or play.get("result", {}).get("event")
        or play.get("eventType")
    )

    # 坐标
    x = (play.get("xCoord") or play.get("coordinates", {}).get("x")
         or play.get("details", {}).get("xCoord"))
    y = (play.get("yCoord") or play.get("coordinates", {}).get("y")
         or play.get("details", {}).get("yCoord"))

    # 节与节内时间
    period = (
        play.get("period")
        or play.get("about", {}).get("period")
        or play.get("periodDescriptor", {}).get("number")
    )
    period_time = (
        play.get("periodTime")
        or play.get("about", {}).get("periodTime")
        or play.get("timeInPeriod")
    )

    empty_net = (
        play.get("emptyNet")
        or play.get("result", {}).get("emptyNet")
        or play.get("details", {}).get("emptyNet")
        or 0
    )

    return {
        "game_id": str(game_id),
        "period": int(period) if period is not None else np.nan,
        "period_time": period_time,
        "eventType": event_type,
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
        "empty_net": int(bool(empty_net)),
        "home_team": home_team,
        "away_team": away_team,
    }

def parse_game_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    game_id = (
        data.get("gameId")
        or data.get("gamePk")
        or data.get("game", {}).get("pk")
        or Path(path).stem
    )

    plays = (
        data.get("plays")
        or data.get("liveData", {}).get("plays", {}).get("allPlays")
        or data.get("allPlays")
        or []
    )

    home_team = (
        data.get("homeTeam", {}).get("abbrev")
        or data.get("gameData", {}).get("teams", {}).get("home", {}).get("name")
    )
    away_team = (
        data.get("awayTeam", {}).get("abbrev")
        or data.get("gameData", {}).get("teams", {}).get("away", {}).get("name")
    )

    rows = [extract_event(p, game_id, home_team, away_team) for p in plays]
    df = pd.DataFrame(rows)

    # 事件顺序与 event_idx（每场内累加）
    if not df.empty:
        # 排序：尽量使用 period + period_time 的自然顺序
        if "period_time" in df.columns:
            df["_mm"] = df["period_time"].astype(str).str.split(":").str[0].astype("Int64")
            df["_ss"] = df["period_time"].astype(str).str.split(":").str[1].astype("Int64")
            df = df.sort_values(["period", "_mm", "_ss"], na_position="last")
            df.drop(columns=["_mm","_ss"], errors="ignore", inplace=True)
        else:
            df = df.sort_values(["period"], na_position="last")
        df["event_idx"] = range(len(df))
    else:
        df["event_idx"] = []

    return df

# ---------- game_seconds 计算（稳健法） ----------

def mmss_to_sec(s: str) -> int:
    m, ss = str(s).split(":")
    return int(m) * 60 + int(ss)

def compute_game_seconds(df: pd.DataFrame,
                         game_col="game_id",
                         period_col="period",
                         time_col="period_time",
                         out_col="game_seconds") -> pd.Series:
    if time_col not in df.columns:
        raise ValueError(f"missing column: {time_col}")

    # 允许缺失，能转的转
    tsec = df[time_col].astype(str)
    mask_valid = tsec.str.contains(":", na=False)
    tsec = tsec.where(mask_valid, None)
    tsec = tsec.dropna().map(mmss_to_sec)
    tsec_full = pd.Series(np.nan, index=df.index, dtype="float")
    tsec_full.loc[tsec.index] = tsec.values

    tmp = df.copy()
    tmp["_tsec_"] = tsec_full

    # 每场每节最大节内秒数作为该节时长
    per_len = (tmp
               .groupby([game_col, period_col], dropna=False, as_index=False)["_tsec_"]
               .max()
               .rename(columns={"_tsec_": "_period_len_"}))

    # 每场内按 period 升序，累加偏移
    per_len = per_len.sort_values([game_col, period_col])
    per_len["_period_offset_"] = per_len.groupby(game_col)["_period_len_"].cumsum().shift(fill_value=0)

    merged = df[[game_col, period_col]].merge(
        per_len[[game_col, period_col, "_period_offset_"]],
        on=[game_col, period_col], how="left"
    )

    offsets = merged["_period_offset_"].astype("float")
    game_seconds = offsets.values + tsec_full.values
    return pd.Series(game_seconds, index=df.index, name=out_col)

# ---------- 主流程 ----------

def process_year(year: int, root="data/raw", out_dir="data/clean") -> dict:
    """处理某一年的 type-02（常规赛）所有 JSON，输出 year 级 CSV 和带 game_seconds 的 CSV"""
    in_glob = f"{root}/{year}/type-02/*.json"
    files = sorted(glob.glob(in_glob))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    frames = []
    for fp in files:
        try:
            frames.append(parse_game_json(fp))
        except Exception as e:
            print(f"[WARN] skip {fp}: {e}")

    if not frames:
        print(f"[WARN] {year}: no JSON matched {in_glob}")
        df = pd.DataFrame(columns=["game_id","period","period_time","eventType","x","y","empty_net","home_team","away_team","event_idx"])
    else:
        df = pd.concat(frames, ignore_index=True)

    # 保存 year 事件级
    out_csv = f"{out_dir}/events_{year}_type02.csv"
    df.to_csv(out_csv, index=False)

    # 计算 game_seconds
    if not df.empty:
        try:
            df["game_seconds"] = compute_game_seconds(df)
        except Exception as e:
            print(f"[WARN] {year}: game_seconds failed ({e}); fill NaN")
            df["game_seconds"] = np.nan
    else:
        df["game_seconds"] = []

    out_csv_gsec = f"{out_dir}/events_{year}_type02_with_gsec.csv"
    df.to_csv(out_csv_gsec, index=False)

    print(f"[OK] {year}: events={len(df)}  -> {out_csv_gsec}")
    return {"year": year, "path": out_csv_gsec, "rows": len(df)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2016,2017,2018,2019])
    ap.add_argument("--raw_root", default="data/raw")
    ap.add_argument("--clean_dir", default="data/clean")
    ap.add_argument("--combined_out", default="data/clean/events_2016_2019_with_gsec.csv")
    args = ap.parse_args()

    produced = []
    for y in args.years:
        produced.append(process_year(y, root=args.raw_root, out_dir=args.clean_dir))

    # 合并
    paths = [p["path"] for p in produced if p["rows"] > 0]
    if paths:
        frames = [pd.read_csv(p) for p in paths]
        all_df = pd.concat(frames, ignore_index=True)
    else:
        all_df = pd.DataFrame()

    Path(args.combined_out).parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.combined_out, index=False)
    print(f"[OK] combined saved -> {args.combined_out} | rows={len(all_df)}")

if __name__ == "__main__":
    main()

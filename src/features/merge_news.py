from __future__ import annotations
import pathlib
from typing import List
import pandas as pd
from loguru import logger
import yaml

RAW_DIR = pathlib.Path("data/raw")
RAW_NEWS_DIR = pathlib.Path("data/raw_news")
FEAT_DIR = pathlib.Path("data/features")
OUT_DIR = FEAT_DIR  # ugyanoda írjuk

NEWS_COLS = ["sent_mean", "sent_pos_ratio", "headline_cnt"]

def _list_latest_news() -> pathlib.Path | None:
    files = sorted(RAW_NEWS_DIR.glob("news_*.parquet"))
    return files[-1] if files else None

def _load_features(asset: str, tf: str) -> pd.DataFrame:
    p = FEAT_DIR / f"{asset}_{tf}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df

def _agg_news(df_news: pd.DataFrame, window_hours: int = 24) -> pd.DataFrame:
    if df_news.empty:
        return pd.DataFrame(columns=["time", "asset", *NEWS_COLS])

    df = df_news.copy().sort_values("time")
    df.set_index("time", inplace=True)
    out = []
    for asset, g in df.groupby("asset"):
        g = g[["score"]].copy()
        hourly = g.resample("1h").agg(
            sent_mean=("score", "mean"),
            sent_pos_ratio=("score", lambda x: (x > 0).mean() if len(x) > 0 else 0.0),
            headline_cnt=("score", "count"),
        ).fillna(0.0)
        roll = hourly.rolling(f"{window_hours}h", min_periods=1).mean()
        roll = roll.reset_index()
        roll["asset"] = asset
        out.append(roll)
    res = pd.concat(out, ignore_index=True)
    res["time"] = pd.to_datetime(res["time"], utc=True, errors="coerce")
    res = res.dropna(subset=["time"]).sort_values("time")
    return res

def merge_latest_news(cfg_path: str | pathlib.Path = "config.yaml", window_hours: int = 24) -> None:
    news_path = _list_latest_news()
    if news_path is None:
        logger.warning("No raw_news file found. Skipping merge.")
        return

    news = pd.read_parquet(news_path)
    if news.empty:
        logger.warning("Raw news file is empty. Skipping merge.")
        return

    agg = _agg_news(news, window_hours=window_hours)
    if agg.empty:
        logger.warning("Aggregated news is empty. Skipping merge.")
        return

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assets: List[str] = cfg["assets"]
    tfs: List[str] = cfg.get("timeframes", ["4h", "1d"])

    for asset in assets:
        for tf in (tfs if asset.upper().endswith("USDT") else ["1d"]):
            feat = _load_features(asset, tf)
            if feat.empty:
                continue

            feat["time"] = pd.to_datetime(feat["time"], utc=True, errors="coerce")
            feat = feat.dropna(subset=["time"]).sort_values("time")

            # --- töröljük az esetleges régi news oszlopokat ---
            for c in NEWS_COLS:
                if c in feat.columns:
                    feat.drop(columns=c, inplace=True)

            # --- csak az adott asset hírei ---
            asset_agg = agg[agg["asset"] == asset]
            if not asset_agg.empty:
                rhs = asset_agg[["time", *NEWS_COLS]].copy()
                join = pd.merge_asof(
                    feat,
                    rhs,
                    on="time",
                    direction="backward",
                    tolerance=pd.Timedelta("48h"),
                )
                for c in NEWS_COLS:
                    if c not in join.columns:
                        join[c] = 0.0
                    join[c] = join[c].fillna(0.0).astype(float)
            else:
                join = feat.copy()
                for c in NEWS_COLS:
                    join[c] = 0.0

            out = OUT_DIR / f"{asset}_{tf}.parquet"
            join.to_parquet(out, index=False)
            logger.info(f"Merged news -> {out} (window={window_hours}h)")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=24)
    args = ap.parse_args()
    merge_latest_news(window_hours=args.window)

if __name__ == "__main__":
    main()


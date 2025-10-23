# src/features/build_dataset.py
from __future__ import annotations
import pathlib
from typing import List, Dict
import pandas as pd
import yaml
from loguru import logger

from src.features.ta_features import compute_indicators

RAW_DIR = pathlib.Path("data/raw")
OUT_DIR = pathlib.Path("data/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_config(path: str | pathlib.Path = "config.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _guess_timeframes(asset: str, cfg_tfs: List[str]) -> List[str]:
    # kriptóknál 4h+1d, a többinél 1d (M1 szabály)
    return cfg_tfs if asset.upper().endswith("USDT") else ["1d"]

def _load_raw(asset: str, tf: str) -> pd.DataFrame:
    p = RAW_DIR / f"{asset}_{tf}.parquet"
    if not p.exists():
        logger.warning(f"Missing raw file: {p}")
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])
    df = pd.read_parquet(p)
    # biztos UTC datetime
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df

def _make_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    # Következő periódus hozam + előjel (klasszifikáció)
    df = df.copy()
    df["target_ret"] = df["close"].pct_change(horizon).shift(-horizon)
    df["target_sign"] = (df["target_ret"] > 0).astype("int8")
    return df

def build_features_for(asset: str, tf: str) -> pd.DataFrame:
    df = _load_raw(asset, tf)
    if df.empty or len(df) < 60:
        return pd.DataFrame()
    df = compute_indicators(df)
    df = _make_target(df, horizon=1)
    # tisztítás: az indikátorok eleje NaN → dobjuk
    df = df.dropna().reset_index(drop=True)
    # rendeljük az asset+tf meta infót
    df.insert(0, "asset", asset)
    df.insert(1, "timeframe", tf)
    return df

def build_all_features(cfg_path: str | pathlib.Path = "config.yaml") -> None:
    cfg = load_config(cfg_path)
    assets: List[str] = cfg["assets"]
    cfg_tfs: List[str] = cfg.get("timeframes", ["4h","1d"])
    logger.info(f"Building features for {len(assets)} assets…")

    total_rows = 0
    for asset in assets:
        for tf in _guess_timeframes(asset, cfg_tfs):
            try:
                feat = build_features_for(asset, tf)
                if feat.empty:
                    logger.warning(f"Skip features: empty {asset} {tf}")
                    continue
                out = OUT_DIR / f"{asset}_{tf}.parquet"
                feat.to_parquet(out, index=False)
                total_rows += len(feat)
                logger.info(f"Saved features {asset} {tf}: {len(feat):,} rows -> {out}")
            except Exception as e:
                logger.error(f"Feature build failed {asset} {tf}: {e}")

    logger.info(f"Features build done ✅ total_rows={total_rows:,}")

def main():
    build_all_features()

if __name__ == "__main__":
    main()

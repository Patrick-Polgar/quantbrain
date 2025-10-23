# src/viz/plot_asset.py
from __future__ import annotations
import pathlib
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

RAW_DIR = pathlib.Path("data/raw")
FEAT_DIR = pathlib.Path("data/features")
REPORTS_DIR = pathlib.Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def _read_raw(asset: str, tf: str) -> pd.DataFrame:
    p = RAW_DIR / f"{asset}_{tf}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df

def _read_signals(asset: str, tf: str, model: str) -> pd.DataFrame:
    p = REPORTS_DIR / f"signals_{asset}_{tf}_{model}.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df

def plot_with_signals(asset: str, tf: str, model: str = "logreg") -> pathlib.Path:
    price = _read_raw(asset, tf)
    sig = _read_signals(asset, tf, model)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=price["time"], open=price["open"], high=price["high"],
        low=price["low"], close=price["close"], name="Price"
    ))
    # BUY / SELL markerek
    buys = sig[sig["signal"] == "BUY"]
    sells = sig[sig["signal"] == "SELL"]
    fig.add_trace(go.Scatter(x=buys["time"], y=buys["close"], mode="markers",
                             marker_symbol="triangle-up", marker_size=10, name="BUY"))
    fig.add_trace(go.Scatter(x=sells["time"], y=sells["close"], mode="markers",
                             marker_symbol="triangle-down", marker_size=10, name="SELL"))

    fig.update_layout(title=f"{asset} {tf} – {model} signals",
                      xaxis_title="time", yaxis_title="price",
                      xaxis_rangeslider_visible=False)

    out = REPORTS_DIR / f"plot_{asset}_{tf}_{model}.html"
    fig.write_html(out)
    logger.info(f"Saved plot -> {out}")
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--model", choices=["logreg","rf"], default="logreg")
    args = ap.parse_args()
    plot_with_signals(args.asset, args.tf, args.model)

if __name__ == "__main__":
    main()

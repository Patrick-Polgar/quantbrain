# src/signals/generate.py
from __future__ import annotations
import pathlib
import pandas as pd
from loguru import logger
from src.models.baseline import predict_proba

REPORTS_DIR = pathlib.Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_signals(
    asset: str,
    tf: str,
    model_type: str = "logreg",
    prob_threshold: float = 0.6,
    hold: float = 0.4,
) -> pd.DataFrame:
    """
    Egyszerű, backtest-kompatibilis jelgenerálás:
      - BUY, ha p_buy >= th  -> +1
      - SELL, ha p_buy <= 1-th -> -1
      - különben HOLD -> 0
    A 'hold' opcionálisan enyhe kisimításra szolgál (rolling átlag).
    """
    probs = predict_proba(asset, tf, model_type)  # tartalmazza a 'time' oszlopot
    p = probs["p_buy"].astype(float)

    # Nyers jelek (−1 / 0 / +1)
    sig = pd.Series(0.0, index=probs.index)
    sig[p >= prob_threshold] = 1.0
    sig[p <= (1.0 - prob_threshold)] = -1.0

    # Enyhe kisimítás: ha hold > 0, egy kis ablakos átlag, majd kerekítés
    if hold and hold > 0:
        win = max(1, int(round(hold * 5)))  # kis, adaptív ablak
        sig = sig.rolling(window=win, min_periods=1).mean().round().clip(-1, 1)

    out = probs[["time"]].copy()
    out["signal"] = sig.astype(float)  # numeric kell a backtestnek
    out["confidence"] = (p - 0.5).abs() * 2.0  # 0..1 skála, opcionális
    logger.info(
        f"Signals generated for {asset} {tf} (th={prob_threshold}, hold={hold}) | "
        f"non-flat bars={(sig != 0).sum()}"
    )
    return out

def save_signals(df: pd.DataFrame, asset: str, tf: str, model: str):
    path = REPORTS_DIR / f"signals_{asset}_{tf}_{model}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved signals -> {path}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--model", choices=["logreg","rf"], default="logreg")
    ap.add_argument("--th", type=float, default=0.6)
    ap.add_argument("--hold", type=float, default=0.4)
    args = ap.parse_args()

    df = generate_signals(args.asset, args.tf, args.model, args.th, args.hold)
    save_signals(df, args.asset, args.tf, args.model)

if __name__ == "__main__":
    main()


# src/features/ta_features.py
from __future__ import annotations
import pandas as pd
import pandas_ta as ta

REQ_COLS = ["time", "open", "high", "low", "close", "volume"]

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Alap TA csomag: SMA10/50, RSI14, MACD_hist, ATR14, OBV."""
    df = df.copy()
    for c in REQ_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    df = df.sort_values("time").reset_index(drop=True)

    # SMA
    df["sma10"] = ta.sma(df["close"], length=10)
    df["sma50"] = ta.sma(df["close"], length=50)
    # RSI
    df["rsi14"] = ta.rsi(df["close"], length=14)
    # MACD (csak a hisztogramot tartjuk)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["macd_hist"] = macd.iloc[:, 2]
    else:
        df["macd_hist"] = pd.NA
    # ATR
    df["atr14"] = ta.atr(high=df["high"], low=df["low"], close=df["close"], length=14)
    # OBV
    df["obv"] = ta.obv(close=df["close"], volume=df["volume"])

    # derivált jelzők (opcionális, hasznos baseline)
    df["sma_cross"] = (df["sma10"] > df["sma50"]).astype("int8")
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)

    return df

# QuantBrain — simple_bt.py
# Egyszerű jel-alapú backtest min. 1 bar tartási védelemmel (min_hold_bars).
# Jel-formátum: {-1, 0, +1} vagy folytonos [-1..+1], ahol "hold" float esetén küszöb.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    ret_series: pd.Series
    pos: pd.Series
    trades: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equity_curve": self.equity_curve,
            "ret_series": self.ret_series,
            "pos": self.pos,
            "trades": self.trades,
        }


def _threshold_to_position(sig: pd.Series, hold: Optional[float]) -> pd.Series:
    """Folytonos jel -> diszkrét pozíció küszöböléssel, egyébként kerekítés/clip."""
    if np.issubdtype(sig.dtype, np.integer):
        return sig.clip(-1, 1).astype(int)

    if isinstance(hold, float):  # pl. 0.4 -> +/-0.4 küszöb
        thr = float(hold)
        pos = (sig >= thr).astype(int) - (sig <= -thr).astype(int)
        return pos.clip(-1, 1).astype(int)

    # nincs küszöb paraméter -> kerekítés
    return sig.round().clip(-1, 1).astype(int)


def _apply_hold_bars(pos: pd.Series, hold: Optional[int]) -> pd.Series:
    """Ha hold int, klasszikus k bárig tartjuk az előző pozíciót váltás után."""
    if not isinstance(hold, int) or hold <= 0:
        return pos

    hold_n = int(hold)
    last = 0
    k = 0
    out = pos.copy()
    for i in range(len(pos)):
        cur = int(pos.iat[i])
        if cur != last:
            if k < hold_n:
                out.iat[i] = last
                k += 1
            else:
                last = cur
                k = 0
        else:
            out.iat[i] = last
            k = 0
    return out


def _apply_min_hold_guard(pos: pd.Series, min_hold_bars: int = 1) -> pd.Series:
    """
    Minimum tartási védelem MINDEN esetben.
    Megakadályozza a "villogást", pl. --hold 0.4 mellett is legalább 1 bar-t bent kell maradni.
    """
    if min_hold_bars is None or min_hold_bars < 1:
        min_hold_bars = 1

    accepted = 0
    bars_since_change = 10**9
    out = pos.copy()

    for i in range(len(pos)):
        desired = int(pos.iat[i])
        if i == 0:
            accepted = desired
            bars_since_change = 0
            out.iat[i] = accepted
            continue

        if desired != accepted:
            if bars_since_change < min_hold_bars:
                # még tartjuk a korábbit
                out.iat[i] = accepted
                bars_since_change += 1
            else:
                accepted = desired
                out.iat[i] = accepted
                bars_since_change = 0
        else:
            out.iat[i] = accepted
            bars_since_change += 1
    return out


def run_backtest(
    df: pd.DataFrame,
    price_col: str = "close",
    signal_col: str = "signal",
    fee_bps: float = 2.0,
    slippage_bps: float = 0.0,
    hold: Optional[float | int] = None,
    min_hold_bars: int = 1,
    equity0: float = 1.0,
) -> Dict[str, Any]:
    """
    Egyszerű jel-alapú backtest.

    Paraméterek:
      - df: időrendezett DataFrame (index: datetime), oszlopok közt ár és jel
      - signal_col: {-1,0,+1} vagy folytonos [-1..+1]
      - hold:
           * float -> küszöb (pl. 0.4)
           * int   -> ennyi bar-ig kötelező tartani váltás után (klasszikus hold)
      - min_hold_bars: globális min. tartás MINDEN esetben (>=1)
      - fee_bps, slippage_bps: együttesen a váltáskor levont költség (bázispont)

    Visszatérés: dict( equity_curve, ret_series, pos, trades )
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index legyen DatetimeIndex (UTC idő javasolt).")
    if signal_col not in df.columns or price_col not in df.columns:
        raise ValueError(f"Hiányzó oszlop: {signal_col=} vagy {price_col=}.")

    df = df.copy()
    df = df.sort_index()

    # 1) Jel -> kezdeti diszkrét pozíció
    sig = df[signal_col]
    pos = _threshold_to_position(sig, hold)

    # 2) Klasszikus tartás (ha hold int)
    pos = _apply_hold_bars(pos, hold)

    # 3) Minimum 1 bar (vagy nagyobb) tartási védelem minden esetben
    pos = _apply_min_hold_guard(pos, min_hold_bars=min_hold_bars)

    # 4) PnL számítás
    px = df[price_col].astype(float)
    ret = px.pct_change().fillna(0.0)

    # kereskedési váltások (abs diff): 0->1, 1->-1 stb.
    trade_change = pos.diff().abs().fillna(0.0)
    cost_per_trade = (fee_bps + slippage_bps) / 1e4

    # pozíció az előző barra vonatkozóan érvényes
    pos_shift = pos.shift(1).fillna(0).astype(float)
    gross = pos_shift * ret  # long: +ret, short: -ret
    costs = trade_change * cost_per_trade
    ret_trd = gross - costs

    equity_curve = (ret_trd + 1.0).cumprod() * float(equity0)

    out = BacktestResult(
        equity_curve=equity_curve.rename("equity"),
        ret_series=ret_trd.rename("ret"),
        pos=pos.rename("pos"),
        trades=int(trade_change.sum()),
    )
    return out.to_dict()


# src/run/pipeline.py — Orchestration a 'src' modulokkal (CorrNet-ready)
from __future__ import annotations
import os, json, subprocess as sp, sys
from pathlib import Path
from typing import List
import yaml
from loguru import logger

PY = sys.executable
# Ez a file: <project>/src/run/pipeline.py → a projekt gyökér: parents[2]
PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "src"

def sh(args: List[str]) -> str:
    """Run a child process with PYTHONPATH=src, cwd=project root."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    logger.info("RUN: {}", " ".join(args))
    p = sp.Popen(args, cwd=str(PROJECT), env=env,
                 stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
    out, _ = p.communicate()
    if p.returncode != 0:
        print(out)
        raise SystemExit(p.returncode)
    return out

def load_cfg() -> dict:
    cfg_path = PROJECT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Defaults – ha nincs megadva a configban
    cfg.setdefault("model", "corrnet")
    cfg.setdefault("variant", "base_news")
    cfg.setdefault("fee_bps", 1.0)
    cfg.setdefault("hold", 0.40)
    cfg.setdefault("th_from", 0.55)
    cfg.setdefault("th_to", 0.70)
    cfg.setdefault("th_step", 0.01)
    cfg.setdefault("news_window_hours", 24)
    # A te config-odban 'timeframes' kulcs van — ezt használjuk
    if "timeframes" not in cfg:
        cfg["timeframes"] = ["1d"]
    if "assets" not in cfg:
        cfg["assets"] = ["GC=F"]
    return cfg

def main():
    cfg = load_cfg()

    # 1) Hírek/feature merge (ha van modulod hozzá)
    # Ha nincs ilyen modul, ezt a blokkot kommenteld ki.
    try:
        sh([PY, "-m", "features.merge_news", "--window", str(cfg["news_window_hours"])])
    except SystemExit:
        logger.warning("features.merge_news nem futott le — folytatom a tréninggel.")

    # 2) Train → Tune → Backtest minden asset×tf kombinációra
    for asset in cfg["assets"]:
        for tf in cfg["timeframes"]:
            # 2/a Tanítás + jelgenerálás (models/__main__.py végzi)
            sh([PY, "-m", "models.baseline",
                "--asset", asset, "--tf", tf,
                "--model", cfg["model"], "--variant", cfg["variant"]])

            # 2/b Küszöb-tuning
            out = sh([PY, "-m", "backtest.tune_threshold",
                      "--asset", asset, "--tf", tf, "--model", cfg["model"],
                      "--hold", str(cfg["hold"]), "--fee_bps", str(cfg["fee_bps"]),
                      "--th_from", str(cfg["th_from"]),
                      "--th_to", str(cfg["th_to"]),
                      "--th_step", str(cfg["th_step"])])

            # utolsó JSON sort olvassuk ({"best_th": ...})
            best_th = cfg.get("th_default", 0.60)
            for ln in reversed(out.splitlines()):
                if "best_th" in ln and "{" in ln:
                    try:
                        best_th = float(json.loads(ln.replace("'", "\""))["best_th"])
                        break
                    except Exception:
                        pass
            logger.info(f"[{asset} {tf}] best_th = {best_th:.3f}")

            # 2/c Backtest a legjobb küszöbbel
            sh([PY, "-m", "backtest.simple_bt",
                "--asset", asset, "--tf", tf, "--model", cfg["model"],
                "--th", str(best_th), "--hold", str(cfg["hold"]),
                "--fee_bps", str(cfg["fee_bps"])])

    logger.info("Pipeline OK ✅")

if __name__ == "__main__":
    main()


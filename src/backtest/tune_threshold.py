from pathlib import Path
import numpy as np
import subprocess as sp
import sys, json

PY = sys.executable
ROOT = Path(__file__).resolve().parents[0]

def run_bt(asset, tf, model, th, hold, fee):
    p = sp.Popen(
        [PY, "simple_bt.py",
         "--asset", asset, "--tf", tf, "--model", model,
         "--th", str(th), "--hold", str(hold), "--fee_bps", str(fee)],
        cwd=str(ROOT), stdout=sp.PIPE, stderr=sp.STDOUT, text=True
    )
    out, _ = p.communicate()
    # simple_bt a végén kiír egy dictet (pl. {"sharpe": ...})
    val = 0.0
    for ln in out.splitlines()[::-1]:
        if "sharpe" in ln and "{" in ln:
            try:
                obj = json.loads(ln.replace("'", "\""))
                val = float(obj.get("sharpe", val))
                break
            except Exception:
                continue
    return val

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--model", default="corrnet")
    ap.add_argument("--hold", type=float, default=0.4)
    ap.add_argument("--fee_bps", type=float, default=1.0)
    ap.add_argument("--th_from", type=float, default=0.55)
    ap.add_argument("--th_to", type=float, default=0.7)
    ap.add_argument("--th_step", type=float, default=0.02)
    args = ap.parse_args()

    best = (-1e9, None)
    ths = np.arange(args.th_from, args.th_to + 1e-9, args.th_step)
    for th in ths:
        s = run_bt(args.asset, args.tf, args.model, th, args.hold, args.fee_bps)
        if s > best[0]:
            best = (s, th)
    print({"best_sharpe": best[0], "best_th": best[1]})

if __name__ == "__main__":
    main()


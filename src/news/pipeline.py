from __future__ import annotations
import os, pathlib, json
from typing import Dict, List
from datetime import datetime, timezone, timedelta
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

from src.news.sources import fetch_newsapi, fetch_twitter, fetch_rss
from src.nlp.sentiment import score_many

RAW_NEWS_DIR = pathlib.Path("data/raw_news")
RAW_NEWS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

# Eszköz → kulcsszavak (bővíthető)
ASSET_KEYWORDS: Dict[str, List[str]] = {
    # --- Crypto
    "BTCUSDT": ["Bitcoin", "BTC", "crypto", "ETF", "SEC", "halving"],
    "ETHUSDT": ["Ethereum", "ETH", "staking", "L2"],
    "DOGEUSDT": ["Dogecoin", "DOGE", "Elon"],
    "SOLUSDT": ["Solana", "SOL", "DeFi", "NFT"],
    "XRPUSDT": ["XRP", "Ripple", "SEC"],

    # --- FX (FOMC / makró erősen releváns)
    "EURUSD=X": ["EURUSD", "ECB", "Fed", "FOMC", "Dollar Index", "DXY", "US CPI", "NFP", "PCE"],
    "GBPUSD=X": ["GBPUSD", "BoE", "Fed", "FOMC", "UK CPI"],
    "USDJPY=X": ["USDJPY", "BoJ", "Yen", "Ueda", "Fed", "FOMC", "Yield Curve"],
    "USDMXN=X": ["USDMXN", "Banxico", "Fed"],

    # --- Index
    "^GDAXI": ["DAX", "Germany stocks", "Bundesbank"],
    "QQQ": ["Nasdaq 100", "tech stocks", "mega-cap"],

    # --- Metals (itt a lényeg 😉)
    "GC=F": [  # Gold futures
        "gold", "XAUUSD", "bullion", "safe haven",
        "Fed", "FOMC", "Powell", "rate hike", "rate cut", "dot plot",
        "Treasury yields", "real yield", "DXY",
        "CPI", "PCE", "NFP", "ISM", "PMI",
        "geopolitics", "central bank gold buying"
    ],
    "SI=F": [  # Silver
        "silver", "XAGUSD", "industrial demand",
        "Fed", "FOMC", "Treasury yields", "DXY",
        "CPI", "NFP", "ISM", "PMI"
    ],
    "PL=F": [  # Platinum
        "platinum", "XPTUSD", "auto catalyst demand",
        "Fed", "FOMC", "DXY"
    ],
    "PA=F": [  # Palladium
        "palladium", "XPDUSD", "auto sector",
        "Fed", "FOMC", "DXY"
    ],
}

# Globális makró kulcsszavak – minden assethez hozzáadjuk
GLOBAL_MACRO: List[str] = [
    "Fed", "FOMC", "Powell", "Treasury yields", "DXY", "YoY",
    "CPI", "Core CPI", "PCE", "Core PCE", "NFP",
    "ISM", "PMI", "recession", "soft landing"
]

# Opcionális RSS-ek (pl. makró)
RSS_FEEDS = [
    # Központi bankok
    "https://www.federalreserve.gov/feeds/press_all.xml",
    "https://www.ecb.europa.eu/rss/press.html",

    # Kripto hírek
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://coincodex.com/rss/news/",

    # Általános/piaci
    "https://www.reuters.com/finance/markets/rss",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^NDX",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD",
]

def _collect_for_asset(asset: str, hours_newsapi=24, hours_twitter=12) -> pd.DataFrame:
    # Mindig legyen rows! (UnboundLocalError ellen)
    rows: List[Dict] = []

    # kulcsszavak + globális makró
    kws = ASSET_KEYWORDS.get(asset, [asset])
    kws = list(dict.fromkeys(kws + GLOBAL_MACRO))

    # ---- NewsAPI (OR-olt + szingli kulcsok; védett hívások) ----
    try:
        if os.getenv("NEWSAPI_KEY"):
            # OR-olt nagy lekérdezés
            rows += fetch_newsapi(" OR ".join(kws), since_hours=hours_newsapi)
            # plusz pár top-kulcsszó külön (dev plan-on több találat)
            for k in kws[:5]:
                rows += fetch_newsapi(k, since_hours=hours_newsapi)
    except Exception as e:
        # nem dőlünk el, megy tovább más forrásokra
        pass

    # ---- Twitter / X (snscrape) ----
    try:
        q = " OR ".join(kws[:6])  # rövidebb, stabilabb
        rows += fetch_twitter(q, since_hours=hours_twitter, limit=300)
    except Exception:
        pass

    # ---- RSS (mindig legyen) ----
    try:
        for url in RSS_FEEDS:
            rows += fetch_rss(url, since_hours=hours_newsapi)
    except Exception:
        pass

    # üres/sérült → üres DF standard sémával
    if not rows:
        return pd.DataFrame(columns=["time","source","url","title","text","provider","asset","score"])

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    df["asset"] = asset
    df["score"] = score_many(df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str))
    return df

def run_news_snapshot(assets: List[str], hours_newsapi=24, hours_twitter=12) -> pathlib.Path | None:
    all_df: List[pd.DataFrame] = []

    for a in assets:
        try:
            df = _collect_for_asset(a, hours_newsapi, hours_twitter)
            if not df.empty:
                all_df.append(df)
            else:
                logger.warning(f"No news for {a}")
        except Exception as e:
            # itt csak figyelmeztetünk, nem állítjuk le a futást
            logger.warning(f"News fetch failed for {a}: {e}")

    if not all_df:
        logger.warning("No news collected.")
        return None

    df = pd.concat(all_df, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out = RAW_NEWS_DIR / f"news_{ts}.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"Saved raw news -> {out} ({len(df)} rows)")

    return out

def main():
    import yaml, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--thours", type=int, default=12)
    args = ap.parse_args()

    with open("config.yaml","r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_news_snapshot(cfg["assets"], hours_newsapi=args.hours, hours_twitter=args.thours)

if __name__ == "__main__":
    main()

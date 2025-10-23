from __future__ import annotations
import os, time, json, re
from typing import List, Dict, Iterable
from datetime import datetime, timedelta, timezone
import requests
import feedparser

# X/Twitter (snscrape)
try:
    import snscrape.modules.twitter as sntwitter
except Exception:
    sntwitter = None

UTC = timezone.utc

def _now_utc():
    return datetime.now(tz=UTC)

def _to_iso(dt: datetime):
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def _clean(text: str) -> str:
    if not text: return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------- NewsAPI --------
def fetch_newsapi(query: str, since_hours: int = 24, lang: str = "en",
                  page_size: int = 50, sources: str | None = None) -> List[Dict]:
    key = os.getenv("NEWSAPI_KEY")
    if not key:
        return []
    from_dt = _now_utc() - timedelta(hours=since_hours)

    # 1) everything (lehet, hogy dev planon üres/limitált)
    url1 = "https://newsapi.org/v2/everything"
    p1 = {
        "q": query, "language": lang, "pageSize": page_size,
        "sortBy": "publishedAt", "from": _to_iso(from_dt), "apiKey": key,
    }
    if sources: p1["sources"] = sources
    r = requests.get(url1, params=p1, timeout=20)
    arts = []
    if r.status_code == 200:
        arts = (r.json() or {}).get("articles", []) or []

    # 2) fallback: top-headlines (domain-szűréssel)
    if not arts:
        url2 = "https://newsapi.org/v2/top-headlines"
        p2 = {"q": query, "language": lang, "pageSize": page_size, "apiKey": key}
        r2 = requests.get(url2, params=p2, timeout=20)
        if r2.status_code == 200:
            arts = (r2.json() or {}).get("articles", []) or []

    out = []
    for a in arts:
        out.append({
            "time": a.get("publishedAt"),
            "source": (a.get("source") or {}).get("name"),
            "url": a.get("url"),
            "title": _clean(a.get("title") or ""),
            "text": _clean((a.get("description") or "") + " " + (a.get("content") or ""))[:800],
            "provider": "newsapi",
        })
    return out

# -------- X / Twitter via snscrape --------
def fetch_twitter(query: str, since_hours: int = 12, limit: int = 300) -> List[Dict]:
    if sntwitter is None:
        return []
    since = _now_utc() - timedelta(hours=since_hours)
    # Szélesebb query: OR + kulcs-szavak, NEM tesszük bele a since: szűrőt (bugos lehet időzónán),
    # inkább utólag dobjuk ki az időn kívülieket.
    q = query  # pl. "Fed OR FOMC OR EURUSD"
    out = []
    try:
        for i, t in enumerate(sntwitter.TwitterSearchScraper(q).get_items()):
            if i >= limit: break
            dt = t.date.astimezone(UTC)
            if dt < since: continue
            txt = getattr(t, "content", "") or ""
            out.append({
                "time": dt.isoformat(),
                "source": t.user.username if getattr(t, "user", None) else "twitter",
                "url": f"https://twitter.com/i/web/status/{t.id}",
                "title": _clean(txt[:120]),
                "text": _clean(txt[:1000]),
                "provider": "twitter",
            })
    except Exception:
        # ha scraping blokkolt, térjünk vissza üresen
        return []
    return out

# -------- RSS (opcionális) --------
def fetch_rss(feed_url: str, since_hours: int = 24) -> List[Dict]:
    since = _now_utc() - timedelta(hours=since_hours)
    fp = feedparser.parse(feed_url)
    out = []
    for e in fp.entries:
        # best-effort dátum
        dt = None
        for k in ("published_parsed","updated_parsed"):
            if hasattr(e, k) and getattr(e, k):
                try:
                    from time import mktime
                    dt = datetime.fromtimestamp(mktime(getattr(e, k)), tz=UTC)
                    break
                except Exception:
                    pass
        if not dt:  # ha nincs dátum, átugorjuk
            continue
        if dt < since:
            continue
        out.append({
            "time": dt.isoformat(),
            "source": getattr(e, "source", {}).get("title", "") or getattr(e, "author", "") or "rss",
            "url": getattr(e, "link", ""),
            "title": _clean(getattr(e, "title", "")),
            "text": _clean(getattr(e, "summary", ""))[:800],
            "provider": "rss",
        })
    return out

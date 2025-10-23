from __future__ import annotations
from typing import Iterable, List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_an = SentimentIntensityAnalyzer()

def score_text(text: str) -> float:
    if not text:
        return 0.0
    s = _an.polarity_scores(text)
    return float(s.get("compound", 0.0))

def score_many(texts: Iterable[str]) -> List[float]:
    return [score_text(t) for t in texts]

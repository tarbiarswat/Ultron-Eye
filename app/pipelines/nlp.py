import pandas as pd
try:
    import nltk
    nltk.data.find('sentiment/vader_lexicon.zip')
except Exception:
    import nltk
    nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()

def sentiment_scores(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    out = df.copy()
    out["sent_neg"] = out[text_col].map(lambda s: _sia.polarity_scores(s)["neg"])
    out["sent_neu"] = out[text_col].map(lambda s: _sia.polarity_scores(s)["neu"])
    out["sent_pos"] = out[text_col].map(lambda s: _sia.polarity_scores(s)["pos"])
    out["sent_compound"] = out[text_col].map(lambda s: _sia.polarity_scores(s)["compound"])
    return out
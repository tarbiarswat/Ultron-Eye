import re
import regex as rxx
from unidecode import unidecode

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMOJI_RE = rxx.compile(r"\p{Emoji}", flags=rxx.VERSION1)

def normalize(text: str) -> str:
    return unidecode(text or "")

def strip_urls(text: str) -> str:
    return URL_RE.sub(" ", text)

def strip_emojis(text: str) -> str:
    return EMOJI_RE.sub(" ", text)

def strip_nonletters(text: str, keep_spaces=True) -> str:
    return re.sub(r"[^A-Za-z\s]" if keep_spaces else r"[^A-Za-z]", " ", text)
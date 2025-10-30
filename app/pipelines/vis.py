from pathlib import Path
from typing import Iterable, List
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from io import BytesIO

def wordcloud_from_tokens(tokens_iter: Iterable[List[str]], width=1000, height=500, bg="white"):
    text_blob = " ".join([" ".join(toks) for toks in tokens_iter])
    if not text_blob.strip():
        return None
    wc = WordCloud(width=width, height=height, background_color=bg).generate(text_blob)
    return wc.to_array()

def animated_wordcloud(tokens_iter: Iterable[List[str]], out_path: Path, frames: int = 12,
                       step: int = 1, width: int = 900, height: int = 400, bg: str = "white", fps: int = 3):
    tokens = list(tokens_iter)
    if not tokens:
        return None

    images = []
    n = len(tokens)

    for i in range(1, frames + 1):
        end = max(1, int(i * n / frames))
        blob = " ".join([" ".join(toks) for toks in tokens[:end:step]])
        if not blob.strip():
            continue

        wc = WordCloud(width=width, height=height, background_color=bg).generate(blob)
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

        # Portable render → PNG bytes → numpy array
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img = imageio.imread(buf)
        images.append(img)

        plt.close(fig)

    if not images:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, images, duration=1.0 / fps)
    return out_path

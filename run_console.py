import pandas as pd
from app.pipelines.clean import clean_frame, token_counts
from app.utils.io import write_df, ts
from app.config import RAW_DIR, PROC_DIR

if __name__ == "__main__":
    demo = pd.DataFrame({"text": [
        "AI camera upgrades dominate the iphone 17 pro chatter!",
        "iphone 17 air praised for portability and battery life.",
        "Users compare iphone 17 pro vs iphone 17 air.",
    ]})
    write_df(demo, RAW_DIR / f"raw_console_{ts()}.csv")
    cleaned = clean_frame(demo, text_col="text")
    write_df(cleaned, PROC_DIR / f"processed_console_{ts()}.csv")
    print(token_counts(cleaned).head(20))
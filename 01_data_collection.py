"""
DeepSentinel — Data Collection Module
Collects data from Reddit (via PRAW) and loads CrisisNLP/HatEval datasets.
"""

import praw
import pandas as pd
import json
import time
import os
from datetime import datetime

# ─── Reddit Collector ────────────────────────────────────────────────────────

def get_reddit_client():
    """
    Initialize PRAW Reddit client.
    Set env vars: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
    """
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID", "YOUR_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "DeepSentinel/1.0"),
    )


CRISIS_SUBREDDITS = [
    "mentalhealth", "depression", "anxiety",
    "SuicideWatch", "offmychest",
    "worldnews", "conspiracy", "politics",
    "AmItheAsshole", "relationship_advice"
]

CRISIS_KEYWORDS = [
    "crisis", "emergency", "help me", "i cant take it",
    "fake news", "misinformation", "rumor",
    "bullying", "harassed", "threat", "hate",
    "protest", "riot", "unrest", "violence"
]


def collect_reddit_posts(subreddits=CRISIS_SUBREDDITS, limit=500, save_path="data/reddit_raw.csv"):
    """Collect posts from crisis-related subreddits."""
    reddit = get_reddit_client()
    records = []

    for sub_name in subreddits:
        print(f"  Scraping r/{sub_name}...")
        try:
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.hot(limit=limit):
                records.append({
                    "id":          post.id,
                    "source":      f"reddit/r/{sub_name}",
                    "text":        f"{post.title} {post.selftext}".strip(),
                    "score":       post.score,
                    "num_comments":post.num_comments,
                    "created_utc": datetime.utcfromtimestamp(post.created_utc).isoformat(),
                    "url":         post.url,
                    "label":       None,   # to be annotated
                })
            time.sleep(1)          # polite rate-limiting
        except Exception as e:
            print(f"    Warning: could not scrape r/{sub_name}: {e}")

    df = pd.DataFrame(records).drop_duplicates("id")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} Reddit posts → {save_path}")
    return df


# ─── CrisisNLP / HatEval Loaders ─────────────────────────────────────────────

def load_crisisnlp(path="data/crisisnlp/"):
    """
    Load CrisisNLP dataset (https://crisisnlp.qcri.org/).
    Expected TSV columns: tweet_id, label, text
    """
    frames = []
    if not os.path.exists(path):
        print(f"CrisisNLP path not found: {path}. Download from crisisnlp.qcri.org")
        return pd.DataFrame()

    for fname in os.listdir(path):
        if fname.endswith(".tsv") or fname.endswith(".csv"):
            sep = "\t" if fname.endswith(".tsv") else ","
            df = pd.read_csv(os.path.join(path, fname), sep=sep)
            df["source"] = "crisisnlp"
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"tweet_id": "id", "tweet_text": "text"})
    return combined[["id", "text", "label", "source"]]


def load_hateval(path="data/hateval/hateval2019_en_train.csv"):
    """
    Load HatEval 2019 dataset (SemEval Task 5).
    Columns: id, text, HS (hate speech 0/1), TR (target range 0/1), AG (aggressiveness 0/1)
    """
    if not os.path.exists(path):
        print(f"HatEval path not found: {path}. Download from competitions.codalab.org/competitions/19935")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["source"] = "hateval"
    df["label"] = df["HS"].map({0: "normal", 1: "cyberbullying"})
    return df[["id", "text", "label", "source"]]


# ─── Merge & Preprocess ───────────────────────────────────────────────────────

LABEL_MAP = {
    "misinformation":  0,
    "mental_health":   1,
    "cyberbullying":   2,
    "political_unrest":3,
    "normal":          4,
}


def build_dataset(save_path="data/dataset.csv"):
    """Combine all sources into a single labeled dataset."""
    frames = []

    reddit = pd.read_csv("data/reddit_raw.csv") if os.path.exists("data/reddit_raw.csv") else pd.DataFrame()
    cnlp   = load_crisisnlp()
    hatev  = load_hateval()

    for df in [reddit, cnlp, hatev]:
        if not df.empty:
            frames.append(df)

    if not frames:
        print("No data found. Run collect_reddit_posts() first.")
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.dropna(subset=["text"])
    dataset["text"] = dataset["text"].str.strip()
    dataset = dataset[dataset["text"].str.len() > 10]
    dataset["label_id"] = dataset["label"].map(LABEL_MAP).fillna(4).astype(int)

    dataset.to_csv(save_path, index=False)
    print(f"Final dataset: {len(dataset)} samples → {save_path}")
    print(dataset["label"].value_counts())
    return dataset


if __name__ == "__main__":
    print("=== DeepSentinel Data Collection ===")
    collect_reddit_posts(limit=200)
    build_dataset()

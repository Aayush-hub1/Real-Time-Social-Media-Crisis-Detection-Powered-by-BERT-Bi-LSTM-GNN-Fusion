"""
DeepSentinel — Preprocessing Module
Cleans text, extracts emoji features, builds hashtag graphs, tokenises for BERT.
"""

import re
import json
import emoji
import numpy as np
import pandas as pd
import networkx as nx
from transformers import BertTokenizer

# ─── Text Cleaning ────────────────────────────────────────────────────────────

URL_RE    = re.compile(r"http\S+|www\.\S+")
MENTION_RE= re.compile(r"@\w+")
HASH_RE   = re.compile(r"#(\w+)")
SPACE_RE  = re.compile(r"\s+")


def clean_text(text: str, keep_hashtags=True) -> str:
    """Remove noise while preserving semantic tokens."""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    if not keep_hashtags:
        text = HASH_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text.strip())
    return text.lower()


# ─── Emoji Encoding ───────────────────────────────────────────────────────────

# High-signal crisis emojis mapped to feature categories
EMOJI_CRISIS_MAP = {
    "😭": "distress",  "😢": "distress",  "💀": "distress",
    "🔥": "urgency",   "⚠️": "urgency",   "🚨": "urgency",
    "😡": "aggression","🤬": "aggression","💢": "aggression",
    "❤️": "support",   "🙏": "support",   "💪": "support",
}

EMOJI_CATEGORIES = ["distress", "urgency", "aggression", "support", "other"]


def extract_emoji_features(text: str) -> dict:
    """Return a dict of emoji category counts."""
    counts = {cat: 0 for cat in EMOJI_CATEGORIES}
    for char in text:
        if char in emoji.EMOJI_DATA:
            cat = EMOJI_CRISIS_MAP.get(char, "other")
            counts[cat] += 1
    counts["total_emoji"] = sum(counts.values())
    return counts


def encode_emoji_vector(text: str) -> np.ndarray:
    """Return a 6-dim numpy vector of emoji features."""
    feats = extract_emoji_features(text)
    return np.array([feats[c] for c in EMOJI_CATEGORIES + ["total_emoji"]], dtype=np.float32)


# ─── Hashtag Graph ────────────────────────────────────────────────────────────

def extract_hashtags(text: str) -> list:
    return [h.lower() for h in HASH_RE.findall(text)]


def build_hashtag_graph(texts: list) -> nx.Graph:
    """
    Build a co-occurrence graph: nodes = hashtags, edges = appear in same post.
    Edge weight = co-occurrence count.
    """
    G = nx.Graph()
    for text in texts:
        tags = extract_hashtags(text)
        for i, t1 in enumerate(tags):
            for t2 in tags[i+1:]:
                if G.has_edge(t1, t2):
                    G[t1][t2]["weight"] += 1
                else:
                    G.add_edge(t1, t2, weight=1)
    return G


def hashtag_graph_features(G: nx.Graph, hashtags: list) -> np.ndarray:
    """
    For a single post's hashtags, compute graph-level features:
    [max_degree, avg_degree, max_betweenness, subgraph_density]
    """
    if not hashtags or len(G) == 0:
        return np.zeros(4, dtype=np.float32)

    sub_nodes = [h for h in hashtags if h in G]
    if len(sub_nodes) < 2:
        return np.zeros(4, dtype=np.float32)

    sub = G.subgraph(sub_nodes)
    degrees = [d for _, d in sub.degree()]
    bc = list(nx.betweenness_centrality(sub).values()) if len(sub) > 1 else [0]

    return np.array([
        max(degrees, default=0),
        np.mean(degrees) if degrees else 0,
        max(bc),
        nx.density(sub),
    ], dtype=np.float32)


# ─── BERT Tokeniser ───────────────────────────────────────────────────────────

TOKENIZER = None   # lazy-loaded

def get_tokenizer(model_name="bert-base-uncased"):
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = BertTokenizer.from_pretrained(model_name)
    return TOKENIZER


def tokenize_batch(texts: list, max_length=128):
    """Return HuggingFace BatchEncoding (input_ids, attention_mask, token_type_ids)."""
    tok = get_tokenizer()
    return tok(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# ─── Full Preprocessing Pipeline ─────────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame, text_col="text") -> pd.DataFrame:
    """Apply full preprocessing to a dataset DataFrame."""
    print("Cleaning text...")
    df["clean_text"] = df[text_col].fillna("").apply(clean_text)

    print("Extracting emoji features...")
    emoji_feats = df[text_col].apply(extract_emoji_features).apply(pd.Series)
    df = pd.concat([df, emoji_feats.add_prefix("emoji_")], axis=1)

    print("Extracting hashtags...")
    df["hashtags"] = df[text_col].apply(extract_hashtags)

    print("Done. Shape:", df.shape)
    return df


if __name__ == "__main__":
    sample = pd.DataFrame({
        "text": [
            "I can't take this anymore 😭😭 #mentalhealth #help",
            "BREAKING: Massive #protest in city center #riot #unrest 🔥🔥",
            "This story is FAKE NEWS!! 😡 #misinformation #fakenews",
            "Sending love to everyone struggling tonight ❤️🙏 #MentalHealthAwareness",
        ]
    })

    processed = preprocess_dataframe(sample)
    print(processed[["clean_text", "hashtags", "emoji_distress", "emoji_urgency"]].to_string())

    print("\nTokenising...")
    enc = tokenize_batch(processed["clean_text"].tolist())
    print("input_ids shape:", enc["input_ids"].shape)

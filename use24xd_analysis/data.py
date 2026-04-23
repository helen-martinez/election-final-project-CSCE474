from __future__ import annotations
import ast, re
from pathlib import Path
import numpy as np
import pandas as pd
from .config import Config

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

RENAME_MAP = {
    "public_metrics.retweet_count": "retweet_count",
    "public_metrics.reply_count": "reply_count",
    "public_metrics.like_count": "like_count",
    "public_metrics.quote_count": "quote_count",
    "public_metrics.bookmark_count": "bookmark_count",
    "public_metrics.impression_count": "impression_count",
}

REQUIRED = ["id", "created_at", "text", "text_clean", "lang", "sentiment_vader_raw", "sentiment_vader_label"]


def clean_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s)
    return s.strip().lower()


def parse_list_like(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "[]"}:
        return []
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, list):
            items = obj
        else:
            items = [text]
    except Exception:
        items = [text]
    out: list[str] = []
    for item in items:
        for tok in TOKEN_RE.findall(str(item).lower()):
            tok = tok.strip("#")
            if tok:
                out.append(tok)
    return sorted(set(out))


def add_time_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    election_day = pd.Timestamp(cfg.election_day, tz="UTC")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_at"]).copy()
    df["date"] = df["created_at"].dt.date.astype(str)
    df["week"] = df["created_at"].dt.to_period("W-SUN").apply(lambda p: p.start_time).dt.tz_localize("UTC")
    df["month"] = df["created_at"].dt.to_period("M").astype(str)
    df["day_of_week"] = df["created_at"].dt.day_name()
    df["hour"] = df["created_at"].dt.hour
    df["days_to_election"] = (df["created_at"] - election_day).dt.days
    df["election_period"] = np.select(
        [df["days_to_election"] < 0, df["days_to_election"].between(0, 7), df["days_to_election"] > 7],
        ["pre_election", "election_week", "post_election"],
        default="unknown",
    )
    return df


def load_and_preprocess(cfg: Config) -> pd.DataFrame:
    cfg.ensure_dirs()
    df = pd.read_csv(cfg.dataset_csv, low_memory=False)
    df = df.rename(columns=RENAME_MAP)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if cfg.max_rows is not None and len(df) > cfg.max_rows:
        df = df.sample(cfg.max_rows, random_state=cfg.random_state).copy()

    df = df.drop_duplicates(subset=["id"]).copy()
    df["lang"] = df["lang"].astype(str).str.lower().str.strip()
    df = df[df["lang"] == "en"].copy()

    # Text field: prefer provided clean text, fall back to our cleaner.
    df["clean_text"] = df["text_clean"].fillna("").astype(str).str.strip().str.lower()
    missing_clean = df["clean_text"].eq("") | df["clean_text"].isin(["nan", "none"])
    df.loc[missing_clean, "clean_text"] = df.loc[missing_clean, "text"].map(clean_text)
    df["word_count_clean"] = df["clean_text"].map(lambda x: len(str(x).split()))
    df = df[df["word_count_clean"] >= cfg.min_text_words].copy()

    df = add_time_features(df, cfg)

    for col in ["retweet_count", "reply_count", "like_count", "quote_count", "bookmark_count", "impression_count", "sentiment_vader_raw", "word_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["retweet_count", "reply_count", "like_count", "quote_count", "bookmark_count", "impression_count"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).clip(lower=0)

    df["engagement_total"] = df[["retweet_count", "reply_count", "like_count", "quote_count", "bookmark_count"]].sum(axis=1)
    df["log_engagement"] = np.log1p(df["engagement_total"])
    df["engagement_rate"] = np.where(df["impression_count"] > 0, df["engagement_total"] / df["impression_count"], np.nan)

    for col in cfg.labels:
        if col not in df.columns:
            raise ValueError(f"Missing harmful-content label column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    df["harmful_label_count"] = df[list(cfg.labels)].sum(axis=1)
    df["any_harmful"] = (df["harmful_label_count"] > 0).astype(int)

    df["hashtag_tokens"] = df.get("hashtags", pd.Series([""] * len(df))).map(parse_list_like)
    df["mention_tokens"] = df.get("entities.mentions", pd.Series([""] * len(df))).map(parse_list_like)
    df["has_hashtags"] = df["hashtag_tokens"].map(lambda x: len(x) > 0)
    df["has_mentions"] = df["mention_tokens"].map(lambda x: len(x) > 0)
    df["has_state"] = df.get("user_location_USA_state", pd.Series([np.nan] * len(df))).notna()

    df["verified"] = df.get("verified", False).astype(str).str.lower().isin(["true", "1", "yes"])
    df["possibly_sensitive"] = df.get("possibly_sensitive", False).astype(str).str.lower().isin(["true", "1", "yes"])

    # Engagement buckets based on quantiles.
    q33, q66 = df["engagement_total"].quantile([0.33, 0.66])
    df["engagement_bucket"] = pd.cut(
        df["engagement_total"], bins=[-1, q33, q66, np.inf], labels=["low", "medium", "high"]
    ).astype(str)

    out = cfg.output_dir / "preprocessing"
    df.to_parquet(out / "clean_posts.parquet", index=False)
    df.head(1000).to_csv(out / "clean_posts_sample1000.csv", index=False)

    summary = pd.DataFrame({
        "metric": ["rows", "unique_authors", "min_date", "max_date", "english_rows", "any_harmful_share", "avg_engagement"],
        "value": [len(df), df.get("author_id", pd.Series(dtype=object)).nunique(), df["created_at"].min(), df["created_at"].max(), len(df), df["any_harmful"].mean(), df["engagement_total"].mean()],
    })
    summary.to_csv(out / "preprocessing_summary.csv", index=False)
    return df.reset_index(drop=True)

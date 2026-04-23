from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .config import Config
from .visuals import savefig


def add_sentiment_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sentiment_vader_raw"] = pd.to_numeric(out["sentiment_vader_raw"], errors="coerce").fillna(0)
    out["sentiment_5bin"] = pd.cut(
        out["sentiment_vader_raw"],
        bins=[-1.01, -0.60, -0.05, 0.05, 0.60, 1.01],
        labels=["very_negative", "negative", "neutral", "positive", "very_positive"],
    ).astype(str)
    return out


def run_sentiment_analysis(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out_dir = cfg.output_dir / "sentiment"
    viz = cfg.output_dir / "visualizations"
    df = add_sentiment_bins(df)

    overall = df["sentiment_5bin"].value_counts(normalize=True).rename_axis("sentiment_5bin").reset_index(name="share")
    overall.to_csv(out_dir / "sentiment_distribution_5bin.csv", index=False)

    by_label_rows = []
    for lab in cfg.labels:
        tmp = df.groupby(lab)["sentiment_vader_raw"].agg(["count", "mean", "median", "std"]).reset_index()
        tmp["label"] = lab
        by_label_rows.append(tmp)
    by_label = pd.concat(by_label_rows, ignore_index=True)
    by_label.to_csv(out_dir / "sentiment_by_harmful_label.csv", index=False)

    weekly = df.groupby(["week", "sentiment_5bin"]).size().reset_index(name="count")
    weekly["share"] = weekly["count"] / weekly.groupby("week")["count"].transform("sum")
    weekly.to_csv(out_dir / "weekly_sentiment_5bin_share.csv", index=False)

    plt.figure(figsize=(8, 4))
    order = ["very_negative", "negative", "neutral", "positive", "very_positive"]
    sns.countplot(data=df, x="sentiment_5bin", order=order)
    plt.title("Five-Level VADER Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Posts")
    plt.xticks(rotation=20)
    savefig(viz / "20_sentiment_5bin_distribution.png")

    label_melt = df.melt(id_vars=["sentiment_vader_raw"], value_vars=list(cfg.labels), var_name="label", value_name="present")
    label_melt = label_melt[label_melt["present"] == 1]
    if not label_melt.empty:
        plt.figure(figsize=(10, 5))
        sns.violinplot(data=label_melt, x="label", y="sentiment_vader_raw", inner="quartile", cut=0)
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.title("Sentiment Score Distribution by Harmful-Content Label")
        plt.xlabel("Label")
        plt.ylabel("VADER compound score")
        plt.xticks(rotation=25, ha="right")
        savefig(viz / "21_sentiment_violin_by_label.png")

    pivot = weekly.pivot(index="week", columns="sentiment_5bin", values="share").fillna(0).reindex(columns=order, fill_value=0)
    plt.figure(figsize=(11, 5))
    pivot.plot(kind="area", stacked=True, figsize=(11, 5), alpha=0.8)
    plt.title("Weekly Sentiment Share")
    plt.xlabel("Week")
    plt.ylabel("Share")
    plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(viz / "22_weekly_sentiment_share_area.png")

    return df

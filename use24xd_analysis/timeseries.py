from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .config import Config
from .visuals import savefig


def run_timeseries_analysis(df: pd.DataFrame, cfg: Config) -> None:
    out = cfg.output_dir / "timeseries"
    viz = cfg.output_dir / "visualizations"
    election_day = pd.Timestamp(cfg.election_day, tz="UTC")

    weekly = df.groupby("week").agg(
        posts=("id", "count"),
        avg_engagement=("engagement_total", "mean"),
        median_engagement=("engagement_total", "median"),
        avg_sentiment=("sentiment_vader_raw", "mean"),
        harmful_share=("any_harmful", "mean"),
        avg_harmful_label_count=("harmful_label_count", "mean"),
    ).reset_index().sort_values("week")
    weekly["posts_rollmean"] = weekly["posts"].rolling(cfg.rolling_window_weeks, min_periods=1).mean()
    weekly["harmful_rollmean"] = weekly["harmful_share"].rolling(cfg.rolling_window_weeks, min_periods=1).mean()
    weekly.to_csv(out / "weekly_summary.csv", index=False)

    label_weekly = df.groupby("week")[list(cfg.labels)].mean().reset_index()
    label_weekly.to_csv(out / "weekly_harmful_label_share.csv", index=False)

    topic_weekly = df.groupby(["week", "topic_label"]).size().reset_index(name="count")
    topic_weekly["share"] = topic_weekly["count"] / topic_weekly.groupby("week")["count"].transform("sum")
    topic_weekly.to_csv(out / "weekly_topic_share.csv", index=False)

    plt.figure(figsize=(11, 4))
    sns.lineplot(data=weekly, x="week", y="posts", marker="o", label="Weekly posts")
    sns.lineplot(data=weekly, x="week", y="posts_rollmean", linestyle="--", label=f"{cfg.rolling_window_weeks}-week rolling mean")
    plt.axvline(election_day, linestyle="--", linewidth=1.5, label="Election Day")
    plt.title("Weekly Post Volume with Election Day Marker")
    plt.xlabel("Week")
    plt.ylabel("Posts")
    plt.legend()
    savefig(viz / "40_timeseries_weekly_post_volume.png")

    plt.figure(figsize=(11, 4))
    sns.lineplot(data=weekly, x="week", y="harmful_share", marker="o", label="Any harmful label")
    sns.lineplot(data=weekly, x="week", y="harmful_rollmean", linestyle="--", label=f"{cfg.rolling_window_weeks}-week rolling mean")
    plt.axvline(election_day, linestyle="--", linewidth=1.5, label="Election Day")
    plt.title("Weekly Harmful-Content Share")
    plt.xlabel("Week")
    plt.ylabel("Share")
    plt.legend()
    savefig(viz / "41_timeseries_harmful_share.png")

    melted = label_weekly.melt(id_vars="week", value_vars=list(cfg.labels), var_name="label", value_name="share")
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=melted, x="week", y="share", hue="label", marker="o")
    plt.axvline(election_day, linestyle="--", linewidth=1.5)
    plt.title("Weekly Prevalence of Harmful-Content Categories")
    plt.xlabel("Week")
    plt.ylabel("Share of posts")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(viz / "42_timeseries_harmful_labels.png")

    pivot = label_weekly.set_index("week")[list(cfg.labels)]
    plt.figure(figsize=(11, 5))
    sns.heatmap(pivot, cmap="rocket_r", cbar_kws={"label": "Share"})
    plt.title("Weekly Harmful-Content Label Share Heatmap")
    savefig(viz / "43_timeseries_label_heatmap.png")

    topic_pivot = topic_weekly.pivot(index="week", columns="topic_label", values="share").fillna(0)
    plt.figure(figsize=(12, 6))
    topic_pivot.plot(kind="area", stacked=True, figsize=(12, 6), alpha=0.85)
    plt.axvline(election_day, linestyle="--", linewidth=1.5)
    plt.title("Topic Prevalence Over Time")
    plt.xlabel("Week")
    plt.ylabel("Share")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    savefig(viz / "44_timeseries_topic_stream_area.png")

    # Event-window plot around election day: -30 to +30 days.
    event = df[df["days_to_election"].between(-30, 30)].copy()
    if not event.empty:
        daily = event.groupby("days_to_election").agg(posts=("id", "count"), harmful_share=("any_harmful", "mean"), avg_sentiment=("sentiment_vader_raw", "mean")).reset_index()
        plt.figure(figsize=(11, 4))
        sns.lineplot(data=daily, x="days_to_election", y="harmful_share", marker="o")
        plt.axvline(0, linestyle="--", linewidth=1.5)
        plt.title("Event Window: Harmful-Content Share Around Election Day")
        plt.xlabel("Days from Election Day")
        plt.ylabel("Harmful share")
        savefig(viz / "45_event_window_harmful_share.png")
        daily.to_csv(out / "event_window_daily_summary.csv", index=False)

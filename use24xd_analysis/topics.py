from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .config import Config
from .visuals import savefig

@dataclass
class TopicResult:
    document_topics: pd.DataFrame
    topic_words: pd.DataFrame
    topic_summary: pd.DataFrame
    topic_metrics: pd.DataFrame


def _top_words(model, feature_names, n: int) -> pd.DataFrame:
    rows = []
    for topic_idx, comp in enumerate(model.components_):
        inds = np.argsort(comp)[::-1][:n]
        for rank, i in enumerate(inds, 1):
            rows.append({"topic_id": topic_idx, "rank": rank, "word": feature_names[i], "weight": comp[i]})
    return pd.DataFrame(rows)


def run_topic_modeling(df: pd.DataFrame, cfg: Config) -> TopicResult:
    out = cfg.output_dir / "topics"
    viz = cfg.output_dir / "visualizations"

    vectorizer = TfidfVectorizer(
        max_features=cfg.tfidf_max_features,
        min_df=cfg.tfidf_min_df,
        max_df=cfg.tfidf_max_df,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["clean_text"])
    terms = np.array(vectorizer.get_feature_names_out())

    nmf = NMF(n_components=cfg.n_topics, random_state=cfg.random_state, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)
    nmf_topic = W.argmax(axis=1)
    nmf_strength = W.max(axis=1)

    # Comparison models: KMeans and LDA. These are saved for evaluation/reporting.
    svd_for_cluster = TruncatedSVD(n_components=min(100, X.shape[1]-1), random_state=cfg.random_state)
    X_svd = svd_for_cluster.fit_transform(X)
    kmeans = KMeans(n_clusters=cfg.n_topics, random_state=cfg.random_state, n_init="auto")
    kmeans_labels = kmeans.fit_predict(X_svd)

    lda = LatentDirichletAllocation(n_components=cfg.n_topics, random_state=cfg.random_state, learning_method="batch", max_iter=10)
    lda_doc = lda.fit_transform(X)
    lda_topic = lda_doc.argmax(axis=1)

    try:
        sil = silhouette_score(X_svd, kmeans_labels, sample_size=min(10000, len(df)), random_state=cfg.random_state)
    except Exception:
        sil = np.nan

    topic_words = _top_words(nmf, terms, cfg.top_words_per_topic)
    topic_words["topic_label"] = topic_words.groupby("topic_id")["word"].transform(lambda s: " / ".join(s.head(4)))
    topic_label_map = topic_words.groupby("topic_id")["topic_label"].first().to_dict()

    doc = df.copy()
    doc["topic_id"] = nmf_topic
    doc["topic_label"] = doc["topic_id"].map(topic_label_map)
    doc["topic_strength"] = nmf_strength
    doc["kmeans_topic"] = kmeans_labels
    doc["lda_topic"] = lda_topic

    topic_summary = (
        doc.groupby(["topic_id", "topic_label"])
        .agg(posts=("id", "count"), avg_engagement=("engagement_total", "mean"), avg_sentiment=("sentiment_vader_raw", "mean"), harmful_share=("any_harmful", "mean"))
        .reset_index()
        .sort_values("posts", ascending=False)
    )
    topic_summary["post_share"] = topic_summary["posts"] / len(doc)

    # Representative posts are the highest-strength posts per topic.
    reps = (
        doc.sort_values(["topic_id", "topic_strength"], ascending=[True, False])
        .groupby("topic_id")
        .head(cfg.n_representative_posts)[["topic_id", "topic_label", "topic_strength", "text", "created_at", "engagement_total", *cfg.labels]]
    )

    topic_metrics = pd.DataFrame([
        {"model": "NMF", "metric": "reconstruction_error", "value": float(nmf.reconstruction_err_)},
        {"model": "KMeans_on_SVD", "metric": "silhouette_score_sampled", "value": sil},
        {"model": "LDA", "metric": "perplexity", "value": float(lda.perplexity(X))},
    ])

    doc.to_parquet(out / "document_topics.parquet", index=False)
    topic_words.to_csv(out / "topic_words_nmf.csv", index=False)
    topic_summary.to_csv(out / "topic_summary.csv", index=False)
    reps.to_csv(out / "representative_posts_by_topic.csv", index=False)
    topic_metrics.to_csv(out / "topic_model_metrics.csv", index=False)

    # Visuals
    plt.figure(figsize=(10, 5))
    sns.barplot(data=topic_summary, y="topic_label", x="posts")
    plt.title("Discovered Topic Sizes (NMF)")
    plt.xlabel("Posts")
    plt.ylabel("Topic")
    savefig(viz / "10_topic_sizes_nmf.png")

    heat = doc.groupby(["week", "topic_label"]).size().reset_index(name="count")
    heat["share"] = heat["count"] / heat.groupby("week")["count"].transform("sum")
    pivot = heat.pivot(index="week", columns="topic_label", values="share").fillna(0)
    plt.figure(figsize=(13, 6))
    sns.heatmap(pivot, cmap="viridis", cbar_kws={"label": "Weekly share"})
    plt.title("Weekly Topic Share Heatmap")
    savefig(viz / "11_weekly_topic_share_heatmap.png")

    # 2D SVD projection for a lightweight embedding visualization.
    svd2 = TruncatedSVD(n_components=2, random_state=cfg.random_state)
    xy = svd2.fit_transform(X)
    plot_df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "topic": doc["topic_label"], "any_harmful": doc["any_harmful"]})
    if len(plot_df) > 20000:
        plot_df = plot_df.sample(20000, random_state=cfg.random_state)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="topic", s=10, alpha=0.55, linewidth=0)
    plt.title("2D Projection of Posts Colored by Discovered Topic")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    savefig(viz / "12_topic_projection_svd.png")

    label_topic = doc.groupby("topic_label")[list(cfg.labels)].mean()
    plt.figure(figsize=(10, 6))
    sns.heatmap(label_topic, cmap="magma", annot=True, fmt=".2f", cbar_kws={"label": "Label prevalence"})
    plt.title("Harmful-Content Prevalence by Discovered Topic")
    savefig(viz / "13_topic_by_harmful_label_heatmap.png")

    sent_topic = pd.crosstab(doc["topic_label"], doc["sentiment_vader_label"], normalize="index")
    plt.figure(figsize=(10, 6))
    sns.heatmap(sent_topic, cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Sentiment Distribution by Discovered Topic")
    savefig(viz / "14_topic_by_sentiment_heatmap.png")

    return TopicResult(doc, topic_words, topic_summary, topic_metrics)

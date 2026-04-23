from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def preprocessing_visuals(df: pd.DataFrame, out_dir: Path, labels: tuple[str, ...]) -> None:
    viz = out_dir / "visualizations"
    key_cols = ["text", "created_at", "user_location_USA_state", "sentiment_vader_raw", "sentiment_vader_label", *labels]
    miss = df[[c for c in key_cols if c in df.columns]].isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(9, 4))
    sns.barplot(x=miss.values, y=miss.index)
    plt.title("Missingness Rate for Key Columns")
    plt.xlabel("Missing share")
    plt.ylabel("Column")
    savefig(viz / "01_missingness_key_columns.png")

    plt.figure(figsize=(9, 4))
    weekly = df.groupby("week").size().reset_index(name="posts")
    sns.lineplot(data=weekly, x="week", y="posts", marker="o")
    plt.title("Weekly Post Volume")
    plt.xlabel("Week")
    plt.ylabel("Posts")
    savefig(viz / "02_weekly_volume_overview.png")

    plt.figure(figsize=(8, 4))
    sns.histplot(df["word_count_clean"], bins=40)
    plt.title("Clean Text Word Count Distribution")
    plt.xlabel("Words")
    savefig(viz / "03_word_count_distribution.png")

    plt.figure(figsize=(9, 4))
    label_counts = df[list(labels)].sum().sort_values(ascending=False)
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title("Harmful-Content Label Counts")
    plt.ylabel("Number of posts")
    plt.xticks(rotation=25, ha="right")
    savefig(viz / "04_harmful_label_counts.png")

    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="harmful_label_count")
    plt.title("Number of Harmful Labels per Post")
    plt.xlabel("Label count")
    plt.ylabel("Posts")
    savefig(viz / "05_multilabel_density.png")

    top_states = df["user_location_USA_state"].dropna().value_counts().head(15)
    if not top_states.empty:
        plt.figure(figsize=(9, 5))
        sns.barplot(x=top_states.values, y=top_states.index)
        plt.title("Top Self-Reported U.S. States")
        plt.xlabel("Posts")
        plt.ylabel("State")
        savefig(viz / "06_top_states_self_reported.png")

    # Simple UpSet-like combination chart without extra dependency.
    combos = df[list(labels)].astype(str).agg("".join, axis=1).value_counts().head(15)
    combo_names = []
    for code in combos.index:
        active = [lab for lab, bit in zip(labels, code) if bit == "1"]
        combo_names.append(" + ".join(active) if active else "No harmful label")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=combos.values, y=combo_names)
    plt.title("Top Harmful Label Combinations")
    plt.xlabel("Posts")
    plt.ylabel("Combination")
    savefig(viz / "07_top_label_combinations.png")


def rule_network(rules: pd.DataFrame, path: Path, max_rules: int = 25) -> None:
    if rules.empty:
        return
    G = nx.DiGraph()
    for _, row in rules.head(max_rules).iterrows():
        ant = str(row["antecedents"])
        con = str(row["consequents"])
        G.add_edge(ant, con, weight=float(row.get("lift", 1)))
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=0.9)
    widths = [max(1, G[u][v]["weight"]) for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=900, alpha=0.85)
    nx.draw_networkx_edges(G, pos, width=widths, arrows=True, alpha=0.45)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Association Rule Network: Antecedents → Consequents")
    plt.axis("off")
    savefig(path)

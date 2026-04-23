from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from .config import Config
from .visuals import savefig, rule_network


def _items_for_row(row: pd.Series, labels: tuple[str, ...], top_hashtags: set[str]) -> list[str]:
    items = [
        f"topic={row['topic_label']}",
        f"sentiment={row['sentiment_5bin']}",
        f"engagement={row['engagement_bucket']}",
        f"period={row['election_period']}",
        f"verified={int(bool(row['verified']))}",
        f"sensitive={int(bool(row['possibly_sensitive']))}",
        f"any_harmful={int(row['any_harmful'])}",
    ]
    for lab in labels:
        if int(row[lab]) == 1:
            items.append(f"label={lab}")
    for tag in row.get("hashtag_tokens", []):
        if tag in top_hashtags:
            items.append(f"hashtag={tag}")
    return sorted(set(map(str, items)))


def run_association_mining(df: pd.DataFrame, cfg: Config):
    out = cfg.output_dir / "association"
    viz = cfg.output_dir / "visualizations"

    all_tags = df["hashtag_tokens"].explode().dropna().astype(str)
    top_hashtags = set(all_tags.value_counts().head(cfg.max_hashtags_for_rules).index)

    transactions = [_items_for_row(row, cfg.labels, top_hashtags) for _, row in df.iterrows()]
    pd.DataFrame({"id": df["id"], "items": transactions}).to_parquet(out / "association_transactions.parquet", index=False)

    te = TransactionEncoder()
    enc = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(enc, columns=te.columns_)

    itemsets = fpgrowth(basket, min_support=cfg.association_min_support, use_colnames=True)
    if itemsets.empty:
        rules = pd.DataFrame()
    else:
        rules = association_rules(itemsets, metric="confidence", min_threshold=cfg.association_min_confidence)
        rules = rules[rules["lift"] >= cfg.association_min_lift].copy()
        rules["antecedents"] = rules["antecedents"].map(lambda s: ", ".join(sorted(s)))
        rules["consequents"] = rules["consequents"].map(lambda s: ", ".join(sorted(s)))
        # Prioritize rules that conclude harmful labels, sentiment, or high engagement.
        important = rules["consequents"].str.contains("label=|any_harmful=1|engagement=high|sentiment=", regex=True)
        rules = rules[important].sort_values(["lift", "confidence", "support"], ascending=False).head(cfg.association_rule_limit)

    itemsets_out = itemsets.copy()
    if not itemsets_out.empty:
        itemsets_out["itemsets"] = itemsets_out["itemsets"].map(lambda s: ", ".join(sorted(s)))
    itemsets_out.to_csv(out / "frequent_itemsets.csv", index=False)
    rules.to_csv(out / "association_rules.csv", index=False)

    if not rules.empty:
        plt.figure(figsize=(10, 6))
        plot = rules.head(20).copy()
        plot["rule"] = plot["antecedents"] + " → " + plot["consequents"]
        plot = plot.sort_values("lift")
        sns.barplot(data=plot, x="lift", y="rule")
        plt.title("Top Association Rules by Lift")
        plt.xlabel("Lift")
        plt.ylabel("Rule")
        savefig(viz / "30_top_association_rules_lift.png")

        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=rules, x="support", y="confidence", size="lift", hue="lift", sizes=(40, 300), alpha=0.75)
        plt.title("Association Rules: Support vs Confidence")
        savefig(viz / "31_association_rules_bubble.png")
        rule_network(rules, viz / "32_association_rule_network.png")

    return itemsets, rules

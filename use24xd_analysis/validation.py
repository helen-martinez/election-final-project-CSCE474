from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, cohen_kappa_score
from .config import Config
from .visuals import savefig

HUMAN_MAP = {
    "Conspiracy": "conspiracy_majority",
    "Hate_Speech": "hate_speech_majority",
    "Satire": "satire_majority",
    "Sensationalism": "sensationalism_majority",
    "Speculation": "speculation_majority",
}
ANNOTATOR_MAP = {
    "Conspiracy": ["conspiracy1", "conspiracy2", "conspiracy3"],
    "Hate_Speech": ["hate_speech1", "hate_speech2", "hate_speech3"],
    "Satire": ["satire1", "satire2", "satire3"],
    "Sensationalism": ["sensationalism1", "sensationalism2", "sensationalism3"],
    "Speculation": ["speculation1", "speculation2", "speculation3"],
}


def run_validation(main_df: pd.DataFrame, cfg: Config) -> None:
    out = cfg.output_dir / "validation"
    viz = cfg.output_dir / "visualizations"
    human = pd.read_csv(cfg.human_csv, low_memory=False)
    human["id"] = human["id"].astype(str)
    main = main_df.copy()
    main["id"] = main["id"].astype(str)
    merged = main.merge(human, on="id", how="inner", suffixes=("", "_human"))
    merged.to_parquet(out / "merged_human_validation_subset.parquet", index=False)

    if merged.empty:
        raise ValueError("No overlap found between main dataset and human annotation subset using id.")

    metric_rows = []
    error_rows = []
    for label, human_col in HUMAN_MAP.items():
        if label not in merged.columns or human_col not in merged.columns:
            continue
        y_pred = pd.to_numeric(merged[label], errors="coerce").fillna(0).astype(int)
        y_true = pd.to_numeric(merged[human_col], errors="coerce").fillna(0).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metric_rows.append({
            "label": label, "n": len(y_true), "accuracy": acc, "precision": precision,
            "recall": recall, "f1": f1, "true_positive": tp, "false_positive": fp,
            "true_negative": tn, "false_negative": fn, "human_positive_rate": y_true.mean(),
            "dataset_positive_rate": y_pred.mean(),
        })
        mism = merged[y_true.values != y_pred.values].copy().head(25)
        for _, row in mism.iterrows():
            error_rows.append({
                "label": label, "id": row["id"], "human_majority": int(row[human_col]),
                "dataset_label": int(row[label]), "text": row.get("text", "")[:350],
                "sentiment": row.get("sentiment_vader_label", ""), "engagement_total": row.get("engagement_total", np.nan),
            })

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.figure(figsize=(4.8, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["Human 0", "Human 1"])
        plt.title(f"Validation Confusion Matrix: {label}")
        savefig(viz / f"50_validation_confusion_{label}.png")

    metrics = pd.DataFrame(metric_rows).sort_values("f1", ascending=False)
    metrics.to_csv(out / "label_validation_metrics.csv", index=False)
    pd.DataFrame(error_rows).to_csv(out / "label_validation_error_examples.csv", index=False)

    plt.figure(figsize=(10, 5))
    plot = metrics.melt(id_vars="label", value_vars=["precision", "recall", "f1"], var_name="metric", value_name="score")
    sns.barplot(data=plot, x="label", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.title("Dataset Labels vs Human Majority Labels")
    plt.xticks(rotation=25, ha="right")
    savefig(viz / "51_validation_precision_recall_f1.png")

    # Human annotator agreement: pairwise Cohen kappa and exact three-way agreement.
    agreement_rows = []
    for label, cols in ANNOTATOR_MAP.items():
        if not all(c in human.columns for c in cols):
            continue
        sub = human[cols].apply(pd.to_numeric, errors="coerce").dropna().astype(int)
        kappas = []
        for i in range(3):
            for j in range(i + 1, 3):
                kappas.append(cohen_kappa_score(sub.iloc[:, i], sub.iloc[:, j]))
        exact = (sub.nunique(axis=1) == 1).mean()
        agreement_rows.append({"label": label, "exact_three_annotator_agreement": exact, "mean_pairwise_cohen_kappa": float(np.mean(kappas))})
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(out / "human_annotator_agreement.csv", index=False)

    if not agreement.empty:
        plt.figure(figsize=(10, 5))
        p = agreement.melt(id_vars="label", value_vars=["exact_three_annotator_agreement", "mean_pairwise_cohen_kappa"], var_name="agreement_metric", value_name="value")
        sns.barplot(data=p, x="label", y="value", hue="agreement_metric")
        plt.title("Human Annotator Agreement by Label")
        plt.xticks(rotation=25, ha="right")
        savefig(viz / "52_human_annotator_agreement.png")

    # Human positive prevalence compared with dataset prevalence on matched subset.
    if not metrics.empty:
        prev = metrics.melt(id_vars="label", value_vars=["human_positive_rate", "dataset_positive_rate"], var_name="source", value_name="positive_rate")
        plt.figure(figsize=(10, 5))
        sns.barplot(data=prev, x="label", y="positive_rate", hue="source")
        plt.title("Positive Label Rate: Human Majority vs Dataset Label")
        plt.xticks(rotation=25, ha="right")
        savefig(viz / "53_validation_positive_rate_comparison.png")

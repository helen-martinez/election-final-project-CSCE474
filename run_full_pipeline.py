from __future__ import annotations
import argparse
from pathlib import Path
from use24xd_analysis.config import Config
from use24xd_analysis.data import load_and_preprocess
from use24xd_analysis.visuals import preprocessing_visuals
from use24xd_analysis.topics import run_topic_modeling
from use24xd_analysis.sentiment import run_sentiment_analysis
from use24xd_analysis.association import run_association_mining
from use24xd_analysis.timeseries import run_timeseries_analysis
from use24xd_analysis.validation import run_validation


def parse_args():
    p = argparse.ArgumentParser(description="Run full USE24-XD election discourse analysis pipeline.")
    p.add_argument("--dataset_csv", default="data/U.S_Election_2024_Xcom_Dataset.csv")
    p.add_argument("--human_csv", default="data/Human_Annotation_Subset.csv")
    p.add_argument("--output_dir", default="output/use24xd_analysis")
    p.add_argument("--max_rows", type=int, default=None, help="Optional row cap for testing.")
    p.add_argument("--n_topics", type=int, default=12)
    p.add_argument("--min_support", type=float, default=0.015)
    p.add_argument("--min_confidence", type=float, default=0.35)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        dataset_csv=Path(args.dataset_csv),
        human_csv=Path(args.human_csv),
        output_dir=Path(args.output_dir),
        max_rows=args.max_rows,
        n_topics=args.n_topics,
        association_min_support=args.min_support,
        association_min_confidence=args.min_confidence,
    )
    cfg.ensure_dirs()

    print("[1/7] Loading and preprocessing dataset...")
    df = load_and_preprocess(cfg)
    print(f"Rows after preprocessing: {len(df):,}")

    print("[2/7] Creating preprocessing/data quality visualizations...")
    preprocessing_visuals(df, cfg.output_dir, cfg.labels)

    print("[3/7] Running unsupervised topic modeling and clustering...")
    topic_result = run_topic_modeling(df, cfg)
    df_topics = topic_result.document_topics

    print("[4/7] Running sentiment analysis and visualizations...")
    df_sentiment = run_sentiment_analysis(df_topics, cfg)

    print("[5/7] Running association rule mining...")
    run_association_mining(df_sentiment, cfg)

    print("[6/7] Running time-series analysis...")
    run_timeseries_analysis(df_sentiment, cfg)

    print("[7/7] Running validation against human annotation subset...")
    run_validation(df_sentiment, cfg)

    final_path = cfg.output_dir / "final_model_ready_dataset.parquet"
    df_sentiment.to_parquet(final_path, index=False)
    print("Done.")
    print(f"Final enriched dataset: {final_path.resolve()}")
    print(f"Outputs: {cfg.output_dir.resolve()}")


if __name__ == "__main__":
    main()

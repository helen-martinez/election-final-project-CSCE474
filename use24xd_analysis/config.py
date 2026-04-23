from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Input files
    dataset_csv: Path = Path("data/U.S_Election_2024_Xcom_Dataset.csv")
    human_csv: Path = Path("data/Human_Annotation_Subset.csv")
    output_dir: Path = Path("output/use24xd_analysis")

    # Election date
    election_day: str = "2024-11-05"

    # Runtime controls
    random_state: int = 42
    max_rows: int | None = None      # set e.g. 50000 for testing
    min_text_words: int = 3

    # Topic modeling controls
    tfidf_max_features: int = 8000
    tfidf_min_df: int = 5
    tfidf_max_df: float = 0.85
    n_topics: int = 12
    top_words_per_topic: int = 15
    n_representative_posts: int = 8

    # Association mining controls
    association_min_support: float = 0.015
    association_min_confidence: float = 0.35
    association_min_lift: float = 1.1
    association_rule_limit: int = 75
    max_hashtags_for_rules: int = 30

    # Time series controls
    rolling_window_weeks: int = 3

    labels: tuple[str, ...] = (
        "Speculation", "Sensationalism", "Conspiracy", "Hate_Speech", "Satire"
    )

    def ensure_dirs(self) -> None:
        for sub in [
            "preprocessing", "topics", "sentiment", "association", "timeseries",
            "validation", "visualizations", "report_tables"
        ]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

# USE24-XD Election Discourse Analysis Pipeline

This pipeline uses the USE24-XD dataset: https://github.com/Sensify-Lab/USE24-XD ('U.S_Election_2024_Xcom_Dataset.csv') & ('Human_Annotation_Subset.csv')

1. Data preprocessing
2. Topic detection and clustering
3. Sentiment and association analysis
4. Time-series analysis

It also has a validation stage using `Human_Annotation_Subset.csv`.

## Folder setup

Create this structure in VS Code:

```text
project_folder/
├── data/
│   ├── U.S_Election_2024_Xcom_Dataset.csv
│   └── Human_Annotation_Subset.csv
├── run_full_pipeline.py
├── requirements.txt
└── use24xd_analysis/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── visuals.py
    ├── topics.py
    ├── sentiment.py
    ├── association.py
    ├── timeseries.py
    └── validation.py
```

## Install dependencies

```powershell
pip install -r requirements.txt
```

## Test run first

```powershell
python run_full_pipeline.py --max_rows 10000 --n_topics 8
```

## Full run

```powershell
python run_full_pipeline.py --n_topics 12
```

## Main outputs

The pipeline writes results to:

```text
output/use24xd_analysis/
```

Important outputs:

Event Window
- event_window_harmful_share.png

Weekly Harmful Share
- timeseries_harmful_share.png

Harmful Categories Over Time
- timeseries_harmful_labels.png

Topic Stream Plot
- topic_stream_area.png

Heatmap
- label_heatmap.png

Post Volume
- weekly_post_volume.png


## Validation interpretation

The validation stage compares the main dataset labels against the human majority-vote labels:

- `Conspiracy` vs `conspiracy_majority`
- `Hate_Speech` vs `hate_speech_majority`
- `Satire` vs `satire_majority`
- `Sensationalism` vs `sensationalism_majority`
- `Speculation` vs `speculation_majority`

It reports accuracy, precision, recall, F1, confusion matrices, human positive rates, dataset positive rates, and human annotator agreement.



import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import statsmodels.api as sm

# Load dataset
df = pd.read_csv("data/U.S_Election_2024_Xcom_Dataset.csv")

# Convert datetime
df["date"] = pd.to_datetime(
    df["created_at"],
    format="ISO8601",
    errors="coerce",
    utc=True
)

print("Missing dates:", df["date"].isna().sum())

# Define election day
election_day = pd.Timestamp("2024-11-05", tz="UTC")

#Creating binary variable for harmful content
harmful_cols = ["Speculation", "Sensationalism", "Conspiracy", "Hate_Speech", "Satire"]
df["any_harmful"] = df[harmful_cols].max(axis=1)

#Event window t-test
# Define 14-day window
window = 14

before_window = df[
    (df["date"] >= election_day - pd.Timedelta(days=window)) &
    (df["date"] < election_day)
]["any_harmful"]

after_window = df[
    (df["date"] >= election_day) &
    (df["date"] <= election_day + pd.Timedelta(days=window))
]["any_harmful"]

# Run Welch's t-test
t_stat, p_value = ttest_ind(before_window, after_window, equal_var=False)

print("=== EVENT WINDOW T-TEST ===")
print("Mean BEFORE:", before_window.mean())
print("Mean AFTER:", after_window.mean())
print("Difference:", after_window.mean() - before_window.mean())
print("T-statistic:", t_stat)
print("P-value:", p_value)


#### Logistic regression
df["post_election"] = (df["date"] >= election_day).astype(int)

engagement_cols = [
    "public_metrics.like_count",
    "public_metrics.retweet_count",
    "public_metrics.reply_count",
    "public_metrics.quote_count"
]

df_model = df.dropna(subset=[
    "any_harmful",
    "post_election",
    *engagement_cols
]).copy()

# Convert engagement columns to numeric
for col in engagement_cols:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0)

# Log transform engagement variables
for col in engagement_cols:
    df_model[col] = np.log1p(df_model[col])

X = df_model[[
    "post_election",
    "public_metrics.like_count",
    "public_metrics.retweet_count",
    "public_metrics.reply_count",
    "public_metrics.quote_count"
]]

# Rename columns for cleaner regression output
X.columns = [
    "post_election",
    "log_likes",
    "log_retweets",
    "log_replies",
    "log_quotes"
]

y = df_model["any_harmful"]

X = sm.add_constant(X)

model = sm.Logit(y, X).fit()

print(model.summary())

odds_ratios = np.exp(model.params)
print("\nOdds Ratios:\n", odds_ratios)
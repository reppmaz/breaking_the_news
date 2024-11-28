import pandas as pd

df_left = pd.read_csv('/Users/reppmazc/Documents/IRONHACK/quests/final_project/breaking_the_news/breaking_news_data.csv')
df_right = pd.read_csv('/Users/reppmazc/Documents/IRONHACK/quests/final_project/breaking_the_news/test.csv')

# Merge the two DataFrames
df_combined = pd.concat([df_left, df_right])

# Identify duplicates based on the 'url' column
# Keep rows from the right DataFrame only if they have no NaNs
df_combined = df_combined.sort_values(by='url', ascending=False)  # Ensure right rows come first
mask_no_nans = ~df_combined.isna().any(axis=1)  # True if no NaNs in the row
df_combined = df_combined.loc[mask_no_nans | ~df_combined.duplicated(subset=['url'], keep=False)]

# Keep only columns from the left DataFrame
df_final = df_combined[df_left.columns]

df_final.to_csv('/Users/reppmazc/Documents/IRONHACK/quests/final_project/breaking_the_news/breaking_news_data_new.csv')

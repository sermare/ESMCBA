# other_utils.py

from imports import *

def split_dataset(df, test_size=0.2, random_state=42):
    """
    Example function that splits a DataFrame into train/test sets.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def compute_correlations(x, y):
    """
    Example function returning Spearman and Pearson correlations.
    """
    return spearmanr(x, y)[0], pearsonr(x, y)[0]

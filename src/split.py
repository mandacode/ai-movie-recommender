"""Train/test split.

A realistic recommender must be trained on the past and evaluated on the
future — never a random split, which would leak future ratings into training.

We use a **per-user leave-last-N** protocol: for every user we hold out their
own N most-recent ratings for testing and keep everything earlier for training.
Time order is preserved *per user*, so we still train on the past and test on
the future, while evaluating essentially every user — which gives far more
stable metrics than a single global time cutoff (that would concentrate the
test set in a narrow recent window and leave only a handful of users to score).
"""
from __future__ import annotations

import pandas as pd


def per_user_leave_last_n(
    ratings: pd.DataFrame,
    n: int = 10,
    time_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's ``n`` most-recent ratings for testing.

    For every user we sort their ratings by time and move the last ``n`` into
    the test set; everything earlier stays in train.

    Args:
        ratings: rating events with a sortable ``time_col``.
        n: number of most-recent ratings held out per user.
        time_col: column to order by.

    Returns:
        ``(train, test)`` DataFrames.
    """
    ordered = ratings.sort_values(time_col, kind="stable")
    test = ordered.groupby("userId", group_keys=False).tail(n)
    train = ordered.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)

from __future__ import annotations

import re

import numpy as np
import pandas as pd

__all__ = ["add_catch_trial_column", "catch_trial_mask"]


def _normalize_condition_name(condition):
    if pd.isna(condition):
        return np.nan
    c = str(condition).strip().lower()
    if c == "":
        return np.nan
    if c.startswith("stim"):
        return "stimulation"
    if c.startswith("wash"):
        return "washout"
    if c.startswith("base"):
        return "baseline"
    return c


def _parse_condition_epoch_label(label):
    if pd.isna(label):
        return np.nan, np.nan
    s = str(label).strip().lower()
    if s == "":
        return np.nan, np.nan
    if s.startswith("baseline"):
        return "baseline", 0
    match = re.match(r"([a-z]+)_epoch(?:_(\d+))?$", s)
    if match:
        cond = _normalize_condition_name(match.group(1))
        epoch_id = int(match.group(2)) if match.group(2) is not None else 0
        return cond, epoch_id
    return _normalize_condition_name(s), np.nan


def _ideal_condition_series(event_meta: pd.DataFrame, *, condition_epoch_col: str, condition_col: str) -> pd.Series:
    if condition_epoch_col in event_meta.columns:
        parsed = event_meta[condition_epoch_col].map(_parse_condition_epoch_label)
        return parsed.map(lambda value: value[0] if isinstance(value, tuple) else np.nan)
    if condition_col in event_meta.columns:
        return event_meta[condition_col].map(_normalize_condition_name)
    raise ValueError(
        f"event_meta must contain '{condition_epoch_col}' or '{condition_col}' to infer the ideal condition."
    )


def _actual_condition_series(event_meta: pd.DataFrame, *, real_condition_col: str, condition_col: str) -> pd.Series:
    fallback = (
        event_meta[condition_col]
        if condition_col in event_meta.columns
        else pd.Series(index=event_meta.index, dtype=object)
    )
    if real_condition_col in event_meta.columns:
        return event_meta[real_condition_col].where(event_meta[real_condition_col].notna(), fallback).map(
            _normalize_condition_name
        )
    if condition_col in event_meta.columns:
        return fallback.map(_normalize_condition_name)
    raise ValueError(
        f"event_meta must contain '{real_condition_col}' or '{condition_col}' to infer the actual condition."
    )


def catch_trial_mask(
    event_meta: pd.DataFrame,
    *,
    condition_epoch_col: str = "condition_epoch",
    real_condition_col: str = "real_condition",
    condition_col: str = "condition",
) -> pd.Series:
    """
    Mirror the catch-trial rule used by the PCA plotting code.

    A row is a catch trial when the actual condition differs from the ideal
    condition implied by ``condition_epoch``.
    """
    if not isinstance(event_meta, pd.DataFrame):
        raise TypeError("event_meta must be a pandas DataFrame.")

    ideal_condition = _ideal_condition_series(
        event_meta,
        condition_epoch_col=condition_epoch_col,
        condition_col=condition_col,
    )
    actual_condition = _actual_condition_series(
        event_meta,
        real_condition_col=real_condition_col,
        condition_col=condition_col,
    )

    mask = actual_condition.notna() & ideal_condition.notna() & (actual_condition != ideal_condition)
    return mask.astype(bool).rename("catch_trial")


def add_catch_trial_column(
    event_meta: pd.DataFrame,
    *,
    output_col: str = "catch_trial",
    condition_epoch_col: str = "condition_epoch",
    real_condition_col: str = "real_condition",
    condition_col: str = "condition",
    copy: bool = True,
) -> pd.DataFrame:
    """
    Return a dataframe with a boolean catch-trial column added.

    Example
    -------
    ``pca_event_meta = add_catch_trial_column(pca_event_meta)``
    """
    out = event_meta.copy() if copy else event_meta
    out[output_col] = catch_trial_mask(
        out,
        condition_epoch_col=condition_epoch_col,
        real_condition_col=real_condition_col,
        condition_col=condition_col,
    ).to_numpy(dtype=bool)
    return out

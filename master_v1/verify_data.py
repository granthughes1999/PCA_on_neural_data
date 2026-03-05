from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import json
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

def check_stim_event_timing(df_stim, max_window=4.0, show_detailed_output=True) -> dict[str, Any]:
    """
    Compute average time differences between task events and return
    averages, counts, and the actual valid time pairs used.

    Parameters
    ----------
    df_stim : pandas.DataFrame
        Must contain columns ['stimulus', 'start_time'].
    max_window : float
        Maximum allowed time difference (seconds) for valid pairing.

    Returns
    -------
    results : dict
        Dictionary containing averages, pair counts, and valid_pairs.
    """

    import numpy as np

    # --- Extract event times ---
    all_stimROI_triggers = df_stim[df_stim['stimulus'] == 'reachInit_stimROI_timestamps']
    stim_ROI_df          = df_stim[df_stim['stimulus'] == 'stimROI_timestamps']
    optical_df           = df_stim[df_stim['stimulus'] == 'optical_timestamps']
    tone2_df             = df_stim[df_stim['stimulus'] == 'tone2_timestamps']
    tone1_df             = df_stim[df_stim['stimulus'] == 'tone1_timestamps']

    tone1_start_times = tone1_df['start_time'].values
    tone2_start_times = tone2_df['start_time'].values
    stimROI_start_times = stim_ROI_df['start_time'].values
    optical_start_times = optical_df['start_time'].values
    all_stimROI_triggers_start_times = all_stimROI_triggers['start_time'].values

    def compute_avg_diff(reference_times, target_times, max_window):
        valid_pairs = []

        if len(reference_times) == 0 or len(target_times) == 0:
            return np.nan, 0, valid_pairs

        for t_ref in reference_times:
            idx = np.argmin(np.abs(target_times - t_ref))
            closest = target_times[idx]
            if 0 < closest - t_ref < max_window:
                valid_pairs.append((t_ref, closest))

        if len(valid_pairs) == 0:
            return np.nan, 0, valid_pairs

        avg_diff = np.mean([t2 - t1 for t1, t2 in valid_pairs])
        return avg_diff, len(valid_pairs), valid_pairs

    # --- Compute pairwise relationships ---
    avg_t1_t2, n_t1_t2, pairs_t1_t2 = compute_avg_diff(tone1_start_times, tone2_start_times, max_window)
    avg_t1_stimROI, n_t1_stimROI, pairs_t1_stimROI = compute_avg_diff(tone1_start_times, stimROI_start_times, max_window)
    avg_t2_stimROI, n_t2_stimROI, pairs_t2_stimROI = compute_avg_diff(tone2_start_times, stimROI_start_times, max_window)
    avg_t1_allStimROI, n_t1_allStimROI, pairs_t1_allStimROI = compute_avg_diff(tone1_start_times, all_stimROI_triggers_start_times, max_window)
    avg_t2_allStimROI, n_t2_allStimROI, pairs_t2_allStimROI = compute_avg_diff(tone2_start_times, all_stimROI_triggers_start_times, max_window)
    avg_allStimROI_stimROI, n_allStimROI_stimROI, pairs_allStimROI_stimROI = compute_avg_diff(
        all_stimROI_triggers_start_times, stimROI_start_times, max_window
    )

    # --- Print structured summary ---
    print('=== Average time differences between events (valid pairs within expected window): ===')

    print('\n---- Expected ~2 s -----')
    print('tone1 and tone2: ',
          None if np.isnan(avg_t1_t2) else round(avg_t1_t2, 2),
          f'({n_t1_t2} pairs)\n')

    print('---- These two should be similar -----')
    print('tone1 and stimROI:             ',
          None if np.isnan(avg_t1_stimROI) else round(avg_t1_stimROI, 2),
          f'({n_t1_stimROI} pairs)')
    print('tone1 and all_stimROI_triggers:',
          None if np.isnan(avg_t1_allStimROI) else round(avg_t1_allStimROI, 2),
          f'({n_t1_allStimROI} pairs)\n')

    print('---- These two should be similar -----')
    print('tone2 and stimROI:             ',
          None if np.isnan(avg_t2_stimROI) else round(avg_t2_stimROI, 2),
          f'({n_t2_stimROI} pairs)')
    print('tone2 and all_stimROI_triggers:',
          None if np.isnan(avg_t2_allStimROI) else round(avg_t2_allStimROI, 2),
          f'({n_t2_allStimROI} pairs)\n')

    print('---- Should be near zero -----')
    print('all_stimROI_triggers and stimROI:',
          None if np.isnan(avg_allStimROI_stimROI) else round(avg_allStimROI_stimROI, 2),
          f'({n_allStimROI_stimROI} pairs)')

    # --- Return structured results ---
    results = {
        'tone1_tone2': {
            'avg_diff': avg_t1_t2,
            'n_pairs': n_t1_t2,
            'valid_pairs': pairs_t1_t2
        },
        'tone1_stimROI': {
            'avg_diff': avg_t1_stimROI,
            'n_pairs': n_t1_stimROI,
            'valid_pairs': pairs_t1_stimROI
        },
        'tone2_stimROI': {
            'avg_diff': avg_t2_stimROI,
            'n_pairs': n_t2_stimROI,
            'valid_pairs': pairs_t2_stimROI
        },
        'tone1_allStimROI': {
            'avg_diff': avg_t1_allStimROI,
            'n_pairs': n_t1_allStimROI,
            'valid_pairs': pairs_t1_allStimROI
        },
        'tone2_allStimROI': {
            'avg_diff': avg_t2_allStimROI,
            'n_pairs': n_t2_allStimROI,
            'valid_pairs': pairs_t2_allStimROI
        },
        'allStimROI_stimROI': {
            'avg_diff': avg_allStimROI_stimROI,
            'n_pairs': n_allStimROI_stimROI,
            'valid_pairs': pairs_allStimROI_stimROI
        }
    }

    if show_detailed_output:
        print('\n=== Detailed valid time pairs (within expected window) ===')
        for key, data in results.items():
            print(f'\n--- {key} ---')
            for t1, t2 in data['valid_pairs']:
                print(f'  {t1:.3f} s  -->  {t2:.3f} s  (diff: {t2 - t1:.3f} s)')

    return results

def _split_epochs_into_contiguous_20_trial_chunks(epochs, trials_per_epoch=20, mode="raise"):
    """
    epochs: list[list[int]] (output of _to_epochs)
    Splits each epoch by gaps (>1) and then chunks each contiguous block into trials_per_epoch.
    mode:
      - "raise": error if any chunk size != trials_per_epoch
      - "drop_incomplete": drop chunks != trials_per_epoch
      - "keep_incomplete": keep chunks != trials_per_epoch
    Returns: (new_epochs, report)
    """
    def _split_on_gaps(trials):
        t = np.array(sorted(set(trials)), dtype=int)
        if t.size == 0:
            return []
        split_ix = np.where(np.diff(t) > 1)[0] + 1
        blocks = np.split(t, split_ix)
        return [b.tolist() for b in blocks if len(b) > 0]

    def _chunk(block):
        b = np.array(sorted(set(block)), dtype=int)
        chunks = []
        for i in range(0, len(b), trials_per_epoch):
            chunks.append(b[i:i + trials_per_epoch].tolist())
        return chunks

    new_epochs = []
    splits = []
    dropped = 0
    kept_incomplete = 0

    for ep_i, ep in enumerate(epochs, start=1):
        blocks = _split_on_gaps(ep)
        chunks = []
        for blk in blocks:
            chunks.extend(_chunk(blk))

        changed = (len(blocks) > 1) or any(len(c) != trials_per_epoch for c in chunks) or (len(chunks) != 1)

        if changed:
            splits.append({
                "original_epoch_index": ep_i,
                "original_n": len(ep),
                "original_range": (min(ep), max(ep)) if len(ep) else None,
                "n_blocks": len(blocks),
                "block_ranges": [(min(b), max(b), len(b)) for b in blocks] if blocks else [],
                "chunk_sizes": [len(c) for c in chunks],
                "n_chunks": len(chunks),
            })

        for c in chunks:
            if len(c) == trials_per_epoch:
                new_epochs.append(c)
            else:
                if mode == "raise":
                    raise ValueError(
                        f"Incomplete chunk n={len(c)} (expected {trials_per_epoch}) "
                        f"range={min(c)}..{max(c)} from original epoch {ep_i}."
                    )
                elif mode == "drop_incomplete":
                    dropped += 1
                elif mode == "keep_incomplete":
                    kept_incomplete += 1
                    new_epochs.append(c)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

    report = {
        "n_input_epochs": len(epochs),
        "n_output_epochs": len(new_epochs),
        "n_epochs_changed": len(splits),
        "splits": splits,
        "dropped_incomplete": dropped,
        "kept_incomplete": kept_incomplete,
        "mode": mode,
        "trials_per_epoch": trials_per_epoch,
    }
    return new_epochs, report


def plot_task_epoch_structure(
    baseline_trials_idx,
    optoicalStim_trials_idx,
    washout_trials_idx,
    *,
    trials_are_one_based=True,
    figsize=(14, 3),
):
    """
    Plot trial order structure.

    Baseline  = blue
    Stimulation = red
    Washout = green

    X-axis = trial number
    """

    def _flatten(x):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(x, (list, tuple)):
            out = []
            for v in x:
                out.extend(_flatten(v))
            return out
        return [int(x)]

    def _to_epochs(idx):
        if isinstance(idx, np.ndarray):
            idx = idx.tolist()
        if isinstance(idx, (list, tuple)) and len(idx) > 0 and any(isinstance(v, (list, tuple, np.ndarray)) for v in idx):
            return [sorted(set(_flatten(ep))) for ep in idx]
        return [sorted(set(_flatten(idx)))]

    base_epochs = _to_epochs(baseline_trials_idx)
    stim_epochs = _to_epochs(optoicalStim_trials_idx)
    wash_epochs = _to_epochs(washout_trials_idx)

    base = base_epochs[0]

    if trials_are_one_based:
        offset = 0
    else:
        offset = 1

    fig, ax = plt.subplots(figsize=figsize)

    # Baseline
    ax.scatter(base, np.ones(len(base))*1,
               color='blue', s=40, label='Baseline')

    # Stimulation epochs
    for i, ep in enumerate(stim_epochs):
        ax.scatter(ep, np.ones(len(ep))*2,
                   color='red', s=40,
                   label='Stimulation' if i == 0 else None)

    # Washout epochs
    for i, ep in enumerate(wash_epochs):
        ax.scatter(ep, np.ones(len(ep))*3,
                   color='green', s=40,
                   label='Washout' if i == 0 else None)

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Baseline', 'Stimulation', 'Washout'])
    ax.set_xlabel('Trial Number')
    ax.set_title('Task Epoch Order Structure')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    # Determine full trial span
    all_trials = (
        base
        + _flatten(stim_epochs)
        + _flatten(wash_epochs)
    )
    if len(all_trials) > 0:
        ax.set_xlim(min(all_trials) - 2, max(all_trials) + 2)

    plt.tight_layout()
    plt.show()


def _flatten(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(_flatten(v))
        return out
    return [int(x)]

def _to_epochs(idx, name):
    """
    baseline: allow flat list/array (single epoch)
    stim/wash: allow list-of-lists (multiple epochs)
    """
    if isinstance(idx, np.ndarray):
        idx = idx.tolist()

    # list-of-lists epochs
    if isinstance(idx, (list, tuple)) and len(idx) > 0 and any(isinstance(v, (list, tuple, np.ndarray)) for v in idx):
        epochs = []
        for ep in idx:
            ep_flat = sorted(set(_flatten(ep)))
            epochs.append(ep_flat)
        return epochs

    # single epoch
    return [sorted(set(_flatten(idx)))]




def verify_task_epoch_structure_noSplitting(
    baseline_trials_idx,
    optoicalStim_trials_idx,
    washout_trials_idx,
    *,
    trials_per_epoch=20,
    expect_stim_epochs=None,
    expect_wash_epochs=None,
    verbose=True,
):
    """
    Verifies:
      1) each epoch has exactly `trials_per_epoch` trials
      2) baseline is one contiguous epoch
      3) ordering is Baseline -> (Stim_i -> Wash_i) repeated
      4) no overlaps across conditions
    """

    # ---- OLD PRINTS (for context) ----
    if verbose:
        print('Total Baseline Trials: ', len(_flatten(baseline_trials_idx)))  # old code
        print('Total Washout epochs: ', len(_to_epochs(washout_trials_idx, "washout_trials_idx")))  # old code
        print('Total Optical Stim epochs: ', len(_to_epochs(optoicalStim_trials_idx, "optoicalStim_trials_idx")))  # old code
        print('')
        print('baseline_trials_idx', baseline_trials_idx)  # old code
        print('optoicalStim_trials_idx', optoicalStim_trials_idx)  # old code
        print('washout_trials_idx', washout_trials_idx)  # old code
        print('')

    # ---- NEW CODE ----
    base_epochs = _to_epochs(baseline_trials_idx, "baseline_trials_idx")
    stim_epochs = _to_epochs(optoicalStim_trials_idx, "optoicalStim_trials_idx")
    wash_epochs = _to_epochs(washout_trials_idx, "washout_trials_idx")

    errors = []
    warnings = []

    # baseline should be exactly 1 epoch
    if len(base_epochs) != 1:
        errors.append(f"Baseline should be 1 epoch, got {len(base_epochs)} epochs.")
    base = base_epochs[0]

    # counts per epoch
    if len(base) != trials_per_epoch:
        errors.append(f"Baseline epoch size = {len(base)} (expected {trials_per_epoch}).")

    for i, ep in enumerate(stim_epochs, 1):
        if len(ep) != trials_per_epoch:
            errors.append(f"Stim epoch {i} size = {len(ep)} (expected {trials_per_epoch}).")
    for i, ep in enumerate(wash_epochs, 1):
        if len(ep) != trials_per_epoch:
            errors.append(f"Wash epoch {i} size = {len(ep)} (expected {trials_per_epoch}).")

    # expected number of epochs (optional strictness)
    if expect_stim_epochs is not None and len(stim_epochs) != expect_stim_epochs:
        errors.append(f"Stim epochs = {len(stim_epochs)} (expected {expect_stim_epochs}).")
    if expect_wash_epochs is not None and len(wash_epochs) != expect_wash_epochs:
        errors.append(f"Wash epochs = {len(wash_epochs)} (expected {expect_wash_epochs}).")

    # overlap checks
    base_set = set(base)
    stim_set = set(_flatten(stim_epochs))
    wash_set = set(_flatten(wash_epochs))

    if base_set & stim_set:
        errors.append(f"Overlap baseline∩stim: {sorted(base_set & stim_set)[:20]}")
    if base_set & wash_set:
        errors.append(f"Overlap baseline∩wash: {sorted(base_set & wash_set)[:20]}")
    if stim_set & wash_set:
        errors.append(f"Overlap stim∩wash: {sorted(stim_set & wash_set)[:20]}")

    # contiguity helpers
    def _is_contiguous(ep):
        if len(ep) == 0:
            return True
        return (max(ep) - min(ep) + 1) == len(ep)

    def _describe_epoch(ep):
        if len(ep) == 0:
            return "empty"
        return f"{min(ep)}..{max(ep)} (n={len(ep)})"

    # baseline contiguous
    if not _is_contiguous(base):
        errors.append(f"Baseline not contiguous: {_describe_epoch(base)}")

    # stim/wash contiguous
    for i, ep in enumerate(stim_epochs, 1):
        if not _is_contiguous(ep):
            errors.append(f"Stim epoch {i} not contiguous: {_describe_epoch(ep)}")
    for i, ep in enumerate(wash_epochs, 1):
        if not _is_contiguous(ep):
            errors.append(f"Wash epoch {i} not contiguous: {_describe_epoch(ep)}")

    # ordering: baseline then alternating stim_i then wash_i
    # baseline must end immediately before stim_1 starts (by index continuity)
    if len(base) > 0 and len(stim_epochs) > 0:
        if min(stim_epochs[0]) <= max(base):
            errors.append(
                f"Stim epoch 1 starts at {min(stim_epochs[0])}, but baseline ends at {max(base)} (should be after)."
            )

    # enforce alternation by index ranges (stim_i should precede wash_i, and wash_i should precede stim_{i+1})
    n_pairs = min(len(stim_epochs), len(wash_epochs))
    if len(stim_epochs) != len(wash_epochs):
        warnings.append(f"Stim epochs ({len(stim_epochs)}) != Wash epochs ({len(wash_epochs)}); checking first {n_pairs} pairs.")

    for i in range(n_pairs):
        stim_ep = stim_epochs[i]
        wash_ep = wash_epochs[i]
        if len(stim_ep) and len(wash_ep):
            if min(wash_ep) <= max(stim_ep):
                errors.append(
                    f"Order violation: Wash epoch {i+1} starts at {min(wash_ep)} but Stim epoch {i+1} ends at {max(stim_ep)}."
                )
        if i < n_pairs - 1:
            next_stim = stim_epochs[i+1]
            if len(wash_ep) and len(next_stim):
                if min(next_stim) <= max(wash_ep):
                    errors.append(
                        f"Order violation: Stim epoch {i+2} starts at {min(next_stim)} but Wash epoch {i+1} ends at {max(wash_ep)}."
                    )

    # ---- REPORT ----
    if verbose:
        print("Epoch summaries (trial indices as provided):")
        print(f"  Baseline: {_describe_epoch(base)}")
        for i, ep in enumerate(stim_epochs, 1):
            print(f"  Stim {i}:  {_describe_epoch(ep)}")
        for i, ep in enumerate(wash_epochs, 1):
            print(f"  Wash {i}:  {_describe_epoch(ep)}")

        if warnings:
            print("\nWARNINGS:")
            for w in warnings:
                print(" -", w)

        if errors:
            print("\nFAILED checks:")
            for e in errors:
                print(" -", e)
        else:
            print("\nPASSED: Baseline -> Optical Stim -> Washout structure is consistent with 20 trials/epoch.")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def verify_task_epoch_structure(
    baseline_trials_idx,
    optoicalStim_trials_idx,
    washout_trials_idx,
    *,
    trials_per_epoch=20,
    expect_stim_epochs=None,
    expect_wash_epochs=None,
    auto_split_merged_epochs=True,
    split_mode="keep_incomplete",  # "raise" | "drop_incomplete" | "keep_incomplete"
    show_plot=False,
    verbose=True,
):
    """
    Verifies:
      1) each epoch has exactly `trials_per_epoch` trials
      2) baseline is one contiguous epoch
      3) ordering is Baseline -> (Stim_i -> Wash_i) repeated
      4) no overlaps across conditions

    If auto_split_merged_epochs=True, stim/wash epochs are first split by gaps and
    re-chunked into trials_per_epoch epochs, fixing cases where multiple epochs were merged.
    """

    if verbose:
        print('Total Baseline Trials: ', len(_flatten(baseline_trials_idx)))
        print('Total Washout epochs: ', len(_to_epochs(washout_trials_idx, "washout_trials_idx")))
        print('Total Optical Stim epochs: ', len(_to_epochs(optoicalStim_trials_idx, "optoicalStim_trials_idx")))
        print('')
        print('baseline_trials_idx', baseline_trials_idx)
        print('optoicalStim_trials_idx', optoicalStim_trials_idx)
        print('washout_trials_idx', washout_trials_idx)
        print('')

    base_epochs = _to_epochs(baseline_trials_idx, "baseline_trials_idx")
    stim_epochs = _to_epochs(optoicalStim_trials_idx, "optoicalStim_trials_idx")
    wash_epochs = _to_epochs(washout_trials_idx, "washout_trials_idx")

    split_reports = {}

    if auto_split_merged_epochs:
        stim_epochs, stim_report = _split_epochs_into_contiguous_20_trial_chunks(
            stim_epochs, trials_per_epoch=trials_per_epoch, mode=split_mode
        )
        wash_epochs, wash_report = _split_epochs_into_contiguous_20_trial_chunks(
            wash_epochs, trials_per_epoch=trials_per_epoch, mode=split_mode
        )
        split_reports = {"stim": stim_report, "wash": wash_report}

        if verbose:
            if stim_report["n_epochs_changed"] > 0:
                print(f"[auto_split] stim: {stim_report['n_input_epochs']} -> {stim_report['n_output_epochs']} epochs")
            if wash_report["n_epochs_changed"] > 0:
                print(f"[auto_split] wash: {wash_report['n_input_epochs']} -> {wash_report['n_output_epochs']} epochs")
            print('')

    if show_plot:
        plot_task_epoch_structure(base_epochs[0] if len(base_epochs) else [],
                                 stim_epochs, wash_epochs)

    errors = []
    warnings = []

    if len(base_epochs) != 1:
        errors.append(f"Baseline should be 1 epoch, got {len(base_epochs)} epochs.")
    base = base_epochs[0] if len(base_epochs) else []

    if len(base) != trials_per_epoch:
        errors.append(f"Baseline epoch size = {len(base)} (expected {trials_per_epoch}).")

    for i, ep in enumerate(stim_epochs, 1):
        if len(ep) != trials_per_epoch:
            errors.append(f"Stim epoch {i} size = {len(ep)} (expected {trials_per_epoch}).")
    for i, ep in enumerate(wash_epochs, 1):
        if len(ep) != trials_per_epoch:
            errors.append(f"Wash epoch {i} size = {len(ep)} (expected {trials_per_epoch}).")

    if expect_stim_epochs is not None and len(stim_epochs) != expect_stim_epochs:
        errors.append(f"Stim epochs = {len(stim_epochs)} (expected {expect_stim_epochs}).")
    if expect_wash_epochs is not None and len(wash_epochs) != expect_wash_epochs:
        errors.append(f"Wash epochs = {len(wash_epochs)} (expected {expect_wash_epochs}).")

    base_set = set(base)
    stim_set = set(_flatten(stim_epochs))
    wash_set = set(_flatten(wash_epochs))

    if base_set & stim_set:
        errors.append(f"Overlap baseline∩stim: {sorted(base_set & stim_set)[:20]}")
    if base_set & wash_set:
        errors.append(f"Overlap baseline∩wash: {sorted(base_set & wash_set)[:20]}")
    if stim_set & wash_set:
        errors.append(f"Overlap stim∩wash: {sorted(stim_set & wash_set)[:20]}")

    def _is_contiguous(ep):
        if len(ep) == 0:
            return True
        return (max(ep) - min(ep) + 1) == len(ep)

    def _describe_epoch(ep):
        if len(ep) == 0:
            return "empty"
        return f"{min(ep)}..{max(ep)} (n={len(ep)})"

    if not _is_contiguous(base):
        errors.append(f"Baseline not contiguous: {_describe_epoch(base)}")

    for i, ep in enumerate(stim_epochs, 1):
        if not _is_contiguous(ep):
            errors.append(f"Stim epoch {i} not contiguous: {_describe_epoch(ep)}")
    for i, ep in enumerate(wash_epochs, 1):
        if not _is_contiguous(ep):
            errors.append(f"Wash epoch {i} not contiguous: {_describe_epoch(ep)}")

    if len(base) > 0 and len(stim_epochs) > 0:
        if min(stim_epochs[0]) <= max(base):
            errors.append(
                f"Stim epoch 1 starts at {min(stim_epochs[0])}, but baseline ends at {max(base)} (should be after)."
            )

    n_pairs = min(len(stim_epochs), len(wash_epochs))
    if len(stim_epochs) != len(wash_epochs):
        warnings.append(f"Stim epochs ({len(stim_epochs)}) != Wash epochs ({len(wash_epochs)}); checking first {n_pairs} pairs.")

    for i in range(n_pairs):
        stim_ep = stim_epochs[i]
        wash_ep = wash_epochs[i]
        if len(stim_ep) and len(wash_ep):
            if min(wash_ep) <= max(stim_ep):
                errors.append(
                    f"Order violation: Wash epoch {i+1} starts at {min(wash_ep)} but Stim epoch {i+1} ends at {max(stim_ep)}."
                )
        if i < n_pairs - 1:
            next_stim = stim_epochs[i + 1]
            if len(wash_ep) and len(next_stim):
                if min(next_stim) <= max(wash_ep):
                    errors.append(
                        f"Order violation: Stim epoch {i+2} starts at {min(next_stim)} but Wash epoch {i+1} ends at {max(wash_ep)}."
                    )

    if verbose:
        print("Epoch summaries (trial indices as used for verification):")
        print(f"  Baseline: {_describe_epoch(base)}")
        for i, ep in enumerate(stim_epochs, 1):
            print(f"  Stim {i}:  {_describe_epoch(ep)}")
        for i, ep in enumerate(wash_epochs, 1):
            print(f"  Wash {i}:  {_describe_epoch(ep)}")

        if warnings:
            print("\nWARNINGS:")
            for w in warnings:
                print(" -", w)

        if errors:
            print("\nFAILED checks:")
            for e in errors:
                print(" -", e)
        else:
            print("\nPASSED: Baseline -> Optical Stim -> Washout structure is consistent with 20 trials/epoch.")

    # Convert to downstream-compatible structure (object arrays of lists)
    baseline_out = np.array([base], dtype=object)
    stim_out = np.array(stim_epochs, dtype=object)
    wash_out = np.array(wash_epochs, dtype=object)

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "used_epochs": {
            "baseline": baseline_out,
            "stim": stim_out,
            "wash": wash_out,
        },
        "split_reports": split_reports,
    }
    # return {
    #     "ok": len(errors) == 0,
    #     "errors": errors,
    #     "warnings": warnings,
    #     "used_epochs": {"baseline": base, "stim": stim_epochs, "wash": wash_epochs},
    #     "split_reports": split_reports,
    # }
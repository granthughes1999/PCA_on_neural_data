from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _sanitize_name(x: str) -> str:
    s = str(x)
    bad = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for b in bad:
        s = s.replace(b, '_')
    return s.strip().replace(' ', '_')


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def zscore_rows(x: np.ndarray) -> np.ndarray:
    ss = StandardScaler(with_mean=True, with_std=True)
    return ss.fit_transform(x.T).T


def _pca_norm_kslabel(v):
    if pd.isna(v):
        return np.nan
    try:
        iv = int(float(v))
        if iv == 2:
            return "good"
        if iv == 1:
            return "mua"
    except Exception:
        pass
    s = str(v).strip().lower()
    if s in {"2", "good", "single", "singleunit", "single_unit"}:
        return "good"
    if s in {"1", "mua", "multi", "multiunit", "multi_unit"}:
        return "mua"
    return s

_BC_LABEL_ALLOWED = {"NON-SOMA", "NOISE", "MUA", "GOOD"}


def _pca_norm_bc_label(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().upper()
    if s in {"", "NAN", "NONE", "NULL"}:
        return np.nan
    s = s.replace("_", "-").replace(" ", "-")
    if s in {"NONSOMA", "NON-SOMA"}:
        return "NON-SOMA"
    if s in {"NOISE", "MUA", "GOOD"}:
        return s
    return s


def _parse_bc_label_filter(bc_label_filter):
    if bc_label_filter is None:
        return None

    if isinstance(bc_label_filter, str):
        s = bc_label_filter.strip()
        if s == "" or s.lower() in {"all", "any", "both", "none", "*"}:
            return None
        parts = [p for p in re.split(r"[,\|]", s) if str(p).strip() != ""]
    elif isinstance(bc_label_filter, Iterable):
        parts = list(bc_label_filter)
    else:
        parts = [bc_label_filter]

    keep = []
    for p in parts:
        if p is None:
            continue
        ps = str(p).strip()
        if ps == "":
            continue
        if ps.lower() in {"all", "any", "both", "none", "*"}:
            return None
        n = _pca_norm_bc_label(p)
        if pd.isna(n):
            continue
        keep.append(n)

    if len(keep) == 0:
        return None

    keep_set = sorted(set(keep))
    invalid = [v for v in keep_set if v not in _BC_LABEL_ALLOWED]
    if invalid:
        raise ValueError(
            "Invalid bc_label_filter values="
            f"{invalid}. Allowed values: {sorted(_BC_LABEL_ALLOWED)} or use 'all'."
        )
    return keep_set


def pca_get_probe_units_df(merged_dic, probe, roi_filter=None, kslabel_filter="both", bc_label_filter=None):
    if probe not in merged_dic:
        raise ValueError(f"Probe {probe} not in merged_dic keys: {list(merged_dic.keys())}")

    df = merged_dic[probe].copy().reset_index(drop=True)
    if "spike_times" not in df.columns:
        raise ValueError(f"Probe {probe} DataFrame missing spike_times column.")

    if "probe" in df.columns:
        probe_norm = df["probe"].astype(str).str.strip().str.upper().str[0]
        df = df[probe_norm == str(probe).strip().upper()].reset_index(drop=True)

    if roi_filter is not None:
        if "in_brainRegion" not in df.columns:
            raise ValueError("ROI filter requested but in_brainRegion column is missing.")
        df = df[df["in_brainRegion"].astype(str) == str(roi_filter)].reset_index(drop=True)

    ks_mode = "both" if kslabel_filter is None else str(kslabel_filter).strip().lower()
    if ks_mode not in {"both", "all", "none"}:
        ks_col = None
        for c in ["KSlabel", "KSLabel", "kslabel", "ks_label"]:
            if c in df.columns:
                ks_col = c
                break
        if ks_col is None:
            raise ValueError("KSLabel filter requested but no KSlabel/KSLabel column exists.")
        target = _pca_norm_kslabel(kslabel_filter)
        if target not in {"good", "mua"}:
            raise ValueError(f"Invalid kslabel_filter={kslabel_filter}. Use 'good', 'mua', or 'both'.")
        ks_norm = df[ks_col].map(_pca_norm_kslabel)
        df = df[ks_norm == target].reset_index(drop=True)

    bc_keep = _parse_bc_label_filter(bc_label_filter)
    if bc_keep is not None:
        if "bc_label" not in df.columns:
            raise ValueError("bc_label filter requested but bc_label column is missing.")
        bc_norm = df["bc_label"].map(_pca_norm_bc_label)
        df = df[bc_norm.isin(bc_keep)].reset_index(drop=True)

    valid = df["spike_times"].apply(lambda x: isinstance(x, (list, np.ndarray))).to_numpy()
    df = df[valid].reset_index(drop=True)
    return df


def pca_bin_spikes_around_events(
    spike_times_list,
    event_times_s,
    win_start_s,
    win_end_s,
    bin_size_s,
    max_tensor_gb=8.0,
):
    edges = np.arange(win_start_s, win_end_s + bin_size_s, bin_size_s)
    n_bins = len(edges) - 1
    n_trials = len(event_times_s)
    n_units = len(spike_times_list)

    est_gb = (n_trials * n_units * n_bins * np.dtype(np.float32).itemsize) / (1024**3)
    print(f"Requested tensor shape=({n_trials}, {n_units}, {n_bins}) est_mem={est_gb:.2f} GB")
    if est_gb > max_tensor_gb:
        raise MemoryError(
            f"Estimated tensor memory {est_gb:.2f} GB exceeds max_tensor_gb={max_tensor_gb}. "
            "Reduce events/units or increase bin_size_s."
        )

    X = np.zeros((n_trials, n_units, n_bins), dtype=np.float32)
    for u, st in enumerate(spike_times_list):
        st = np.asarray(st, dtype=float)
        if st.size == 0:
            continue
        for t, t0 in enumerate(event_times_s):
            i0 = np.searchsorted(st, t0 + win_start_s, side="left")
            i1 = np.searchsorted(st, t0 + win_end_s, side="right")
            rel = st[i0:i1] - t0
            if rel.size:
                counts, _ = np.histogram(rel, bins=edges)
                X[t, u, :] = counts / bin_size_s
    return X, edges[:-1]


def pca_trial_level(trials, labels, n_components=12):
    X_trial = trials.mean(axis=2).T
    Xz = zscore_rows(X_trial)
    pca = PCA(n_components=min(n_components, Xz.shape[0], Xz.shape[1]))
    Xp = pca.fit_transform(Xz.T).T
    trial_types = pd.unique(labels)
    t_type_ind = [np.where(labels == t)[0] for t in trial_types]
    return Xp, pca.explained_variance_ratio_, trial_types, t_type_ind


def pca_trajectory(trials, labels, n_components=12):
    trial_types = pd.unique(labels)
    t_type_ind = [np.where(labels == t)[0] for t in trial_types]
    trial_averages = []
    kept_labels = []
    for t, idx in zip(trial_types, t_type_ind):
        if len(idx) > 0:
            trial_averages.append(trials[idx].mean(axis=0))
            kept_labels.append(t)
    if len(trial_averages) < 2:
        raise ValueError("Need at least 2 non-empty event labels for trajectory PCA.")
    Xa = np.hstack(trial_averages)
    Xaz = zscore_rows(Xa)
    pca = PCA(n_components=min(n_components, Xaz.shape[0], Xaz.shape[1]))
    Xa_p = pca.fit_transform(Xaz.T).T
    return Xa_p, pca.explained_variance_ratio_, kept_labels


def pca_plot_trial_level(
    Xp,
    trial_types,
    t_type_ind,
    title_prefix="",
    save_root: str | Path = "master/results",
    show_plots=True,
):
    projections = [(0, 1), (1, 2), (0, 2)]
    pal = sns.color_palette("colorblind", len(trial_types))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (i, j) in zip(axes, projections):
        for k, t in enumerate(trial_types):
            idx = t_type_ind[k]
            ax.scatter(Xp[i, idx], Xp[j, idx], s=28, alpha=0.8, color=pal[k], label=str(t))
        ax.set_xlabel(f"PC {i+1}")
        ax.set_ylabel(f"PC {j+1}")
    axes[0].set_title(f"{title_prefix} Trial PCA")
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine()
    plt.tight_layout()

    save_path = _ensure_dir(Path(save_root) / "trial_level")
    out = save_path / f"{_sanitize_name(title_prefix)}.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")


def pca_plot_trajectory(
    Xa_p,
    kept_labels,
    n_bins,
    time,
    smooth_sigma=2,
    title_prefix="",
    save_root: str | Path = "master/results",
    show_plots=True,
):
    pal = sns.color_palette("colorblind", len(kept_labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for comp in range(min(3, Xa_p.shape[0])):
        ax = axes[comp]
        for k, lbl in enumerate(kept_labels):
            s = k * n_bins
            e = (k + 1) * n_bins
            x = Xa_p[comp, s:e]
            if smooth_sigma and smooth_sigma > 0:
                x = gaussian_filter1d(x, sigma=smooth_sigma)
            ax.plot(time, x, lw=2, color=pal[k], label=str(lbl))
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.set_ylabel(f"PC {comp+1}")
    axes[1].set_xlabel("Time from event (s)")
    axes[0].set_title(f"{title_prefix} Trajectory PCA")
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine()
    plt.tight_layout()

    save_path = _ensure_dir(Path(save_root) / "trajectory")
    out = save_path / f"{_sanitize_name(title_prefix)}_trajectory.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")


def pca_single_stim_trajectory(trials, n_components=12):
    Xa = trials.mean(axis=0)
    Xaz = zscore_rows(Xa)
    pca = PCA(n_components=min(n_components, Xaz.shape[0], Xaz.shape[1]))
    Xa_p = pca.fit_transform(Xaz.T).T
    return Xa_p, pca.explained_variance_ratio_


def get_stim_events(events_df, stim_label_col, stim_name, time_col, max_events_per_stim=None):
    stim_target = str(stim_name).strip().lower()
    stim_vals = events_df[stim_label_col].astype(str).str.strip().str.lower()
    sdf = events_df[stim_vals == stim_target].copy().reset_index(drop=True)
    if max_events_per_stim is not None and len(sdf) > max_events_per_stim:
        idx = np.linspace(0, len(sdf) - 1, max_events_per_stim, dtype=int)
        sdf = sdf.iloc[idx].reset_index(drop=True)
    times = pd.to_numeric(sdf[time_col], errors="coerce").dropna().to_numpy(dtype=float)
    return sdf, times

def plot_probe_stimulus_panel(
    probe,
    merged_dic,
    events_df,
    stim_label_col,
    time_col,
    roi_filter=None,
    kslabel_filter="both",
    n_components=12,
    smooth_sigma=2,
    max_tensor_gb=8.0,
    max_events_per_stim=None,
    ncols=3,
    one_stim_per_fig=True,
    panel_per_probe=True,
    win_start_s=-1.0,
    win_end_s=1.0,
    bin_size_s=0.025,
    save_root: str | Path = "master/results",
    show_plots=True,
):
    probe_df = pca_get_probe_units_df(
        merged_dic=merged_dic,
        probe=probe,
        roi_filter=roi_filter,
        kslabel_filter=kslabel_filter,
    )
    if probe_df.empty:
        print(f"Probe {probe}: no units after ROI/KS filters.")
        return None

    spike_times_list = [np.asarray(x, dtype=float) for x in probe_df["spike_times"].values]
    stim_names = sorted(events_df[stim_label_col].astype(str).unique().tolist())
    per_stim = []

    for stim_name in stim_names:
        _, stim_times = get_stim_events(events_df, stim_label_col, stim_name, time_col, max_events_per_stim=max_events_per_stim)
        if len(stim_times) < 2:
            continue
        try:
            trials_stim, time = pca_bin_spikes_around_events(
                spike_times_list=spike_times_list,
                event_times_s=stim_times,
                win_start_s=win_start_s,
                win_end_s=win_end_s,
                bin_size_s=bin_size_s,
                max_tensor_gb=max_tensor_gb,
            )
        except MemoryError as e:
            print(f"Probe {probe} | stim {stim_name}: skipped due to memory guard: {e}")
            continue
        Xa_p, evr = pca_single_stim_trajectory(trials_stim, n_components=n_components)
        per_stim.append({"stim_name": stim_name, "n_events": len(stim_times), "time": time, "traj": Xa_p, "evr": evr})

    if len(per_stim) == 0:
        print(f"Probe {probe}: no stimulus groups to plot.")
        return None

    base = Path(save_root)
    if panel_per_probe:
        n = len(per_stim)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), sharex=True)
        axes = np.array(axes).reshape(-1)
        for ax in axes[n:]:
            ax.axis("off")
        for i, rec in enumerate(per_stim):
            ax = axes[i]
            t = rec["time"]
            x1 = rec["traj"][0]
            x2 = rec["traj"][1] if rec["traj"].shape[0] > 1 else None
            x3 = rec["traj"][2] if rec["traj"].shape[0] > 2 else None
            if smooth_sigma and smooth_sigma > 0:
                x1 = gaussian_filter1d(x1, sigma=smooth_sigma)
                if x2 is not None:
                    x2 = gaussian_filter1d(x2, sigma=smooth_sigma)
                if x3 is not None:
                    x3 = gaussian_filter1d(x3, sigma=smooth_sigma)
            ax.plot(t, x1, lw=2, label="PC1")
            if x2 is not None:
                ax.plot(t, x2, lw=1.5, label="PC2")
            if x3 is not None:
                ax.plot(t, x3, lw=1.2, label="PC3")
            ax.axvline(0, color="gray", ls="--", lw=1)
            ax.set_title(f"{rec['stim_name']} (n={rec['n_events']})")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("PC value")
        axes[0].legend(frameon=False)
        plt.suptitle(f"Probe {probe} | ROI={roi_filter} | KS={kslabel_filter}", y=1.01)
        plt.tight_layout()
        out = _ensure_dir(base / "panel_per_probe") / (
            f"probe_{_sanitize_name(probe)}_ROI_{_sanitize_name(roi_filter)}_KS_{_sanitize_name(kslabel_filter)}.png"
        )
        plt.savefig(out, dpi=250, bbox_inches="tight")
        if show_plots:
            plt.show()
        else:
            plt.close()
            print(f"PCA plot saved to {out}")

    if one_stim_per_fig:
        for rec in per_stim:
            fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
            t = rec["time"]
            x1 = rec["traj"][0]
            x2 = rec["traj"][1] if rec["traj"].shape[0] > 1 else None
            x3 = rec["traj"][2] if rec["traj"].shape[0] > 2 else None
            if smooth_sigma and smooth_sigma > 0:
                x1 = gaussian_filter1d(x1, sigma=smooth_sigma)
                if x2 is not None:
                    x2 = gaussian_filter1d(x2, sigma=smooth_sigma)
                if x3 is not None:
                    x3 = gaussian_filter1d(x3, sigma=smooth_sigma)
            ax.plot(t, x1, lw=2, label="PC1")
            if x2 is not None:
                ax.plot(t, x2, lw=1.5, label="PC2")
            if x3 is not None:
                ax.plot(t, x3, lw=1.2, label="PC3")
            ax.axvline(0, color="gray", ls="--", lw=1)
            ax.set_title(f"Probe {probe} | {rec['stim_name']} (n={rec['n_events']})")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("PC value")
            ax.legend(frameon=False)
            sns.despine()
            plt.tight_layout()
            out = _ensure_dir(base / "one_stim_per_fig" / _sanitize_name(rec["stim_name"])) / (
                f"probe_{_sanitize_name(probe)}_stim_{_sanitize_name(rec['stim_name'])}_ROI_{_sanitize_name(roi_filter)}_KS_{_sanitize_name(kslabel_filter)}.png"
            )
            plt.savefig(out, dpi=250, bbox_inches="tight")
            if show_plots:
                plt.show()
            else:
                plt.close()
                print(f"PCA plot saved to {out}")
    return per_stim


def plot_probe_selected_stimuli_overlay(
    probe,
    merged_dic,
    events_df,
    selected_stimuli: Iterable[str],
    stim_label_col,
    time_col,
    roi_filter=None,
    kslabel_filter="both",
    n_components=12,
    smooth_sigma=2,
    max_tensor_gb=8.0,
    max_events_per_stim=None,
    win_start_s=-1.0,
    win_end_s=1.0,
    bin_size_s=0.025,
    save_root: str | Path = "master/results",
    show_plots=True,
):
    probe_df = pca_get_probe_units_df(merged_dic, probe, roi_filter=roi_filter, kslabel_filter=kslabel_filter)
    if probe_df.empty:
        print(f"Probe {probe}: no units after ROI/KS filters.")
        return None

    spike_times_list = [np.asarray(x, dtype=float) for x in probe_df["spike_times"].values]
    recs = []
    for stim_name in selected_stimuli:
        if stim_name not in events_df[stim_label_col].astype(str).unique():
            continue
        _, stim_times = get_stim_events(
            events_df=events_df,
            stim_label_col=stim_label_col,
            stim_name=stim_name,
            time_col=time_col,
            max_events_per_stim=max_events_per_stim,
        )
        if len(stim_times) < 2:
            continue
        try:
            trials_stim, t = pca_bin_spikes_around_events(
                spike_times_list=spike_times_list,
                event_times_s=stim_times,
                win_start_s=win_start_s,
                win_end_s=win_end_s,
                bin_size_s=bin_size_s,
                max_tensor_gb=max_tensor_gb,
            )
        except MemoryError as e:
            print(f"Probe {probe} | {stim_name}: skipped due to memory guard: {e}")
            continue
        traj, evr = pca_single_stim_trajectory(trials_stim, n_components=n_components)
        recs.append({"stim": stim_name, "n": len(stim_times), "time": t, "traj": traj, "evr": evr})

    if len(recs) == 0:
        print(f"Probe {probe}: no selected stimuli were plottable.")
        return None

    pal = sns.color_palette("colorblind", len(recs))
    fig, axes = plt.subplots(1, 3, figsize=(20, 3.8), sharex=True)
    for k, rec in enumerate(recs):
        x1 = rec["traj"][0]
        x2 = rec["traj"][1] if rec["traj"].shape[0] > 1 else None
        x3 = rec["traj"][2] if rec["traj"].shape[0] > 2 else None
        tt = rec["time"]
        if smooth_sigma and smooth_sigma > 0:
            x1 = gaussian_filter1d(x1, sigma=smooth_sigma)
            if x2 is not None:
                x2 = gaussian_filter1d(x2, sigma=smooth_sigma)
            if x3 is not None:
                x3 = gaussian_filter1d(x3, sigma=smooth_sigma)
        axes[0].plot(tt, x1, lw=2, color=pal[k], label=f"{rec['stim']} (n={rec['n']})")
        if x2 is not None:
            axes[1].plot(tt, x2, lw=2, color=pal[k], label=f"{rec['stim']} (n={rec['n']})")
        if x3 is not None:
            axes[2].plot(tt, x3, lw=2, color=pal[k], label=f"{rec['stim']} (n={rec['n']})")
    for c, ax in enumerate(axes, start=1):
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"PC{c}")
    axes[0].set_xlim(win_start_s, win_end_s)
    plt.subplots_adjust(left=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.02, 0.5), frameon=True)
    plt.suptitle(f"Probe {probe} | ROI={roi_filter} | KS={kslabel_filter} | Selected stimuli overlay={selected_stimuli}")
    sns.despine()
    plt.tight_layout(rect=(0.275, 0, 1, 0.95))

    stim_tag = "__".join([_sanitize_name(s) for s in selected_stimuli])
    out = _ensure_dir(Path(save_root) / "overlay" / stim_tag) / (
        f"probe_{_sanitize_name(probe)}_ROI_{_sanitize_name(roi_filter)}_KS_{_sanitize_name(kslabel_filter)}.png"
    )
    plt.savefig(out, dpi=250, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")
    return recs


def run_epoch_condition_pca_for_probe(
    probe,
    merged_dic,
    event_meta,
    roi_filter=None,
    kslabel_filter="both",
    bc_label_filter=None,
    include_conditions=None,
    n_components=12,
    max_tensor_gb=8.0,
    win_start_s=-1.0,
    win_end_s=1.0,
    bin_size_s=0.025,
):
    if include_conditions is None:
        include_conditions = ["baseline", "stimulation", "washout"]
    em = event_meta.copy()
    em = em[em["condition"].isin(include_conditions)].reset_index(drop=True)
    if em.empty:
        raise ValueError("No events left after include_conditions filter.")

    probe_df = pca_get_probe_units_df(
        merged_dic=merged_dic,
        probe=probe,
        roi_filter=roi_filter,
        kslabel_filter=kslabel_filter,
        bc_label_filter=bc_label_filter,
    )
    if probe_df.empty:
        raise ValueError(f"Probe {probe}: no units after filtering.")

    spike_times = [np.asarray(x, dtype=float) for x in probe_df["spike_times"].values]
    trials, t = pca_bin_spikes_around_events(
        spike_times_list=spike_times,
        event_times_s=em["start_time"].to_numpy(dtype=float),
        win_start_s=win_start_s,
        win_end_s=win_end_s,
        bin_size_s=bin_size_s,
        max_tensor_gb=max_tensor_gb,
    )
    X_trial = trials.mean(axis=2).T
    Xz = zscore_rows(X_trial)
    pca = PCA(n_components=min(n_components, Xz.shape[0], Xz.shape[1]))
    Xp = pca.fit_transform(Xz.T).T
    return em, trials, t, Xp, pca.explained_variance_ratio_


def _probe_brain_region_label(merged_dic, probe, roi_filter=None, kslabel_filter="both", max_regions=6):
    probe_df = pca_get_probe_units_df(merged_dic=merged_dic, probe=probe, roi_filter=roi_filter, kslabel_filter=kslabel_filter)
    if probe_df.empty or ("brain_region" not in probe_df.columns):
        return "brain_region: n/a"
    vals = probe_df["brain_region"].dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    uniq = sorted(vals.unique().tolist())
    if len(uniq) == 0:
        return "brain_region: n/a"
    if len(uniq) <= max_regions:
        return "brain_region: " + ", ".join(uniq)
    head = ", ".join(uniq[:max_regions])
    return f"brain_region: {head}, +{len(uniq) - max_regions} more"

def plot_epoch_condition_scatter(Xp, event_meta, title_prefix="", subtitle="", probe="unknown", save_root: str | Path = "master/results", show_plots=True):
    if Xp.shape[0] < 3:
        raise ValueError(f"Need at least 3 PCs for plotting; got {Xp.shape[0]}.")
    projections = [(0, 1), (1, 2), (0, 2)]
    style_ctx, cond_marker, _ = _epoch_condition_color_marker_maps(event_meta)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, (i, j) in zip(axes, projections):
        for cond in event_meta["condition"].unique():
            em_cond = event_meta[event_meta["condition"] == cond]
            marker = cond_marker.get(cond, "o")
            for ep in sorted(em_cond["epoch_id"].dropna().astype(int).unique()):
                idx = em_cond.index[em_cond["epoch_id"].astype(int) == ep].to_numpy()
                ax.scatter(
                    Xp[i, idx], Xp[j, idx],
                    s=140, alpha=1.0, marker=marker,
                    color=_epoch_condition_color_for(style_ctx, cond, ep), edgecolors="black", linewidths=0.8,
                    label=f"{cond}, epoch {ep}",
                )
        ax.set_xlabel(f"PC {i+1}")
        ax.set_ylabel(f"PC {j+1}")

    handles, labels = axes[-1].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[-1].legend(uniq.values(), uniq.keys(), frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=10)
    fig.suptitle(title_prefix, fontsize=18, y=0.99)
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=11)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    out = _ensure_dir(Path(save_root) /  "2D_scatter") / f"probe_{_sanitize_name(probe)}_epoch_condition_scatter_2D.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")


def plot_epoch_condition_scatter_3d(Xp, event_meta, title_prefix="", subtitle="", probe="unknown", save_root: str | Path = "master/results", show_plots=False):
    if Xp.shape[0] < 3:
        raise ValueError(f"Need at least 3 PCs for 3D plotting; got {Xp.shape[0]}.")
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    style_ctx, cond_marker, _ = _epoch_condition_color_marker_maps(event_meta)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    for cond in event_meta["condition"].unique():
        em_cond = event_meta[event_meta["condition"] == cond]
        marker = cond_marker.get(cond, "o")
        for ep in sorted(em_cond["epoch_id"].dropna().astype(int).unique()):
            idx = em_cond.index[em_cond["epoch_id"].astype(int) == ep].to_numpy()
            ax.scatter(
                Xp[0, idx], Xp[1, idx], Xp[2, idx],
                s=90, alpha=1.0, marker=marker,
                color=_epoch_condition_color_for(style_ctx, cond, ep), edgecolors="black", linewidths=0.6,
                label=f"{cond}, epoch {ep}",
            )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.suptitle(title_prefix + " (3D)", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.85, 0.9])
    out = _ensure_dir(Path(save_root) / "3D_scatter") / f"probe_{_sanitize_name(probe)}_epoch_condition_scatter_3D.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")


def plot_epoch_condition_line_3d_time(
    Xp,
    event_meta,
    title_prefix="",
    subtitle="",
    probe="unknown",
    time_col="start_time",
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if Xp.shape[0] < 2:
        raise ValueError(f"Need at least 2 PCs for PC1-PC2-Time plotting; got {Xp.shape[0]}.")
    if time_col not in event_meta.columns:
        raise ValueError(f"{time_col} not found in event_meta columns: {list(event_meta.columns)}")
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    em = event_meta.copy()
    em[time_col] = pd.to_numeric(em[time_col], errors="coerce")
    em = em.dropna(subset=[time_col, "epoch_id", "condition"]).reset_index(drop=True)
    if em.empty:
        raise ValueError("No valid events available after dropping NaN times/labels.")
    t0 = em[time_col].min()
    em["time_rel_s"] = em[time_col] - t0

    style_ctx, cond_marker, cond_linestyle = _epoch_condition_color_marker_maps(em)

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    for cond in em["condition"].astype(str).unique():
        em_cond = em[em["condition"].astype(str) == cond]
        marker = cond_marker.get(cond, "o")
        ls = _epoch_condition_linestyle_for(cond_linestyle, cond)
        for ep in sorted(em_cond["epoch_id"].astype(int).unique()):
            idx = em_cond.index[em_cond["epoch_id"].astype(int) == ep].to_numpy()
            if idx.size < 2:
                continue
            order = np.argsort(em.loc[idx, "time_rel_s"].to_numpy(dtype=float))
            idx = idx[order]
            ax.plot(
                Xp[0, idx], Xp[1, idx], em.loc[idx, "time_rel_s"].to_numpy(dtype=float),
                color=_epoch_condition_color_for(style_ctx, cond, ep), linestyle=ls, linewidth=2.2,
                marker=marker, markersize=4.0, markeredgecolor="black", markeredgewidth=0.5,
                label=f"{cond}, epoch {ep}",
            )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("Time (s, rel)")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.suptitle(title_prefix + " (PC1-PC2-Time)", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.85, 0.9])
    out = _ensure_dir(Path(save_root)  / "3D_PC12_Time") / f"probe_{_sanitize_name(probe)}_epoch_condition_line_PC12_Time.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")


def _epoch_condition_group_order(event_meta):
    cond_order = ["baseline", "stimulation", "washout"]
    em = event_meta.copy()
    em = em.dropna(subset=["condition", "epoch_id"]).reset_index(drop=True)
    groups = []
    present_conds = em["condition"].astype(str).unique().tolist()
    ordered_conds = [c for c in cond_order if c in present_conds] + [c for c in present_conds if c not in cond_order]
    for cond in ordered_conds:
        em_c = em[em["condition"].astype(str) == cond]
        for ep in sorted(em_c["epoch_id"].astype(int).unique().tolist()):
            idx = em_c.index[em_c["epoch_id"].astype(int) == ep].to_numpy()
            if idx.size > 0:
                groups.append((cond, int(ep), idx))
    return groups


def plot_epoch_condition_scatter_epoch_avg(Xp, event_meta, title_prefix="", subtitle="", probe="unknown", save_root: str | Path = "master/results", show_plots=False):
    if Xp.shape[0] < 3:
        raise ValueError(f"Need at least 3 PCs for plotting; got {Xp.shape[0]}.")
    groups = _epoch_condition_group_order(event_meta)
    if len(groups) == 0:
        raise ValueError("No valid (condition, epoch) groups found for epoch-averaged plotting.")
    style_ctx, cond_marker, _ = _epoch_condition_color_marker_maps(event_meta)

    projections = [(0, 1), (1, 2), (0, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, (i, j) in zip(axes, projections):
        for cond, ep, idx in groups:
            p1 = float(np.nanmean(Xp[i, idx]))
            p2 = float(np.nanmean(Xp[j, idx]))
            ax.scatter(
                [p1], [p2], s=220, alpha=1.0,
                marker=cond_marker.get(cond, "o"), color=_epoch_condition_color_for(style_ctx, cond, ep, default_color="gray"),
                edgecolors="black", linewidths=1.0, label=f"{cond}, epoch {ep}",
            )
        ax.set_xlabel(f"PC {i+1}")
        ax.set_ylabel(f"PC {j+1}")
    handles, labels = axes[-1].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[-1].legend(uniq.values(), uniq.keys(), frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=10)
    fig.suptitle(title_prefix + " (Epoch Avg)", fontsize=18, y=0.99)
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=11)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    out = _ensure_dir(Path(save_root) / "2D_scatter_epoch_avg") / f"probe_{_sanitize_name(probe)}_epoch_avg_scatter_2D.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def plot_epoch_condition_scatter_3d_epoch_avg(Xp, event_meta, title_prefix="", subtitle="", probe="unknown", save_root: str | Path = "master/results", show_plots=False):
    if Xp.shape[0] < 3:
        raise ValueError(f"Need at least 3 PCs for 3D plotting; got {Xp.shape[0]}.")
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    groups = _epoch_condition_group_order(event_meta)
    if len(groups) == 0:
        raise ValueError("No valid (condition, epoch) groups found for epoch-averaged plotting.")
    style_ctx, cond_marker, _ = _epoch_condition_color_marker_maps(event_meta)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    for cond, ep, idx in groups:
        x = float(np.nanmean(Xp[0, idx]))
        y = float(np.nanmean(Xp[1, idx]))
        z = float(np.nanmean(Xp[2, idx]))
        ax.scatter(
            [x], [y], [z], s=160, alpha=1.0,
            marker=cond_marker.get(cond, "o"), color=_epoch_condition_color_for(style_ctx, cond, ep, default_color="gray"),
            edgecolors="black", linewidths=0.8, label=f"{cond}, epoch {ep}",
        )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.suptitle(title_prefix + " (3D, Epoch Avg)", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.85, 0.9])
    out = _ensure_dir(Path(save_root) / "3D_scatter_epoch_avg") / f"probe_{_sanitize_name(probe)}_epoch_avg_scatter_3D.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")


def plot_epoch_condition_line_3d_time_epoch_avg(
    trials,
    event_meta,
    time_bins,
    title_prefix="",
    subtitle="",
    probe="unknown",
    smooth_sigma=2.0,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if trials.ndim != 3:
        raise ValueError(f"trials must be (n_events, n_units, n_bins); got shape {trials.shape}")
    if len(time_bins) != trials.shape[2]:
        raise ValueError(f"time_bins length {len(time_bins)} does not match trials bins {trials.shape[2]}")
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    groups = _epoch_condition_group_order(event_meta)
    if len(groups) == 0:
        raise ValueError("No valid (condition, epoch) groups found for epoch-averaged line plotting.")

    trajs = []
    meta = []
    for cond, ep, idx in groups:
        tr = np.nanmean(trials[idx, :, :], axis=0)
        trajs.append(tr)
        meta.append((cond, ep))

    X = np.concatenate([tr.T for tr in trajs], axis=0)
    Xz = zscore_rows(X.T).T
    pca = PCA(n_components=min(3, Xz.shape[0], Xz.shape[1]))
    Xp = pca.fit_transform(Xz)

    n_bins = len(time_bins)
    pcs_by_group = []
    start = 0
    for _ in trajs:
        pcs_by_group.append(Xp[start:start + n_bins, :])
        start += n_bins

    style_ctx, cond_marker, cond_linestyle = _epoch_condition_color_marker_maps(event_meta)

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    t = np.asarray(time_bins, dtype=float)

    for (cond, ep), pcs in zip(meta, pcs_by_group):
        x = pcs[:, 0].copy()
        y = pcs[:, 1].copy()
        if smooth_sigma and smooth_sigma > 0:
            x = gaussian_filter1d(x, sigma=float(smooth_sigma))
            y = gaussian_filter1d(y, sigma=float(smooth_sigma))
        ax.plot(
            x, y, t,
            color=_epoch_condition_color_for(style_ctx, cond, ep, default_color="gray"),
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
            linewidth=2.6,
            marker=cond_marker.get(cond, "o"),
            markersize=4.0,
            markevery=[0, len(t)-1],
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=f"{cond}, epoch {ep}",
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("Time (s, peri-event)")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.suptitle(title_prefix + " (PC1-PC2-Time, Epoch Avg)", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.85, 0.9])
    out = _ensure_dir(Path(save_root) / "3D_PC12_Time_epoch_avg") / f"probe_{_sanitize_name(probe)}_epoch_avg_line_PC12_Time.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")



# === V2 overrides from all_in_one notebook ===

def run_epoch_condition_pca_for_probe(
    probe,
    merged_dic,
    event_meta,
    roi_filter=None,
    kslabel_filter="both",
    bc_label_filter=None,
    include_conditions=None,
    n_components=12,
    max_tensor_gb=8.0,
    win_start_s=-1.0,
    win_end_s=1.0,
    bin_size_s=0.025,
    brain_region_filter=None,
):
    if include_conditions is None:
        include_conditions = ["baseline", "stimulation", "washout"]

    em = event_meta.copy()
    em = em[em["condition"].isin(include_conditions)].reset_index(drop=True)
    if em.empty:
        raise ValueError("No events left after include_conditions filter.")

    probe_df = pca_get_probe_units_df(
        merged_dic=merged_dic,
        probe=probe,
        roi_filter=roi_filter,
        kslabel_filter=kslabel_filter,
        bc_label_filter=bc_label_filter,
        brain_region_filter=brain_region_filter,
    )
    if probe_df.empty:
        raise ValueError(f"Probe {probe}: no units after filtering.")

    spike_times = [np.asarray(x, dtype=float) for x in probe_df["spike_times"].values]

    trials, t = pca_bin_spikes_around_events(
        spike_times_list=spike_times,
        event_times_s=em["start_time"].to_numpy(dtype=float),
        win_start_s=win_start_s,
        win_end_s=win_end_s,
        bin_size_s=bin_size_s,
        max_tensor_gb=max_tensor_gb,
    )

    X_trial = trials.mean(axis=2).T
    Xz = zscore_rows(X_trial)
    pca = PCA(n_components=min(n_components, Xz.shape[0], Xz.shape[1]))
    Xp = pca.fit_transform(Xz.T).T

    return em, trials, t, Xp, pca.explained_variance_ratio_

def _pca_norm_brain_region(v):
    if pd.isna(v):
        return "unknown"
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return "unknown"
    return s

def pca_get_probe_units_df(merged_dic, probe, roi_filter=None, kslabel_filter="both", brain_region_filter=None, bc_label_filter=None):
    if probe not in merged_dic:
        raise ValueError(f"Probe {probe} not in merged_dic keys: {list(merged_dic.keys())}")

    df = merged_dic[probe].copy().reset_index(drop=True)
    if "spike_times" not in df.columns:
        raise ValueError(f"Probe {probe} DataFrame missing spike_times column.")

    if "probe" in df.columns:
        probe_norm = df["probe"].astype(str).str.strip().str.upper().str[0]
        df = df[probe_norm == str(probe).strip().upper()].reset_index(drop=True)

    if roi_filter is not None:
        if "in_brainRegion" not in df.columns:
            raise ValueError("ROI filter requested but in_brainRegion column is missing.")
        df = df[df["in_brainRegion"].astype(str) == str(roi_filter)].reset_index(drop=True)

    ks_mode = "both" if kslabel_filter is None else str(kslabel_filter).strip().lower()
    if ks_mode not in {"both", "all", "none"}:
        ks_col = None
        for c in ["KSlabel", "KSLabel", "kslabel", "ks_label"]:
            if c in df.columns:
                ks_col = c
                break
        if ks_col is None:
            raise ValueError("KSLabel filter requested but no KSlabel/KSLabel column exists in probe dataframe.")

        target = _pca_norm_kslabel(kslabel_filter)
        if target not in {"good", "mua"}:
            raise ValueError(f"Invalid kslabel_filter={kslabel_filter}. Use 'good', 'mua', or 'both'.")

        ks_norm = df[ks_col].map(_pca_norm_kslabel)
        df = df[ks_norm == target].reset_index(drop=True)

    bc_keep = _parse_bc_label_filter(bc_label_filter)
    if bc_keep is not None:
        if "bc_label" not in df.columns:
            raise ValueError("bc_label filter requested but bc_label column is missing.")
        bc_norm = df["bc_label"].map(_pca_norm_bc_label)
        df = df[bc_norm.isin(bc_keep)].reset_index(drop=True)

    if "brain_region" in df.columns:
        df = df.assign(brain_region=df["brain_region"].map(_pca_norm_brain_region))

    if brain_region_filter is not None:
        if "brain_region" not in df.columns:
            raise ValueError("brain_region_filter requested but brain_region column is missing.")
        target = _pca_norm_brain_region(brain_region_filter)
        df = df[df["brain_region"] == target].reset_index(drop=True)

    valid = df["spike_times"].apply(lambda x: isinstance(x, (list, np.ndarray))).to_numpy()
    df = df[valid].reset_index(drop=True)
    return df

def pca_single_stim_trajectory(trials, n_components=12):
    """
    trials: (n_trials, n_units, n_bins) for one stimulus only
    Returns component trajectories over time for that stimulus.
    """
    Xa = trials.mean(axis=0)  # (n_units, n_bins)
    Xaz = zscore_rows(Xa)
    pca = PCA(n_components=min(n_components, Xaz.shape[0], Xaz.shape[1]))
    Xa_p = pca.fit_transform(Xaz.T).T  # (n_components, n_bins)
    return Xa_p, pca.explained_variance_ratio_

def get_stim_events(events_df, stim_label_col, stim_name, time_col, max_events_per_stim=None):
    stim_target = str(stim_name).strip().lower()
    stim_vals = events_df[stim_label_col].astype(str).str.strip().str.lower()
    sdf = events_df[stim_vals == stim_target].copy().reset_index(drop=True)
    if max_events_per_stim is not None and len(sdf) > max_events_per_stim:
        idx = np.linspace(0, len(sdf) - 1, max_events_per_stim, dtype=int)
        sdf = sdf.iloc[idx].reset_index(drop=True)
    times = pd.to_numeric(sdf[time_col], errors="coerce").dropna().to_numpy(dtype=float)
    return sdf, times

def plot_probe_stimulus_panel(probe,
                              merged_dic,
                              events_df,
                              stim_label_col,
                              time_col,
                              roi_filter=None,
                              kslabel_filter="both",
                              brain_region_filter=None,
                              n_components=12,
                              smooth_sigma=2,
                              max_tensor_gb=8.0,
                              max_events_per_stim=None,
                              ncols=3,
                              save_dir=None,
                              one_stim_per_fig=True,
                              panel_per_probe=True,
                              save_root: str | Path = "master/results",
                              show_plots=True):
    probe_df = pca_get_probe_units_df(
        merged_dic=merged_dic,
        probe=probe,
        roi_filter=roi_filter,
        kslabel_filter=kslabel_filter,
        brain_region_filter=brain_region_filter,
    )
    if probe_df.empty:
        print(f"Probe {probe} | brain_region={brain_region_filter}: no units after ROI/KS/brain_region filters.")
        return None

    n_units = int(len(probe_df))
    region_label = _pca_norm_brain_region(brain_region_filter) if brain_region_filter is not None else "all_regions"

    spike_times_list = [np.asarray(x, dtype=float) for x in probe_df["spike_times"].values]
    stim_names = sorted(events_df[stim_label_col].astype(str).str.strip().unique().tolist())

    per_stim = []
    for stim_name in stim_names:
        sdf, stim_times = get_stim_events(
            events_df=events_df,
            stim_label_col=stim_label_col,
            stim_name=stim_name,
            time_col=time_col,
            max_events_per_stim=max_events_per_stim,
        )
        if len(stim_times) < 2:
            continue

        try:
            trials_stim, time = pca_bin_spikes_around_events(
                spike_times_list=spike_times_list,
                event_times_s=stim_times,
                win_start_s=WINDOW_START_S,
                win_end_s=WINDOW_END_S,
                bin_size_s=BIN_SIZE_S,
                max_tensor_gb=max_tensor_gb,
            )
        except MemoryError as e:
            print(f"Probe {probe} | brain_region={region_label} | stim {stim_name}: skipped due to memory guard: {e}")
            continue

        Xa_p, evr = pca_single_stim_trajectory(trials_stim, n_components=n_components)
        per_stim.append({
            "stim_name": stim_name,
            "n_events": len(stim_times),
            "time": time,
            "traj": Xa_p,
            "evr": evr,
            "brain_region": region_label,
            "n_units": n_units,
        })

    if len(per_stim) == 0:
        print(f"Probe {probe} | brain_region={region_label}: no stimulus groups to plot.")
        return None

    if panel_per_probe:
        n = len(per_stim)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.2*nrows), sharex=True)
        axes = np.array(axes).reshape(-1)

        for ax in axes[n:]:
            ax.axis('off')

        for i, rec in enumerate(per_stim):
            ax = axes[i]
            t = rec["time"]
            x1 = rec["traj"][0]
            x2 = rec["traj"][1] if rec["traj"].shape[0] > 1 else None
            x3 = rec["traj"][2] if rec["traj"].shape[0] > 2 else None
            if smooth_sigma and smooth_sigma > 0:
                x1 = gaussian_filter1d(x1, sigma=smooth_sigma)
                if x2 is not None: x2 = gaussian_filter1d(x2, sigma=smooth_sigma)
                if x3 is not None: x3 = gaussian_filter1d(x3, sigma=smooth_sigma)

            ax.plot(t, x1, lw=2, label='PC1')
            if x2 is not None: ax.plot(t, x2, lw=1.5, label='PC2')
            if x3 is not None: ax.plot(t, x3, lw=1.2, label='PC3')
            ax.axvline(0, color='gray', ls='--', lw=1)
            ax.set_title(f"{rec['stim_name']} (n={rec['n_events']})")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('PC value')

        axes[0].legend(frameon=False)
        plt.suptitle(
            f"Probe {probe} | brain_region={region_label} | ROI={roi_filter} | KS={kslabel_filter}",
            y=1.02,
        )
        fig.text(0.5, 0.96, f"units: {n_units}", ha="center", va="center", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        save_path = Path(save_root) / 'stimulus_panel__byProbe'
        save_path.mkdir(parents=True, exist_ok=True)
        out = save_path / (
            f"probe_{_sanitize_name(probe)}_{_sanitize_name(region_label)}"
            f"_stimulus_panel_ROI_{_sanitize_name(roi_filter)}_KS_{_sanitize_name(kslabel_filter)}.png"
        )
        plt.savefig(out, dpi=250, bbox_inches='tight')
        print("Saved panel:", out)
        if show_plots:
            plt.show()
        else:
            plt.close()
            print(f"PCA plot saved to {out}")

    if one_stim_per_fig:
        for rec in per_stim:
            fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
            t = rec["time"]
            x1 = rec["traj"][0]
            x2 = rec["traj"][1] if rec["traj"].shape[0] > 1 else None
            x3 = rec["traj"][2] if rec["traj"].shape[0] > 2 else None
            if smooth_sigma and smooth_sigma > 0:
                x1 = gaussian_filter1d(x1, sigma=smooth_sigma)
                if x2 is not None: x2 = gaussian_filter1d(x2, sigma=smooth_sigma)
                if x3 is not None: x3 = gaussian_filter1d(x3, sigma=smooth_sigma)

            ax.plot(t, x1, lw=2, label='PC1')
            if x2 is not None: ax.plot(t, x2, lw=1.5, label='PC2')
            if x3 is not None: ax.plot(t, x3, lw=1.2, label='PC3')
            ax.axvline(0, color='gray', ls='--', lw=1)
            ax.set_title(
                f"Probe {probe} | brain_region={region_label} | {rec['stim_name']} "
                f"(n={rec['n_events']}, units={n_units})"
            )
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('PC value')
            ax.legend(frameon=False)
            sns.despine()
            plt.tight_layout()
            subfolder = rec['stim_name']
            save_path = Path(save_root) / 'stimulus_panel__byStim' / _sanitize_name(subfolder)
            save_path.mkdir(parents=True, exist_ok=True)
            out = save_path / (
                f"probe_{_sanitize_name(probe)}_{_sanitize_name(region_label)}"
                f"_stim_{_sanitize_name(rec['stim_name'])}_ROI_{_sanitize_name(roi_filter)}"
                f"_KS_{_sanitize_name(kslabel_filter)}.png"
            )
            plt.savefig(out, dpi=250, bbox_inches='tight')
            if show_plots:
                plt.show()
            else:
                plt.close()
                print(f"PCA plot saved to {out}")
    return per_stim

def plot_probe_selected_stimuli_overlay(probe,
                                        merged_dic,
                                        events_df,
                                        selected_stimuli,
                                        stim_label_col,
                                        time_col,
                                        roi_filter=None,
                                        kslabel_filter="both",
                                        brain_region_filter=None,
                                        n_components=12,
                                        smooth_sigma=2,
                                        max_tensor_gb=8.0,
                                        max_events_per_stim=None,
                                        save_dir=None,
                                        save_root: str | Path = "master/results",
                                        show_plots=True):
    probe_df = pca_get_probe_units_df(
        merged_dic=merged_dic,
        probe=probe,
        roi_filter=roi_filter,
        kslabel_filter=kslabel_filter,
        brain_region_filter=brain_region_filter,
    )
    if probe_df.empty:
        print(f"Probe {probe} | brain_region={brain_region_filter}: no units after ROI/KS/brain_region filters.")
        return None

    n_units = int(len(probe_df))
    region_label = _pca_norm_brain_region(brain_region_filter) if brain_region_filter is not None else "all_regions"
    spike_times_list = [np.asarray(x, dtype=float) for x in probe_df["spike_times"].values]

    recs = []
    stim_values_norm = events_df[stim_label_col].astype(str).str.strip().str.lower().unique()
    for stim_name in selected_stimuli:
        if str(stim_name).strip().lower() not in stim_values_norm:
            print(f"Probe {probe} | region={region_label} | {stim_name}: skipped (not found in events)")
            continue
        sdf, stim_times = get_stim_events(
            events_df=events_df,
            stim_label_col=stim_label_col,
            stim_name=stim_name,
            time_col=time_col,
            max_events_per_stim=max_events_per_stim,
        )
        if len(stim_times) < 2:
            print(f"Probe {probe} | region={region_label} | {stim_name}: skipped (<2 events)")
            continue

        try:
            trials_stim, t = pca_bin_spikes_around_events(
                spike_times_list=spike_times_list,
                event_times_s=stim_times,
                win_start_s=WINDOW_START_S,
                win_end_s=WINDOW_END_S,
                bin_size_s=BIN_SIZE_S,
                max_tensor_gb=max_tensor_gb,
            )
        except MemoryError as e:
            print(f"Probe {probe} | region={region_label} | {stim_name}: skipped due to memory guard: {e}")
            continue

        traj, evr = pca_single_stim_trajectory(trials_stim, n_components=n_components)
        recs.append({
            "stim": stim_name,
            "n": len(stim_times),
            "time": t,
            "traj": traj,
            "evr": evr,
        })

    if len(recs) == 0:
        print(f"Probe {probe} | region={region_label}: no selected stimuli were plottable.")
        return None

    pal = sns.color_palette("colorblind", len(recs))
    fig, axes = plt.subplots(1, 3, figsize=(20, 3.8), sharex=True)

    for k, rec in enumerate(recs):
        x1 = rec["traj"][0]
        x2 = rec["traj"][1] if rec["traj"].shape[0] > 1 else None
        x3 = rec["traj"][2] if rec["traj"].shape[0] > 2 else None
        tt = rec["time"]

        if smooth_sigma and smooth_sigma > 0:
            x1 = gaussian_filter1d(x1, sigma=smooth_sigma)
            if x2 is not None: x2 = gaussian_filter1d(x2, sigma=smooth_sigma)
            if x3 is not None: x3 = gaussian_filter1d(x3, sigma=smooth_sigma)

        axes[0].plot(tt, x1, lw=2, color=pal[k], label=f"{rec['stim']} (n={rec['n']})")
        if x2 is not None:
            axes[1].plot(tt, x2, lw=2, color=pal[k], label=f"{rec['stim']} (n={rec['n']})")
        if x3 is not None:
            axes[2].plot(tt, x3, lw=2, color=pal[k], label=f"{rec['stim']} (n={rec['n']})")

    for c, ax in enumerate(axes, start=1):
        ax.axvline(0, color='gray', ls='--', lw=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'PC{c}')

    axes[0].set_xlim(WINDOW_START_S, WINDOW_END_S)

    plt.subplots_adjust(left=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.02, 0.5),
        frameon=True,
    )

    plt.suptitle(
        f"Probe {probe} | brain_region={region_label} | ROI={roi_filter} | KS={kslabel_filter} | "
        f"Selected stimuli overlay={selected_stimuli}",
        y=0.98,
    )
    fig.text(0.5, 0.93, f"units: {n_units}", ha="center", va="center", fontsize=10)
    sns.despine()

    plt.tight_layout(rect=(0.275, 0, 1, 0.9))

    stim_tag = "__".join([_sanitize_name(s) for s in selected_stimuli])
    save_path = Path(save_root) / 'stimulus_panel_overlay' / stim_tag
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / (
        f"probe_{_sanitize_name(probe)}_{_sanitize_name(region_label)}"
        f"_ROI_{_sanitize_name(roi_filter)}_KS_{_sanitize_name(kslabel_filter)}.png"
    )
    plt.savefig(out, dpi=250, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")
    return recs

def _normalize_brain_region_value(v):
    if pd.isna(v):
        return "unknown"
    s = str(v).strip()
    if s == "":
        return "unknown"
    if s.lower() in {"nan", "none", "null"}:
        return "unknown"
    return s

def _list_probe_brain_regions(merged_dic, probe, roi_filter=None, kslabel_filter="both", bc_label_filter=None):
    probe_df = pca_get_probe_units_df(
        merged_dic=merged_dic,
        probe=probe,
        roi_filter=roi_filter,
        kslabel_filter=kslabel_filter,
        bc_label_filter=bc_label_filter,
    )
    if probe_df.empty:
        return []

    if "brain_region" not in probe_df.columns:
        return ["unknown"]

    vals = probe_df["brain_region"].map(_normalize_brain_region_value)
    return sorted(vals.unique().tolist())

def probe_brain_region_label(merged_dic, probe, roi_filter=None, kslabel_filter="both", max_regions=6, brain_region_filter=None, bc_label_filter=None):
    probe_df = pca_get_probe_units_df(
        merged_dic=merged_dic,
        probe=probe,
        roi_filter=roi_filter,
        kslabel_filter=kslabel_filter,
        bc_label_filter=bc_label_filter,
    )
    if probe_df.empty or ("brain_region" not in probe_df.columns):
        return "brain_region: n/a"

    vals = probe_df["brain_region"].map(_normalize_brain_region_value)

    if brain_region_filter is not None:
        return f"brain_region: {_normalize_brain_region_value(brain_region_filter)}"

    uniq = sorted(vals.unique().tolist())
    if len(uniq) == 0:
        return "brain_region: n/a"
    if len(uniq) <= max_regions:
        return "brain_region: " + ", ".join(uniq)
    head = ", ".join(uniq[:max_regions])
    return f"brain_region: {head}, +{len(uniq) - max_regions} more"

def plot_epoch_condition_scatter(
    Xp,
    event_meta,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=True,
):
    if Xp.shape[0] < 3:
        raise ValueError(f"Need at least 3 PCs for plotting; got {Xp.shape[0]}.")

    projections = [(0, 1), (1, 2), (0, 2)]
    style_ctx, cond_marker, _ = _epoch_condition_color_marker_maps(event_meta)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, (i, j) in zip(axes, projections):
        for cond in event_meta["condition"].unique():
            em_cond = event_meta[event_meta["condition"] == cond]
            marker = cond_marker.get(cond, "o")
            for ep in sorted(em_cond["epoch_id"].dropna().astype(int).unique()):
                idx = em_cond.index[em_cond["epoch_id"].astype(int) == ep].to_numpy()
                label = f"{cond}, epoch {ep}"
                ax.scatter(
                    Xp[i, idx],
                    Xp[j, idx],
                    s=140,
                    alpha=1.0,
                    marker=marker,
                    color=_epoch_condition_color_for(style_ctx, cond, ep),
                    edgecolors="black",
                    linewidths=0.8,
                    label=label,
                )

        ax.set_xlabel(f"PC {i+1}")
        ax.set_ylabel(f"PC {j+1}")

    handles, labels = axes[-1].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[-1].legend(uniq.values(), uniq.keys(), frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=10)

    fig.suptitle(title_prefix, fontsize=18, y=0.99)
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=11)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.91 if subtitle else 0.94, units_text, ha="center", va="center", fontsize=10)

    sns.despine()
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    save_path = Path(save_root) / '2D_scatter'
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_{_sanitize_name(brain_region)}_epoch_condition_scatter_2D.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def plot_epoch_condition_scatter_3d(
    Xp,
    event_meta,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if Xp.shape[0] < 3:
        raise ValueError(f"Need at least 3 PCs for 3D plotting; got {Xp.shape[0]}.")

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    style_ctx, cond_marker, _ = _epoch_condition_color_marker_maps(event_meta)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    for cond in event_meta["condition"].unique():
        em_cond = event_meta[event_meta["condition"] == cond]
        marker = cond_marker.get(cond, "o")
        for ep in sorted(em_cond["epoch_id"].dropna().astype(int).unique()):
            idx = em_cond.index[em_cond["epoch_id"].astype(int) == ep].to_numpy()
            label = f"{cond}, epoch {ep}"
            ax.scatter(
                Xp[0, idx],
                Xp[1, idx],
                Xp[2, idx],
                s=90,
                alpha=1.0,
                marker=marker,
                color=_epoch_condition_color_for(style_ctx, cond, ep),
                edgecolors="black",
                linewidths=0.6,
                label=label,
            )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)

    fig.suptitle(title_prefix + " (3D)", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.90 if subtitle else 0.93, units_text, ha="center", va="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.85, 0.9])

    save_path = Path(save_root)  / "3D_scatter"
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_{_sanitize_name(brain_region)}_epoch_condition_scatter_3D.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    if show_plots:
         plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def plot_epoch_condition_line_3d_time(
    Xp,
    event_meta,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    time_col="start_time",
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if Xp.shape[0] < 2:
        raise ValueError(f"Need at least 2 PCs for PC1-PC2-Time plotting; got {Xp.shape[0]}.")
    if time_col not in event_meta.columns:
        raise ValueError(f"{time_col} not found in event_meta columns: {list(event_meta.columns)}")

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    em = event_meta.copy()
    em[time_col] = pd.to_numeric(em[time_col], errors="coerce")
    em = em.dropna(subset=[time_col, "epoch_id", "condition"]).reset_index(drop=True)
    if em.empty:
        raise ValueError("No valid events available after dropping NaN times/labels.")

    t0 = em[time_col].min()
    em["time_rel_s"] = em[time_col] - t0

    style_ctx, cond_marker, cond_linestyle = _epoch_condition_color_marker_maps(em)

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")

    for cond in em["condition"].astype(str).unique():
        em_cond = em[em["condition"].astype(str) == cond]
        marker = cond_marker.get(cond, "o")
        ls = _epoch_condition_linestyle_for(cond_linestyle, cond)

        for ep in sorted(em_cond["epoch_id"].astype(int).unique()):
            idx = em_cond.index[em_cond["epoch_id"].astype(int) == ep].to_numpy()
            if idx.size < 2:
                continue

            order = np.argsort(em.loc[idx, "time_rel_s"].to_numpy(dtype=float))
            idx = idx[order]
            label = f"{cond}, epoch {ep}"
            t_rel = em.loc[idx, "time_rel_s"].to_numpy(dtype=float)

            ax.plot(
                Xp[0, idx],
                Xp[1, idx],
                t_rel,
                color=_epoch_condition_color_for(style_ctx, cond, ep),
                linestyle=ls,
                linewidth=2.2,
                marker=marker,
                markersize=4.0,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=label,
            )
            _plot_mode3_highlight(
                ax,
                style_ctx,
                Xp[0, idx],
                Xp[1, idx],
                t_rel,
                z=t_rel,
                condition=cond,
                epoch_id=ep,
                linewidth=2.2,
                alpha=1.0,
                linestyle=ls,
            )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("Time (s, rel)")

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)

    fig.suptitle(title_prefix + " (PC1-PC2-Time)", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.90 if subtitle else 0.93, units_text, ha="center", va="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.85, 0.9])

    save_path = Path(save_root)  / "3D_PC12_Time"
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_region_{_sanitize_name(brain_region)}_epoch_condition_line_PC12_Time.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

_EPOCH_PLOT_STYLE_DEFAULTS = {
    "color_mode": 1,
    "stimulation_linestyle": ":",
    "washout_linestyle": "-",
    "baseline_linestyle": "-",
    "baseline_color": "orange",
    "mode3_base_color": "#808080",
    "mode3_stimulation_color": "tab:blue",
    "mode3_washout_color": "tab:red",
    "mode3_baseline_color": "orange",
    "mode3_highlight_color": "#2ca25f",
    "mode3_highlight_window_s": 0.5,
}


def set_epoch_plot_style(
    color_mode=1,
    stimulation_linestyle=":",
    washout_linestyle="-",
    baseline_linestyle="-",
    baseline_color="orange",
    mode3_base_color="#808080",
    mode3_stimulation_color="tab:blue",
    mode3_washout_color="tab:red",
    mode3_baseline_color="orange",
    mode3_highlight_color="#2ca25f",
    mode3_highlight_window_s=0.5,
):
    """Set global style defaults used by epoch-condition plotting helpers."""
    global _EPOCH_PLOT_STYLE_DEFAULTS
    _EPOCH_PLOT_STYLE_DEFAULTS = {
        "color_mode": color_mode,
        "stimulation_linestyle": stimulation_linestyle,
        "washout_linestyle": washout_linestyle,
        "baseline_linestyle": baseline_linestyle,
        "baseline_color": baseline_color,
        "mode3_base_color": mode3_base_color,
        "mode3_stimulation_color": mode3_stimulation_color,
        "mode3_washout_color": mode3_washout_color,
        "mode3_baseline_color": mode3_baseline_color,
        "mode3_highlight_color": mode3_highlight_color,
        "mode3_highlight_window_s": mode3_highlight_window_s,
    }
    return _EPOCH_PLOT_STYLE_DEFAULTS.copy()


def _normalize_condition_name(condition):
    c = str(condition).strip().lower()
    if c.startswith("stim"):
        return "stimulation"
    if c.startswith("wash"):
        return "washout"
    if c.startswith("base"):
        return "baseline"
    return c


def _normalize_color_mode(color_mode):
    if isinstance(color_mode, str):
        key = color_mode.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
        if key in {"1", "mode1", "regular"}:
            return 1
        if key in {"2", "mode2", "stimwashgradient", "gradient"}:
            return 2
        if key in {"3", "mode3", "graygreen", "greensegment"}:
            return 3
    try:
        mode = int(color_mode)
        if mode in {1, 2, 3}:
            return mode
    except Exception:
        pass
    return 1


def _interp_color(c0, c1, t):
    a = np.asarray(mcolors.to_rgb(c0), dtype=float)
    b = np.asarray(mcolors.to_rgb(c1), dtype=float)
    t = float(np.clip(t, 0.0, 1.0))
    return tuple((1.0 - t) * a + t * b)


def _epoch_condition_color_marker_maps(
    event_meta,
    color_mode=None,
    stimulation_linestyle=None,
    washout_linestyle=None,
    baseline_linestyle=None,
    baseline_color=None,
    mode3_base_color=None,
    mode3_stimulation_color=None,
    mode3_washout_color=None,
    mode3_baseline_color=None,
    mode3_highlight_color=None,
    mode3_highlight_window_s=None,
):
    cfg = _EPOCH_PLOT_STYLE_DEFAULTS.copy()
    if color_mode is not None:
        cfg["color_mode"] = color_mode
    if stimulation_linestyle is not None:
        cfg["stimulation_linestyle"] = stimulation_linestyle
    if washout_linestyle is not None:
        cfg["washout_linestyle"] = washout_linestyle
    if baseline_linestyle is not None:
        cfg["baseline_linestyle"] = baseline_linestyle
    if baseline_color is not None:
        cfg["baseline_color"] = baseline_color
    if mode3_base_color is not None:
        cfg["mode3_base_color"] = mode3_base_color
    if mode3_stimulation_color is not None:
        cfg["mode3_stimulation_color"] = mode3_stimulation_color
    if mode3_washout_color is not None:
        cfg["mode3_washout_color"] = mode3_washout_color
    if mode3_baseline_color is not None:
        cfg["mode3_baseline_color"] = mode3_baseline_color
    if mode3_highlight_color is not None:
        cfg["mode3_highlight_color"] = mode3_highlight_color
    if mode3_highlight_window_s is not None:
        cfg["mode3_highlight_window_s"] = mode3_highlight_window_s

    mode = _normalize_color_mode(cfg["color_mode"])
    epochs = sorted(event_meta["epoch_id"].dropna().astype(int).unique().tolist())

    style_ctx = {
        "color_mode": mode,
        "epoch_color": {},
        "stim_epoch_color": {},
        "wash_epoch_color": {},
        "baseline_color": cfg["baseline_color"],
        "mode3_base_color": cfg["mode3_base_color"],
        "mode3_stimulation_color": cfg["mode3_stimulation_color"],
        "mode3_washout_color": cfg["mode3_washout_color"],
        "mode3_baseline_color": cfg["mode3_baseline_color"],
        "mode3_highlight_color": cfg["mode3_highlight_color"],
        "mode3_highlight_window_s": float(cfg["mode3_highlight_window_s"]),
    }

    n = max(1, len(epochs))
    for i, ep in enumerate(epochs):
        t = i / max(1, n - 1)
        style_ctx["stim_epoch_color"][ep] = _interp_color("#9ecae1", "#08519c", t)
        style_ctx["wash_epoch_color"][ep] = _interp_color("#fcbba1", "#a50f15", t)

    if mode == 1:
        pal = sns.color_palette("husl", max(3, len(epochs)))
        style_ctx["epoch_color"] = {ep: pal[i % len(pal)] for i, ep in enumerate(epochs)}

    cond_marker = {
        "baseline": "o",
        "stimulation": "^",
        "washout": "s",
    }
    cond_linestyle = {
        "baseline": cfg["baseline_linestyle"],
        "stimulation": cfg["stimulation_linestyle"],
        "washout": cfg["washout_linestyle"],
    }
    return style_ctx, cond_marker, cond_linestyle


def _epoch_condition_color_for(style_ctx, condition, epoch_id, default_color="tab:blue"):
    cond = _normalize_condition_name(condition)
    mode = int(style_ctx.get("color_mode", 1))
    ep = int(epoch_id)

    if mode == 3:
        return style_ctx.get("mode3_base_color", "#808080")

    if cond == "baseline":
        return style_ctx.get("baseline_color", "orange")

    if mode == 1:
        return style_ctx.get("epoch_color", {}).get(ep, default_color)
    if mode == 2:
        if cond == "stimulation":
            return style_ctx.get("stim_epoch_color", {}).get(ep, "#08519c")
        if cond == "washout":
            return style_ctx.get("wash_epoch_color", {}).get(ep, "#a50f15")
        return default_color
    return style_ctx.get("mode3_base_color", "#808080")


def _epoch_condition_mode3_highlight_color(style_ctx, condition, epoch_id):
    cond = _normalize_condition_name(condition)
    ep = int(epoch_id)
    if cond == "stimulation":
        return style_ctx.get("mode3_stimulation_color", style_ctx.get("stim_epoch_color", {}).get(ep, "#08519c"))
    if cond == "washout":
        return style_ctx.get("mode3_washout_color", style_ctx.get("wash_epoch_color", {}).get(ep, "#a50f15"))
    if cond == "baseline":
        return style_ctx.get("mode3_baseline_color", "orange")
    return style_ctx.get("mode3_highlight_color", "#2ca25f")


def _epoch_condition_linestyle_for(cond_linestyle, condition):
    cond = _normalize_condition_name(condition)
    return cond_linestyle.get(cond, cond_linestyle.get("baseline", "-"))


def _plot_mode3_highlight(
    ax,
    style_ctx,
    x,
    y,
    time_rel_s,
    z=None,
    condition=None,
    epoch_id=None,
    linewidth=2.0,
    alpha=0.9,
    linestyle="-",
):
    if int(style_ctx.get("color_mode", 1)) != 3:
        return
    t = np.asarray(time_rel_s, dtype=float)
    mask = np.isfinite(t) & (t >= 0.0) & (t <= float(style_ctx.get("mode3_highlight_window_s", 0.5)))
    if not np.any(mask):
        return

    xh = np.where(mask, np.asarray(x, dtype=float), np.nan)
    yh = np.where(mask, np.asarray(y, dtype=float), np.nan)
    color = _epoch_condition_mode3_highlight_color(style_ctx, condition, epoch_id)
    if z is None:
        ax.plot(
            xh,
            yh,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth + 0.4,
            alpha=min(1.0, alpha + 0.15),
        )
        return
    zh = np.where(mask, np.asarray(z, dtype=float), np.nan)
    ax.plot(
        xh,
        yh,
        zh,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth + 0.4,
        alpha=min(1.0, alpha + 0.15),
    )

def _epoch_mean_pc_table(Xp, event_meta, time_col="start_time"):
    em = event_meta.copy().reset_index(drop=True)
    if Xp.shape[1] != len(em):
        raise ValueError(f"Xp/events mismatch: Xp has {Xp.shape[1]} events, event_meta has {len(em)} rows.")

    em["_event_idx"] = np.arange(len(em), dtype=int)
    rows = []
    n_pc = min(3, Xp.shape[0])

    for (cond, ep), g in em.groupby(["condition", "epoch_id"], dropna=True):
        idx = g["_event_idx"].to_numpy(dtype=int)
        if idx.size == 0:
            continue

        rec = {
            "condition": str(cond),
            "epoch_id": int(ep),
            "n_trials": int(idx.size),
        }
        for pc in range(n_pc):
            rec[f"pc{pc+1}"] = float(np.mean(Xp[pc, idx]))

        if time_col in g.columns:
            tvals = pd.to_numeric(g[time_col], errors="coerce").dropna().to_numpy(dtype=float)
            rec["time_mean"] = float(np.mean(tvals)) if tvals.size else np.nan

        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No epoch groups available for epoch-mean plotting.")

    out = out.sort_values(["condition", "epoch_id"]).reset_index(drop=True)
    if "time_mean" in out.columns:
        t0 = np.nanmin(out["time_mean"].to_numpy(dtype=float))
        if np.isfinite(t0):
            out["time_rel_s"] = out["time_mean"] - t0

    return out

def plot_epoch_condition_epochmean_scatter(
    Xp,
    event_meta,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=True,
):
    if Xp.shape[0] < 3:
        raise ValueError(f"Need at least 3 PCs for plotting; got {Xp.shape[0]}.")

    em_mean = _epoch_mean_pc_table(Xp, event_meta)
    style_ctx, cond_marker, _ = _epoch_condition_color_marker_maps(event_meta)

    projections = [(0, 1), (1, 2), (0, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for ax, (i, j) in zip(axes, projections):
        for _, rec in em_mean.iterrows():
            cond = str(rec["condition"])
            ep = int(rec["epoch_id"])
            ax.scatter(
                rec[f"pc{i+1}"],
                rec[f"pc{j+1}"],
                s=240,
                alpha=1.0,
                marker=cond_marker.get(cond, "o"),
                color=_epoch_condition_color_for(style_ctx, cond, ep),
                edgecolors="black",
                linewidths=1.0,
                label=f"{cond}, epoch {ep} (n={int(rec['n_trials'])})",
            )

        ax.set_xlabel(f"PC {i+1}")
        ax.set_ylabel(f"PC {j+1}")

    handles, labels = axes[-1].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    axes[-1].legend(uniq.values(), uniq.keys(), frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=10)

    fig.suptitle(title_prefix + " (Epoch means)", fontsize=18, y=0.99)
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=11)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.91 if subtitle else 0.94, units_text, ha="center", va="center", fontsize=10)

    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.86, 0.9])

    save_path = Path(save_root) / "2D_scatter_epoch_avg"
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_region_{_sanitize_name(brain_region)}_epochmean_scatter_2D.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def plot_epoch_condition_epochmean_scatter_3d(
    Xp,
    event_meta,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if Xp.shape[0] < 3:
        raise ValueError(f"Need at least 3 PCs for 3D plotting; got {Xp.shape[0]}.")

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    em_mean = _epoch_mean_pc_table(Xp, event_meta)
    style_ctx, cond_marker, _ = _epoch_condition_color_marker_maps(event_meta)

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")

    for _, rec in em_mean.iterrows():
        cond = str(rec["condition"])
        ep = int(rec["epoch_id"])
        ax.scatter(
            rec["pc1"],
            rec["pc2"],
            rec["pc3"],
            s=120,
            alpha=1.0,
            marker=cond_marker.get(cond, "o"),
            color=_epoch_condition_color_for(style_ctx, cond, ep),
            edgecolors="black",
            linewidths=0.8,
            label=f"{cond}, epoch {ep} (n={int(rec['n_trials'])})",
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)

    fig.suptitle(title_prefix + " (Epoch means, 3D)", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.90 if subtitle else 0.93, units_text, ha="center", va="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.85, 0.9])

    save_path = Path(save_root) / "3D_scatter_epoch_avg"
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_region_{_sanitize_name(brain_region)}_epochmean_scatter_3D.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def plot_epoch_condition_epochmean_line_3d_time(
    Xp,
    event_meta,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    smooth_sigma=1.2,
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if Xp.shape[0] < 2:
        raise ValueError(f"Need at least 2 PCs for PC1-PC2-Time plotting; got {Xp.shape[0]}.")

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    em_mean = _epoch_mean_pc_table(Xp, event_meta)
    if "time_rel_s" not in em_mean.columns:
        em_mean = em_mean.copy()
        em_mean["time_rel_s"] = em_mean["epoch_id"].astype(float)

    style_ctx, cond_marker, cond_linestyle = _epoch_condition_color_marker_maps(event_meta)

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")

    for cond in em_mean["condition"].astype(str).unique():
        g = em_mean[em_mean["condition"].astype(str) == cond].sort_values("epoch_id")
        if g.empty:
            continue

        x = g["pc1"].to_numpy(dtype=float)
        y = g["pc2"].to_numpy(dtype=float)
        z = g["time_rel_s"].to_numpy(dtype=float)

        if smooth_sigma and smooth_sigma > 0 and len(x) >= 3:
            x = gaussian_filter1d(x, sigma=smooth_sigma)
            y = gaussian_filter1d(y, sigma=smooth_sigma)
            z = gaussian_filter1d(z, sigma=smooth_sigma)

        trend_color = _epoch_condition_color_for(style_ctx, cond, int(g["epoch_id"].iloc[-1]))
        ax.plot(
            x,
            y,
            z,
            color=trend_color,
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
            linewidth=2.4,
            alpha=0.85,
            label=f"{cond} trend",
        )

        for ii, (_, rec) in enumerate(g.iterrows()):
            ep = int(rec["epoch_id"])
            ax.scatter(
                x[ii],
                y[ii],
                z[ii],
                s=120,
                alpha=1.0,
                marker=cond_marker.get(cond, "o"),
                color=_epoch_condition_color_for(style_ctx, cond, ep),
                edgecolors="black",
                linewidths=0.8,
                label=f"{cond}, epoch {ep} (n={int(rec['n_trials'])})",
            )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("Time (s, rel)")

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)

    fig.suptitle(title_prefix + " (Epoch means, PC1-PC2-Time)", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.90 if subtitle else 0.93, units_text, ha="center", va="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.85, 0.9])

    save_path = Path(save_root) / "3D_PC12_Time"
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_region_{_sanitize_name(brain_region)}_epochmean_line_PC12_Time.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def pca_trial_timebin_scores(trials, n_components=6):
    if trials.ndim != 3:
        raise ValueError(f"Expected trials shape (n_trials, n_units, n_bins), got {trials.shape}")

    n_trials, n_units, n_bins = trials.shape
    X = trials.transpose(0, 2, 1).reshape(n_trials * n_bins, n_units)

    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    pca = PCA(n_components=min(n_components, Xz.shape[0], Xz.shape[1]))
    S = pca.fit_transform(Xz)
    S = S.reshape(n_trials, n_bins, -1)
    return S, pca.explained_variance_ratio_

def plot_epoch_condition_trial_time_lines_2d(
    trial_time_scores,
    event_meta,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    smooth_sigma=1.2,
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if trial_time_scores.shape[2] < 2:
        raise ValueError(f"Need at least 2 PCs; got {trial_time_scores.shape[2]}.")
    if trial_time_scores.shape[0] != len(event_meta):
        raise ValueError("Mismatch between trial_time_scores trials and event_meta rows.")

    em = event_meta.copy().reset_index(drop=True)
    style_ctx, _, cond_linestyle = _epoch_condition_color_marker_maps(em)

    fig, ax = plt.subplots(1, 1, figsize=(13, 10))
    seen = set()

    for i in range(len(em)):
        cond = str(em.loc[i, "condition"])
        ep = int(em.loc[i, "epoch_id"])
        x = trial_time_scores[i, :, 0].astype(float)
        y = trial_time_scores[i, :, 1].astype(float)

        if smooth_sigma and smooth_sigma > 0:
            x = gaussian_filter1d(x, sigma=smooth_sigma)
            y = gaussian_filter1d(y, sigma=smooth_sigma)

        label = f"{cond}, epoch {ep}"
        if label in seen:
            label = None
        else:
            seen.add(label)

        ax.plot(
            x,
            y,
            color=_epoch_condition_color_for(style_ctx, cond, ep),
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
            linewidth=1.6,
            alpha=0.65,
            label=label,
        )
        ax.scatter(
            x,
            y,
            s=8,
            color=_epoch_condition_color_for(style_ctx, cond, ep),
            alpha=0.35,
            linewidths=0,
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=9)

    fig.suptitle(title_prefix + " (Trial lines from time bins, 2D)", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.90 if subtitle else 0.93, units_text, ha="center", va="center", fontsize=10)

    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 0.9])

    save_path = Path(save_root) / "TrialTimeLines_2D"
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_region_{_sanitize_name(brain_region)}_trial_time_lines_2D.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def plot_epoch_condition_trial_time_lines_3d(
    trial_time_scores,
    event_meta,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    smooth_sigma=1.2,
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if trial_time_scores.shape[2] < 3:
        raise ValueError(f"Need at least 3 PCs; got {trial_time_scores.shape[2]}.")
    if trial_time_scores.shape[0] != len(event_meta):
        raise ValueError("Mismatch between trial_time_scores trials and event_meta rows.")

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    em = event_meta.copy().reset_index(drop=True)
    style_ctx, _, cond_linestyle = _epoch_condition_color_marker_maps(em)

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    seen = set()

    for i in range(len(em)):
        cond = str(em.loc[i, "condition"])
        ep = int(em.loc[i, "epoch_id"])
        x = trial_time_scores[i, :, 0].astype(float)
        y = trial_time_scores[i, :, 1].astype(float)
        z = trial_time_scores[i, :, 2].astype(float)

        if smooth_sigma and smooth_sigma > 0:
            x = gaussian_filter1d(x, sigma=smooth_sigma)
            y = gaussian_filter1d(y, sigma=smooth_sigma)
            z = gaussian_filter1d(z, sigma=smooth_sigma)

        label = f"{cond}, epoch {ep}"
        if label in seen:
            label = None
        else:
            seen.add(label)

        ax.plot(
            x,
            y,
            z,
            color=_epoch_condition_color_for(style_ctx, cond, ep),
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
            linewidth=1.6,
            alpha=0.65,
            label=label,
        )
        ax.scatter(
            x,
            y,
            z,
            s=6,
            color=_epoch_condition_color_for(style_ctx, cond, ep),
            alpha=0.30,
            linewidths=0,
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=9)

    fig.suptitle(title_prefix + " (Trial lines from time bins, 3D)", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.90 if subtitle else 0.93, units_text, ha="center", va="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.85, 0.9])

    save_path = Path(save_root) / "TrialTimeLines_3D"
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_region_{_sanitize_name(brain_region)}_trial_time_lines_3D.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def plot_epoch_condition_epochmean_scatter_3d_time(
    Xp,
    event_meta,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if Xp.shape[0] < 2:
        raise ValueError(f"Need at least 2 PCs for PC1-PC2-Time plotting; got {Xp.shape[0]}.")

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    em_mean = _epoch_mean_pc_table(Xp, event_meta)
    if "time_rel_s" not in em_mean.columns:
        raise ValueError("time_rel_s missing from epoch-mean table. Check event time column in event_meta.")

    style_ctx, cond_marker, _ = _epoch_condition_color_marker_maps(event_meta)

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")

    for _, rec in em_mean.iterrows():
        cond = str(rec["condition"])
        ep = int(rec["epoch_id"])
        ax.scatter(
            rec["pc1"],
            rec["pc2"],
            rec["time_rel_s"],
            s=120,
            alpha=1.0,
            marker=cond_marker.get(cond, "o"),
            color=_epoch_condition_color_for(style_ctx, cond, ep),
            edgecolors="black",
            linewidths=0.8,
            label=f"{cond}, epoch {ep} (n={int(rec['n_trials'])})",
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("Time (s, rel)")

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)

    fig.suptitle(title_prefix + " (Epoch means, PC1-PC2-Time points)", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.90 if subtitle else 0.93, units_text, ha="center", va="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.85, 0.9])

    save_path = Path(save_root) / "3D_PC12_Time_epoch_avg"
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_region_{_sanitize_name(brain_region)}_epochmean_scatter_PC12_Time.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def plot_epoch_condition_trial_time_lines_pc12_time(
    trial_time_scores,
    event_meta,
    bin_time=None,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    smooth_sigma=1.2,
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if trial_time_scores.shape[2] < 2:
        raise ValueError(f"Need at least 2 PCs; got {trial_time_scores.shape[2]}.")
    if trial_time_scores.shape[0] != len(event_meta):
        raise ValueError("Mismatch between trial_time_scores trials and event_meta rows.")

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n_trials, n_bins, _ = trial_time_scores.shape
    if bin_time is None:
        z = np.arange(n_bins, dtype=float)
    else:
        z = np.asarray(bin_time, dtype=float).ravel()
        if z.size != n_bins:
            raise ValueError(f"bin_time length ({z.size}) must equal n_bins ({n_bins}).")
    z = z.copy()

    em = event_meta.copy().reset_index(drop=True)
    style_ctx, _, cond_linestyle = _epoch_condition_color_marker_maps(em)

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    seen = set()

    for i in range(n_trials):
        cond = str(em.loc[i, "condition"])
        ep = int(em.loc[i, "epoch_id"])

        x = trial_time_scores[i, :, 0].astype(float)
        y = trial_time_scores[i, :, 1].astype(float)

        if smooth_sigma and smooth_sigma > 0:
            x = gaussian_filter1d(x, sigma=smooth_sigma)
            y = gaussian_filter1d(y, sigma=smooth_sigma)

        label = f"{cond}, epoch {ep}"
        if label in seen:
            label = None
        else:
            seen.add(label)

        ax.plot(
            x,
            y,
            z,
            color=_epoch_condition_color_for(style_ctx, cond, ep),
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
            linewidth=1.6,
            alpha=0.65,
            label=label,
        )
        _plot_mode3_highlight(
            ax,
            style_ctx,
            x,
            y,
            z,
            z=z,
            condition=cond,
            epoch_id=ep,
            linewidth=1.6,
            alpha=0.65,
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
        )
        ax.scatter(
            x,
            y,
            z,
            s=6,
            color=_epoch_condition_color_for(style_ctx, cond, ep),
            alpha=0.30,
            linewidths=0,
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("Time from event (s)")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=9)

    fig.suptitle(title_prefix + " (Per-trial lines, PC1-PC2-Time)", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.90 if subtitle else 0.93, units_text, ha="center", va="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.85, 0.9])

    save_path = Path(save_root) / "3D_PC12_Time"
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_region_{_sanitize_name(brain_region)}_trial_time_lines_PC12_Time.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def pca_epoch_population_timebin_trajectories(trials, event_meta, n_components=6):
    """
    Build one average population trajectory per (condition, epoch_id):
    - average trials within each epoch -> (n_units, n_bins)
    - concatenate all epoch means across time bins
    - run PCA once across concatenated epoch trajectories
    Returns:
        epoch_info_df: row per epoch group with labels and counts
        epoch_traj: list of arrays, each shape (n_components_used, n_bins)
        evr: explained variance ratio
    """
    if trials.ndim != 3:
        raise ValueError(f"Expected trials shape (n_trials, n_units, n_bins), got {trials.shape}")

    n_trials, n_units, n_bins = trials.shape
    em = event_meta.copy().reset_index(drop=True)
    if len(em) != n_trials:
        raise ValueError(f"event_meta rows ({len(em)}) must match n_trials ({n_trials}).")

    for col in ["condition", "epoch_id"]:
        if col not in em.columns:
            raise ValueError(f"Required column '{col}' missing from event_meta.")

    em["_trial_idx"] = np.arange(len(em), dtype=int)

    rows = []
    epoch_means = []
    for (cond, ep), g in em.groupby(["condition", "epoch_id"], dropna=True):
        idx = g["_trial_idx"].to_numpy(dtype=int)
        if idx.size == 0:
            continue

        mean_pop = trials[idx].mean(axis=0)  # (n_units, n_bins)
        epoch_means.append(mean_pop)

        rec = {
            "condition": str(cond),
            "epoch_id": int(ep),
            "n_trials": int(idx.size),
            "first_trial_idx": int(np.min(idx)),
        }
        if "start_time" in g.columns:
            st = pd.to_numeric(g["start_time"], errors="coerce").dropna().to_numpy(dtype=float)
            rec["start_time_mean"] = float(np.mean(st)) if st.size else np.nan
        rows.append(rec)

    if len(rows) == 0:
        raise ValueError("No epoch groups found to build population trajectories.")

    epoch_info_df = pd.DataFrame(rows)
    if "start_time_mean" in epoch_info_df.columns:
        epoch_info_df = epoch_info_df.sort_values(["start_time_mean", "first_trial_idx"], na_position="last").reset_index(drop=True)
    else:
        epoch_info_df = epoch_info_df.sort_values(["condition", "epoch_id", "first_trial_idx"]).reset_index(drop=True)

    # Reorder epoch means to match epoch_info_df
    key_to_mean = {}
    for k, row in enumerate(rows):
        key_to_mean[(row["condition"], int(row["epoch_id"]), int(row["first_trial_idx"]))] = epoch_means[k]
    ordered_means = []
    for _, row in epoch_info_df.iterrows():
        ordered_means.append(key_to_mean[(str(row["condition"]), int(row["epoch_id"]), int(row["first_trial_idx"]))])

    Xa = np.hstack(ordered_means)  # (n_units, n_epochs * n_bins)
    Xaz = zscore_rows(Xa)
    pca = PCA(n_components=min(n_components, Xaz.shape[0], Xaz.shape[1]))
    Xa_p = pca.fit_transform(Xaz.T).T

    epoch_traj = []
    for i in range(len(epoch_info_df)):
        s = i * n_bins
        e = (i + 1) * n_bins
        epoch_traj.append(Xa_p[:, s:e])

    return epoch_info_df, epoch_traj, pca.explained_variance_ratio_

def plot_epoch_population_lines_2d(
    epoch_info_df,
    epoch_traj,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    smooth_sigma=1.2,
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if len(epoch_traj) == 0:
        raise ValueError("epoch_traj is empty.")
    if epoch_traj[0].shape[0] < 2:
        raise ValueError(f"Need at least 2 PCs; got {epoch_traj[0].shape[0]}.")

    style_ctx, cond_marker, cond_linestyle = _epoch_condition_color_marker_maps(epoch_info_df)

    fig, ax = plt.subplots(1, 1, figsize=(13, 10))
    for i, rec in epoch_info_df.reset_index(drop=True).iterrows():
        cond = str(rec["condition"])
        ep = int(rec["epoch_id"])
        x = epoch_traj[i][0].astype(float)
        y = epoch_traj[i][1].astype(float)

        if smooth_sigma and smooth_sigma > 0 and x.size >= 3:
            x = gaussian_filter1d(x, sigma=smooth_sigma)
            y = gaussian_filter1d(y, sigma=smooth_sigma)

        label = f"{cond}, epoch {ep} (n={int(rec['n_trials'])})"
        ax.plot(
            x,
            y,
            color=_epoch_condition_color_for(style_ctx, cond, ep),
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
            linewidth=2.4,
            alpha=0.9,
            label=label,
        )
        ax.scatter(
            x,
            y,
            s=12,
            color=_epoch_condition_color_for(style_ctx, cond, ep),
            marker=cond_marker.get(cond, "o"),
            alpha=0.35,
            linewidths=0,
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=9)

    fig.suptitle(title_prefix + " (One line per epoch, 2D)", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.90 if subtitle else 0.93, units_text, ha="center", va="center", fontsize=10)

    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 0.9])

    save_path = Path(save_root) / "EpochPopulationLines_2D"
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_region_{_sanitize_name(brain_region)}_epoch_population_lines_2D.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def plot_epoch_population_lines_3d(
    epoch_info_df,
    epoch_traj,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    smooth_sigma=1.2,
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if len(epoch_traj) == 0:
        raise ValueError("epoch_traj is empty.")
    if epoch_traj[0].shape[0] < 3:
        raise ValueError(f"Need at least 3 PCs; got {epoch_traj[0].shape[0]}.")

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    style_ctx, cond_marker, cond_linestyle = _epoch_condition_color_marker_maps(epoch_info_df)

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")

    for i, rec in epoch_info_df.reset_index(drop=True).iterrows():
        cond = str(rec["condition"])
        ep = int(rec["epoch_id"])
        x = epoch_traj[i][0].astype(float)
        y = epoch_traj[i][1].astype(float)
        z = epoch_traj[i][2].astype(float)

        if smooth_sigma and smooth_sigma > 0 and x.size >= 3:
            x = gaussian_filter1d(x, sigma=smooth_sigma)
            y = gaussian_filter1d(y, sigma=smooth_sigma)
            z = gaussian_filter1d(z, sigma=smooth_sigma)

        label = f"{cond}, epoch {ep} (n={int(rec['n_trials'])})"
        ax.plot(
            x,
            y,
            z,
            color=_epoch_condition_color_for(style_ctx, cond, ep),
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
            linewidth=2.2,
            alpha=0.9,
            label=label,
        )
        ax.scatter(
            x,
            y,
            z,
            s=8,
            color=_epoch_condition_color_for(style_ctx, cond, ep),
            marker=cond_marker.get(cond, "o"),
            alpha=0.30,
            linewidths=0,
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=9)

    fig.suptitle(title_prefix + " (One line per epoch, 3D)", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.90 if subtitle else 0.93, units_text, ha="center", va="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.85, 0.9])

    save_path = Path(save_root) / "EpochPopulationLines_3D"
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_region_{_sanitize_name(brain_region)}_epoch_population_lines_3D.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def plot_epoch_population_lines_pc12_time(
    epoch_info_df,
    epoch_traj,
    bin_time,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    smooth_sigma=1.2,
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if len(epoch_traj) == 0:
        raise ValueError("epoch_traj is empty.")
    if epoch_traj[0].shape[0] < 2:
        raise ValueError(f"Need at least 2 PCs; got {epoch_traj[0].shape[0]}.")

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    bt = np.asarray(bin_time, dtype=float).ravel()
    n_bins = epoch_traj[0].shape[1]
    if bt.size != n_bins:
        raise ValueError(f"bin_time length ({bt.size}) must equal n_bins ({n_bins}).")
    z = bt.copy()

    style_ctx, cond_marker, cond_linestyle = _epoch_condition_color_marker_maps(epoch_info_df)

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")

    for i, rec in epoch_info_df.reset_index(drop=True).iterrows():
        cond = str(rec["condition"])
        ep = int(rec["epoch_id"])
        x = epoch_traj[i][0].astype(float)
        y = epoch_traj[i][1].astype(float)

        if smooth_sigma and smooth_sigma > 0 and x.size >= 3:
            x = gaussian_filter1d(x, sigma=smooth_sigma)
            y = gaussian_filter1d(y, sigma=smooth_sigma)

        label = f"{cond}, epoch {ep} (n={int(rec['n_trials'])})"
        ax.plot(
            x,
            y,
            z,
            color=_epoch_condition_color_for(style_ctx, cond, ep),
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
            linewidth=2.2,
            alpha=0.9,
            label=label,
        )
        _plot_mode3_highlight(
            ax,
            style_ctx,
            x,
            y,
            z,
            z=z,
            condition=cond,
            epoch_id=ep,
            linewidth=2.2,
            alpha=0.9,
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
        )
        ax.scatter(
            x,
            y,
            z,
            s=8,
            color=_epoch_condition_color_for(style_ctx, cond, ep),
            marker=cond_marker.get(cond, "o"),
            alpha=0.30,
            linewidths=0,
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("Time from event (s)")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=9)

    fig.suptitle(title_prefix + " (One line per epoch, PC1-PC2-Time)", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.90 if subtitle else 0.93, units_text, ha="center", va="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.85, 0.9])

    save_path = Path(save_root) / "EpochPopulationLines_PC12_Time"
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_region_{_sanitize_name(brain_region)}_epoch_population_lines_PC12_Time.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def plot_epoch_condition_scatter_epoch_avg(*args, **kwargs):
    return plot_epoch_condition_epochmean_scatter(*args, **kwargs)


def plot_epoch_condition_scatter_3d_epoch_avg(*args, **kwargs):
    return plot_epoch_condition_epochmean_scatter_3d(*args, **kwargs)


def plot_epoch_condition_line_3d_time_epoch_avg(
    trials,
    event_meta,
    time_bins,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    smooth_sigma=1.2,
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    epoch_info_df, epoch_traj, _ = pca_epoch_population_timebin_trajectories(
        trials=trials,
        event_meta=event_meta,
        n_components=3,
    )
    return plot_epoch_population_lines_pc12_time(
        epoch_info_df=epoch_info_df,
        epoch_traj=epoch_traj,
        bin_time=time_bins,
        title_prefix=title_prefix,
        subtitle=subtitle,
        probe=probe,
        brain_region=brain_region,
        smooth_sigma=smooth_sigma,
        n_units=n_units,
        save_root=save_root,
        show_plots=show_plots,
    )

# === GROUPED EPOCH PLOT SETS ===

def plot_epoch_condition_group_base_trial(
    Xp,
    event_meta,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    time_col="start_time",
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if Xp.shape[0] < 3:
        raise ValueError(f"Need at least 3 PCs; got {Xp.shape[0]}.")
    if time_col not in event_meta.columns:
        raise ValueError(f"{time_col} not in event_meta columns: {list(event_meta.columns)}")

    em = event_meta.copy().reset_index(drop=True)
    em[time_col] = pd.to_numeric(em[time_col], errors="coerce")
    em = em.dropna(subset=[time_col, "condition", "epoch_id"]).reset_index(drop=True)
    if len(em) != Xp.shape[1]:
        raise ValueError(f"event_meta rows ({len(em)}) must match Xp events ({Xp.shape[1]}).")

    t0 = em[time_col].min()
    em["time_rel_s"] = em[time_col] - t0

    style_ctx, cond_marker, cond_linestyle = _epoch_condition_color_marker_maps(em)

    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(1, 3, wspace=0.25)
    ax2d = fig.add_subplot(gs[0, 0])
    ax3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax3t = fig.add_subplot(gs[0, 2], projection="3d")

    for cond in em["condition"].astype(str).unique():
        em_cond = em[em["condition"].astype(str) == cond]
        marker = cond_marker.get(cond, "o")
        ls = _epoch_condition_linestyle_for(cond_linestyle, cond)
        for ep in sorted(em_cond["epoch_id"].astype(int).unique()):
            idx = em_cond.index[em_cond["epoch_id"].astype(int) == ep].to_numpy()
            label = f"{cond}, epoch {ep}"

            ax2d.scatter(
                Xp[0, idx], Xp[1, idx],
                s=45, alpha=0.9, marker=marker,
                color=_epoch_condition_color_for(style_ctx, cond, ep),
                edgecolors="black", linewidths=0.5,
                label=label,
            )

            ax3d.scatter(
                Xp[0, idx], Xp[1, idx], Xp[2, idx],
                s=30, alpha=0.85, marker=marker,
                color=_epoch_condition_color_for(style_ctx, cond, ep),
                edgecolors="black", linewidths=0.4,
                label=label,
            )

            if idx.size >= 2:
                ord_idx = np.argsort(em.loc[idx, "time_rel_s"].to_numpy(dtype=float))
                idx2 = idx[ord_idx]
                ax3t.plot(
                    Xp[0, idx2], Xp[1, idx2], em.loc[idx2, "time_rel_s"].to_numpy(dtype=float),
                    color=_epoch_condition_color_for(style_ctx, cond, ep), linestyle=ls, linewidth=2.0,
                    marker=marker, markersize=3.5,
                    markeredgecolor="black", markeredgewidth=0.5,
                    label=label,
                )
            else:
                ax3t.scatter(
                    Xp[0, idx], Xp[1, idx], em.loc[idx, "time_rel_s"].to_numpy(dtype=float),
                    s=30, alpha=0.85, marker=marker,
                    color=_epoch_condition_color_for(style_ctx, cond, ep),
                    edgecolors="black", linewidths=0.4,
                    label=label,
                )

    ax2d.set_title("2D: PC1 vs PC2")
    ax2d.set_xlabel("PC 1")
    ax2d.set_ylabel("PC 2")

    ax3d.set_title("3D: PC1-PC2-PC3")
    ax3d.set_xlabel("PC 1")
    ax3d.set_ylabel("PC 2")
    ax3d.set_zlabel("PC 3")

    ax3t.set_title("3D: PC1-PC2-Time")
    ax3t.set_xlabel("PC 1")
    ax3t.set_ylabel("PC 2")
    ax3t.set_zlabel("Time (s, rel)")

    handles, labels = ax3t.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax3t.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, fontsize=9)

    fig.suptitle(title_prefix + " | Group: Base Trial-Level Epoch Plots", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.90 if subtitle else 0.93, units_text, ha="center", va="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.86, 0.9])

    save_path = Path(save_root) / "Grouped_BaseTrial"
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_region_{_sanitize_name(brain_region)}_group_base_trial.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def plot_epoch_condition_group_epochmean(
    Xp,
    event_meta,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    smooth_sigma=1.2,
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if Xp.shape[0] < 3:
        raise ValueError(f"Need at least 3 PCs; got {Xp.shape[0]}.")

    em_mean = _epoch_mean_pc_table(Xp, event_meta)
    style_ctx, cond_marker, cond_linestyle = _epoch_condition_color_marker_maps(event_meta)

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, wspace=0.2, hspace=0.25)
    ax2d = fig.add_subplot(gs[0, 0])
    ax3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax3t_pts = fig.add_subplot(gs[1, 0], projection="3d")
    ax3t_line = fig.add_subplot(gs[1, 1], projection="3d")

    for _, rec in em_mean.iterrows():
        cond = str(rec["condition"])
        ep = int(rec["epoch_id"])
        label = f"{cond}, epoch {ep} (n={int(rec['n_trials'])})"

        ax2d.scatter(
            rec["pc1"], rec["pc2"],
            s=220, alpha=1.0, marker=cond_marker.get(cond, "o"),
            color=_epoch_condition_color_for(style_ctx, cond, ep), edgecolors="black", linewidths=0.9,
            label=label,
        )

        ax3d.scatter(
            rec["pc1"], rec["pc2"], rec["pc3"],
            s=110, alpha=1.0, marker=cond_marker.get(cond, "o"),
            color=_epoch_condition_color_for(style_ctx, cond, ep), edgecolors="black", linewidths=0.7,
            label=label,
        )

        ax3t_pts.scatter(
            rec["pc1"], rec["pc2"], rec["time_rel_s"],
            s=110, alpha=1.0, marker=cond_marker.get(cond, "o"),
            color=_epoch_condition_color_for(style_ctx, cond, ep), edgecolors="black", linewidths=0.7,
            label=label,
        )

    for cond in em_mean["condition"].astype(str).unique():
        g = em_mean[em_mean["condition"].astype(str) == cond].sort_values("epoch_id")
        if g.empty:
            continue

        x = g["pc1"].to_numpy(dtype=float)
        y = g["pc2"].to_numpy(dtype=float)
        z = g["time_rel_s"].to_numpy(dtype=float)

        if smooth_sigma and smooth_sigma > 0 and len(x) >= 3:
            x = gaussian_filter1d(x, sigma=smooth_sigma)
            y = gaussian_filter1d(y, sigma=smooth_sigma)
            z = gaussian_filter1d(z, sigma=smooth_sigma)

        trend_color = _epoch_condition_color_for(style_ctx, cond, int(g["epoch_id"].iloc[-1]))
        ax3t_line.plot(
            x, y, z,
            color=trend_color, linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond), linewidth=2.3,
            alpha=0.85, label=f"{cond} trend",
        )

        for ii, (_, rec) in enumerate(g.iterrows()):
            ep = int(rec["epoch_id"])
            ax3t_line.scatter(
                x[ii], y[ii], z[ii],
                s=90, alpha=1.0, marker=cond_marker.get(cond, "o"),
                color=_epoch_condition_color_for(style_ctx, cond, ep), edgecolors="black", linewidths=0.6,
                label=f"{cond}, epoch {ep} (n={int(rec['n_trials'])})",
            )

    ax2d.set_title("2D: Epoch mean PC1 vs PC2")
    ax2d.set_xlabel("PC 1")
    ax2d.set_ylabel("PC 2")

    ax3d.set_title("3D: Epoch mean PC1-PC2-PC3")
    ax3d.set_xlabel("PC 1")
    ax3d.set_ylabel("PC 2")
    ax3d.set_zlabel("PC 3")

    ax3t_pts.set_title("3D: Epoch mean PC1-PC2-Time points")
    ax3t_pts.set_xlabel("PC 1")
    ax3t_pts.set_ylabel("PC 2")
    ax3t_pts.set_zlabel("Time (s, rel)")

    ax3t_line.set_title("3D: Epoch mean PC1-PC2-Time lines")
    ax3t_line.set_xlabel("PC 1")
    ax3t_line.set_ylabel("PC 2")
    ax3t_line.set_zlabel("Time (s, rel)")

    handles, labels = ax3t_line.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax3t_line.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, fontsize=8)

    fig.suptitle(title_prefix + " | Group: Epoch-Mean Family", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=10)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.91 if subtitle else 0.94, units_text, ha="center", va="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.86, 0.92])

    save_path = Path(save_root) / "Grouped_EpochMean"
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_region_{_sanitize_name(brain_region)}_group_epochmean.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def plot_epoch_condition_group_trial_time(
    trial_time_scores,
    event_meta,
    bin_time=None,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    smooth_sigma=1.2,
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if trial_time_scores.shape[2] < 3:
        raise ValueError(f"Need at least 3 PCs; got {trial_time_scores.shape[2]}.")
    if trial_time_scores.shape[0] != len(event_meta):
        raise ValueError("Mismatch between trial_time_scores and event_meta rows.")

    n_trials, n_bins, _ = trial_time_scores.shape
    if bin_time is None:
        zt = np.arange(n_bins, dtype=float)
    else:
        zt = np.asarray(bin_time, dtype=float).ravel()
        if zt.size != n_bins:
            raise ValueError(f"bin_time length ({zt.size}) must equal n_bins ({n_bins}).")
    zt = zt.copy()

    em = event_meta.copy().reset_index(drop=True)
    style_ctx, _, cond_linestyle = _epoch_condition_color_marker_maps(em)

    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(1, 3, wspace=0.25)
    ax2d = fig.add_subplot(gs[0, 0])
    ax3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax3t = fig.add_subplot(gs[0, 2], projection="3d")

    seen = set()
    for i in range(n_trials):
        cond = str(em.loc[i, "condition"])
        ep = int(em.loc[i, "epoch_id"])
        x = trial_time_scores[i, :, 0].astype(float)
        y = trial_time_scores[i, :, 1].astype(float)
        z = trial_time_scores[i, :, 2].astype(float)

        if smooth_sigma and smooth_sigma > 0:
            x = gaussian_filter1d(x, sigma=smooth_sigma)
            y = gaussian_filter1d(y, sigma=smooth_sigma)
            z = gaussian_filter1d(z, sigma=smooth_sigma)

        label = f"{cond}, epoch {ep}"
        label2 = label if label not in seen else None
        seen.add(label)

        ax2d.plot(x, y, color=_epoch_condition_color_for(style_ctx, cond, ep), linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond), linewidth=1.5, alpha=0.65, label=label2)
        _plot_mode3_highlight(
            ax2d,
            style_ctx,
            x,
            y,
            zt,
            condition=cond,
            epoch_id=ep,
            linewidth=1.5,
            alpha=0.65,
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
        )
        ax3d.plot(x, y, z, color=_epoch_condition_color_for(style_ctx, cond, ep), linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond), linewidth=1.4, alpha=0.65, label=label2)
        _plot_mode3_highlight(
            ax3d,
            style_ctx,
            x,
            y,
            zt,
            z=z,
            condition=cond,
            epoch_id=ep,
            linewidth=1.4,
            alpha=0.65,
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
        )
        ax3t.plot(x, y, zt, color=_epoch_condition_color_for(style_ctx, cond, ep), linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond), linewidth=1.5, alpha=0.65, label=label2)
        _plot_mode3_highlight(
            ax3t,
            style_ctx,
            x,
            y,
            zt,
            z=zt,
            condition=cond,
            epoch_id=ep,
            linewidth=1.5,
            alpha=0.65,
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
        )

    ax2d.set_title("2D: Per-trial time-bin lines (PC1-PC2)")
    ax2d.set_xlabel("PC 1")
    ax2d.set_ylabel("PC 2")

    ax3d.set_title("3D: Per-trial time-bin lines (PC1-PC2-PC3)")
    ax3d.set_xlabel("PC 1")
    ax3d.set_ylabel("PC 2")
    ax3d.set_zlabel("PC 3")

    ax3t.set_title("3D: Per-trial time-bin lines (PC1-PC2-Time)")
    ax3t.set_xlabel("PC 1")
    ax3t.set_ylabel("PC 2")
    ax3t.set_zlabel("Time from event (s)")

    handles, labels = ax3t.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax3t.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, fontsize=9)

    fig.suptitle(title_prefix + " | Group: Per-Trial Time-Bin Family", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.90 if subtitle else 0.93, units_text, ha="center", va="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.86, 0.9])

    save_path = Path(save_root) / "Grouped_TrialTime"
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_region_{_sanitize_name(brain_region)}_group_trial_time.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def plot_epoch_condition_group_epoch_population(
    epoch_info_df,
    epoch_traj,
    bin_time,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    smooth_sigma=1.2,
    n_units=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    if len(epoch_traj) == 0:
        raise ValueError("epoch_traj is empty.")
    if epoch_traj[0].shape[0] < 3:
        raise ValueError(f"Need at least 3 PCs; got {epoch_traj[0].shape[0]}.")

    bt = np.asarray(bin_time, dtype=float).ravel()
    n_bins = epoch_traj[0].shape[1]
    if bt.size != n_bins:
        raise ValueError(f"bin_time length ({bt.size}) must equal n_bins ({n_bins}).")
    zt = bt.copy()

    style_ctx, cond_marker, cond_linestyle = _epoch_condition_color_marker_maps(epoch_info_df)

    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(1, 3, wspace=0.25)
    ax2d = fig.add_subplot(gs[0, 0])
    ax3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax3t = fig.add_subplot(gs[0, 2], projection="3d")

    for i, rec in epoch_info_df.reset_index(drop=True).iterrows():
        cond = str(rec["condition"])
        ep = int(rec["epoch_id"])
        label = f"{cond}, epoch {ep} (n={int(rec['n_trials'])})"

        x = epoch_traj[i][0].astype(float)
        y = epoch_traj[i][1].astype(float)
        z = epoch_traj[i][2].astype(float)

        if smooth_sigma and smooth_sigma > 0 and x.size >= 3:
            x = gaussian_filter1d(x, sigma=smooth_sigma)
            y = gaussian_filter1d(y, sigma=smooth_sigma)
            z = gaussian_filter1d(z, sigma=smooth_sigma)

        ax2d.plot(x, y, color=_epoch_condition_color_for(style_ctx, cond, ep), linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond), linewidth=2.1, alpha=0.9, label=label)
        _plot_mode3_highlight(
            ax2d,
            style_ctx,
            x,
            y,
            zt,
            condition=cond,
            epoch_id=ep,
            linewidth=2.1,
            alpha=0.9,
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
        )
        ax2d.scatter(x, y, s=10, color=_epoch_condition_color_for(style_ctx, cond, ep), marker=cond_marker.get(cond, "o"), alpha=0.30, linewidths=0)

        ax3d.plot(x, y, z, color=_epoch_condition_color_for(style_ctx, cond, ep), linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond), linewidth=2.0, alpha=0.9, label=label)
        _plot_mode3_highlight(
            ax3d,
            style_ctx,
            x,
            y,
            zt,
            z=z,
            condition=cond,
            epoch_id=ep,
            linewidth=2.0,
            alpha=0.9,
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
        )
        ax3d.scatter(x, y, z, s=8, color=_epoch_condition_color_for(style_ctx, cond, ep), marker=cond_marker.get(cond, "o"), alpha=0.25, linewidths=0)

        ax3t.plot(x, y, zt, color=_epoch_condition_color_for(style_ctx, cond, ep), linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond), linewidth=2.0, alpha=0.9, label=label)
        _plot_mode3_highlight(
            ax3t,
            style_ctx,
            x,
            y,
            zt,
            z=zt,
            condition=cond,
            epoch_id=ep,
            linewidth=2.0,
            alpha=0.9,
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
        )
        ax3t.scatter(x, y, zt, s=8, color=_epoch_condition_color_for(style_ctx, cond, ep), marker=cond_marker.get(cond, "o"), alpha=0.25, linewidths=0)

    ax2d.set_title("2D: One line per epoch (PC1-PC2)")
    ax2d.set_xlabel("PC 1")
    ax2d.set_ylabel("PC 2")

    ax3d.set_title("3D: One line per epoch (PC1-PC2-PC3)")
    ax3d.set_xlabel("PC 1")
    ax3d.set_ylabel("PC 2")
    ax3d.set_zlabel("PC 3")

    ax3t.set_title("3D: One line per epoch (PC1-PC2-Time)")
    ax3t.set_xlabel("PC 1")
    ax3t.set_ylabel("PC 2")
    ax3t.set_zlabel("Time from event (s)")

    handles, labels = ax3t.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax3t.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, fontsize=9)

    fig.suptitle(title_prefix + " | Group: One-Line-Per-Epoch Population Family", fontsize=16, y=0.98)
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="center", fontsize=10)
    units_text = f"units: {n_units}" if n_units is not None else "units: n/a"
    fig.text(0.5, 0.90 if subtitle else 0.93, units_text, ha="center", va="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.86, 0.9])

    save_path = Path(save_root) / "Grouped_EpochPopulation"
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_region_{_sanitize_name(brain_region)}_group_epoch_population.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

def _legend_unique(ax, *, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8):
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) == 0:
        return
    uniq = {}
    for h, l in zip(handles, labels):
        if l is None or str(l).strip() == "":
            continue
        if l not in uniq:
            uniq[l] = h
    if len(uniq) == 0:
        return
    ax.legend(
        uniq.values(),
        uniq.keys(),
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=False,
        fontsize=fontsize,
    )


def _pad_scores_to_min_pcs(arr, *, pc_axis=0, min_pcs=3, fill_value=0.0):
    """
    Ensure an array has at least `min_pcs` along `pc_axis` by padding constants.
    Returns (padded_array, original_pc_count).
    """
    a = np.asarray(arr, dtype=float)
    if a.ndim == 0:
        raise ValueError("Input array must have at least 1 dimension.")
    n_pc = int(a.shape[pc_axis])
    if n_pc >= int(min_pcs):
        return a, n_pc
    pad_shape = list(a.shape)
    pad_shape[pc_axis] = int(min_pcs) - n_pc
    pad = np.full(pad_shape, float(fill_value), dtype=float)
    return np.concatenate([a, pad], axis=pc_axis), n_pc


def _pc_axis_label(pc_idx_1based: int, *, n_pc_available: int) -> str:
    if pc_idx_1based <= int(n_pc_available):
        return f"PC {pc_idx_1based}"
    return f"PC {pc_idx_1based} (padded)"


def _plot_group_base_trial_on_axes(ax2d, ax3d, ax3t, Xp, event_meta, *, time_col="start_time", show_legend=True):
    Xp_use, n_pc_available = _pad_scores_to_min_pcs(Xp, pc_axis=0, min_pcs=3, fill_value=0.0)
    if time_col not in event_meta.columns:
        raise ValueError(f"{time_col} not in event_meta columns: {list(event_meta.columns)}")

    em = event_meta.copy().reset_index(drop=True)
    em[time_col] = pd.to_numeric(em[time_col], errors="coerce")
    em = em.dropna(subset=[time_col, "condition", "epoch_id"]).reset_index(drop=True)
    if len(em) != Xp_use.shape[1]:
        raise ValueError(f"event_meta rows ({len(em)}) must match Xp events ({Xp_use.shape[1]}).")

    t0 = em[time_col].min()
    em["time_rel_s"] = em[time_col] - t0
    style_ctx, cond_marker, cond_linestyle = _epoch_condition_color_marker_maps(em)

    for cond in em["condition"].astype(str).unique():
        em_cond = em[em["condition"].astype(str) == cond]
        marker = cond_marker.get(cond, "o")
        ls = _epoch_condition_linestyle_for(cond_linestyle, cond)
        for ep in sorted(em_cond["epoch_id"].astype(int).unique()):
            idx = em_cond.index[em_cond["epoch_id"].astype(int) == ep].to_numpy()
            label = f"{cond}, epoch {ep}"
            color = _epoch_condition_color_for(style_ctx, cond, ep)

            ax2d.scatter(
                Xp_use[0, idx],
                Xp_use[1, idx],
                s=45,
                alpha=0.9,
                marker=marker,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                label=label,
            )
            ax3d.scatter(
                Xp_use[0, idx],
                Xp_use[1, idx],
                Xp_use[2, idx],
                s=30,
                alpha=0.85,
                marker=marker,
                color=color,
                edgecolors="black",
                linewidths=0.4,
                label=label,
            )
            if idx.size >= 2:
                ord_idx = np.argsort(em.loc[idx, "time_rel_s"].to_numpy(dtype=float))
                idx2 = idx[ord_idx]
                ax3t.plot(
                    Xp_use[0, idx2],
                    Xp_use[1, idx2],
                    em.loc[idx2, "time_rel_s"].to_numpy(dtype=float),
                    color=color,
                    linestyle=ls,
                    linewidth=2.0,
                    marker=marker,
                    markersize=3.5,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    label=label,
                )
            else:
                ax3t.scatter(
                    Xp_use[0, idx],
                    Xp_use[1, idx],
                    em.loc[idx, "time_rel_s"].to_numpy(dtype=float),
                    s=30,
                    alpha=0.85,
                    marker=marker,
                    color=color,
                    edgecolors="black",
                    linewidths=0.4,
                    label=label,
                )

    ax2d.set_xlabel(_pc_axis_label(1, n_pc_available=n_pc_available))
    ax2d.set_ylabel(_pc_axis_label(2, n_pc_available=n_pc_available))
    ax3d.set_xlabel(_pc_axis_label(1, n_pc_available=n_pc_available))
    ax3d.set_ylabel(_pc_axis_label(2, n_pc_available=n_pc_available))
    ax3d.set_zlabel(_pc_axis_label(3, n_pc_available=n_pc_available))
    ax3t.set_xlabel(_pc_axis_label(1, n_pc_available=n_pc_available))
    ax3t.set_ylabel(_pc_axis_label(2, n_pc_available=n_pc_available))
    ax3t.set_zlabel("Time (s, rel)")
    if show_legend:
        _legend_unique(ax3t, fontsize=7)


def _plot_group_epochmean_on_axes(
    ax2d,
    ax3d,
    ax3t_pts,
    ax3t_line,
    Xp,
    event_meta,
    *,
    smooth_sigma=1.2,
    show_legend=True,
):
    Xp_use, n_pc_available = _pad_scores_to_min_pcs(Xp, pc_axis=0, min_pcs=3, fill_value=0.0)

    em_mean = _epoch_mean_pc_table(Xp_use, event_meta)
    style_ctx, cond_marker, cond_linestyle = _epoch_condition_color_marker_maps(event_meta)

    for _, rec in em_mean.iterrows():
        cond = str(rec["condition"])
        ep = int(rec["epoch_id"])
        label = f"{cond}, epoch {ep} (n={int(rec['n_trials'])})"
        color = _epoch_condition_color_for(style_ctx, cond, ep)

        ax2d.scatter(rec["pc1"], rec["pc2"], s=220, alpha=1.0, marker=cond_marker.get(cond, "o"), color=color, edgecolors="black", linewidths=0.9, label=label)
        ax3d.scatter(rec["pc1"], rec["pc2"], rec["pc3"], s=110, alpha=1.0, marker=cond_marker.get(cond, "o"), color=color, edgecolors="black", linewidths=0.7, label=label)
        ax3t_pts.scatter(rec["pc1"], rec["pc2"], rec["time_rel_s"], s=110, alpha=1.0, marker=cond_marker.get(cond, "o"), color=color, edgecolors="black", linewidths=0.7, label=label)

    for cond in em_mean["condition"].astype(str).unique():
        g = em_mean[em_mean["condition"].astype(str) == cond].sort_values("epoch_id")
        if g.empty:
            continue
        x = g["pc1"].to_numpy(dtype=float)
        y = g["pc2"].to_numpy(dtype=float)
        z = g["time_rel_s"].to_numpy(dtype=float)
        if smooth_sigma and smooth_sigma > 0 and len(x) >= 3:
            x = gaussian_filter1d(x, sigma=smooth_sigma)
            y = gaussian_filter1d(y, sigma=smooth_sigma)
            z = gaussian_filter1d(z, sigma=smooth_sigma)

        trend_color = _epoch_condition_color_for(style_ctx, cond, int(g["epoch_id"].iloc[-1]))
        ax3t_line.plot(
            x,
            y,
            z,
            color=trend_color,
            linestyle=_epoch_condition_linestyle_for(cond_linestyle, cond),
            linewidth=2.3,
            alpha=0.85,
            label=f"{cond} trend",
        )
        for ii, (_, rec) in enumerate(g.iterrows()):
            ep = int(rec["epoch_id"])
            ax3t_line.scatter(
                x[ii],
                y[ii],
                z[ii],
                s=90,
                alpha=1.0,
                marker=cond_marker.get(cond, "o"),
                color=_epoch_condition_color_for(style_ctx, cond, ep),
                edgecolors="black",
                linewidths=0.6,
                label=f"{cond}, epoch {ep} (n={int(rec['n_trials'])})",
            )

    ax2d.set_xlabel(_pc_axis_label(1, n_pc_available=n_pc_available))
    ax2d.set_ylabel(_pc_axis_label(2, n_pc_available=n_pc_available))
    ax3d.set_xlabel(_pc_axis_label(1, n_pc_available=n_pc_available))
    ax3d.set_ylabel(_pc_axis_label(2, n_pc_available=n_pc_available))
    ax3d.set_zlabel(_pc_axis_label(3, n_pc_available=n_pc_available))
    ax3t_pts.set_xlabel(_pc_axis_label(1, n_pc_available=n_pc_available))
    ax3t_pts.set_ylabel(_pc_axis_label(2, n_pc_available=n_pc_available))
    ax3t_pts.set_zlabel("Time (s, rel)")
    ax3t_line.set_xlabel(_pc_axis_label(1, n_pc_available=n_pc_available))
    ax3t_line.set_ylabel(_pc_axis_label(2, n_pc_available=n_pc_available))
    ax3t_line.set_zlabel("Time (s, rel)")
    if show_legend:
        _legend_unique(ax3t_line, fontsize=7)


def _plot_group_trial_time_on_axes(
    ax2d,
    ax3d,
    ax3t,
    trial_time_scores,
    event_meta,
    *,
    bin_time=None,
    smooth_sigma=1.2,
    show_legend=True,
):
    scores_use, n_pc_available = _pad_scores_to_min_pcs(trial_time_scores, pc_axis=2, min_pcs=3, fill_value=0.0)
    if scores_use.shape[0] != len(event_meta):
        raise ValueError("Mismatch between trial_time_scores and event_meta rows.")

    n_trials, n_bins, _ = scores_use.shape
    if bin_time is None:
        zt = np.arange(n_bins, dtype=float)
    else:
        zt = np.asarray(bin_time, dtype=float).ravel()
        if zt.size != n_bins:
            raise ValueError(f"bin_time length ({zt.size}) must equal n_bins ({n_bins}).")
    zt = zt.copy()

    em = event_meta.copy().reset_index(drop=True)
    style_ctx, _, cond_linestyle = _epoch_condition_color_marker_maps(em)
    seen = set()

    for i in range(n_trials):
        cond = str(em.loc[i, "condition"])
        ep = int(em.loc[i, "epoch_id"])
        x = scores_use[i, :, 0].astype(float)
        y = scores_use[i, :, 1].astype(float)
        z = scores_use[i, :, 2].astype(float)

        if smooth_sigma and smooth_sigma > 0:
            x = gaussian_filter1d(x, sigma=smooth_sigma)
            y = gaussian_filter1d(y, sigma=smooth_sigma)
            z = gaussian_filter1d(z, sigma=smooth_sigma)

        label = f"{cond}, epoch {ep}"
        label2 = label if label not in seen else None
        seen.add(label)
        color = _epoch_condition_color_for(style_ctx, cond, ep)
        ls = _epoch_condition_linestyle_for(cond_linestyle, cond)

        ax2d.plot(x, y, color=color, linestyle=ls, linewidth=1.5, alpha=0.65, label=label2)
        _plot_mode3_highlight(ax2d, style_ctx, x, y, zt, condition=cond, epoch_id=ep, linewidth=1.5, alpha=0.65, linestyle=ls)
        ax3d.plot(x, y, z, color=color, linestyle=ls, linewidth=1.4, alpha=0.65, label=label2)
        _plot_mode3_highlight(ax3d, style_ctx, x, y, zt, z=z, condition=cond, epoch_id=ep, linewidth=1.4, alpha=0.65, linestyle=ls)
        ax3t.plot(x, y, zt, color=color, linestyle=ls, linewidth=1.5, alpha=0.65, label=label2)
        _plot_mode3_highlight(ax3t, style_ctx, x, y, zt, z=zt, condition=cond, epoch_id=ep, linewidth=1.5, alpha=0.65, linestyle=ls)

    ax2d.set_xlabel(_pc_axis_label(1, n_pc_available=n_pc_available))
    ax2d.set_ylabel(_pc_axis_label(2, n_pc_available=n_pc_available))
    ax3d.set_xlabel(_pc_axis_label(1, n_pc_available=n_pc_available))
    ax3d.set_ylabel(_pc_axis_label(2, n_pc_available=n_pc_available))
    ax3d.set_zlabel(_pc_axis_label(3, n_pc_available=n_pc_available))
    ax3t.set_xlabel(_pc_axis_label(1, n_pc_available=n_pc_available))
    ax3t.set_ylabel(_pc_axis_label(2, n_pc_available=n_pc_available))
    ax3t.set_zlabel("Time from event (s)")
    if show_legend:
        _legend_unique(ax3t, fontsize=7)


def _plot_group_epoch_population_on_axes(
    ax2d,
    ax3d,
    ax3t,
    epoch_info_df,
    epoch_traj,
    *,
    bin_time,
    smooth_sigma=1.2,
    show_legend=True,
):
    if len(epoch_traj) == 0:
        raise ValueError("epoch_traj is empty.")
    n_pc_available = min(int(np.asarray(tr).shape[0]) for tr in epoch_traj)
    epoch_traj_use = [_pad_scores_to_min_pcs(tr, pc_axis=0, min_pcs=3, fill_value=0.0)[0] for tr in epoch_traj]

    bt = np.asarray(bin_time, dtype=float).ravel()
    n_bins = epoch_traj_use[0].shape[1]
    if bt.size != n_bins:
        raise ValueError(f"bin_time length ({bt.size}) must equal n_bins ({n_bins}).")
    zt = bt.copy()

    style_ctx, cond_marker, cond_linestyle = _epoch_condition_color_marker_maps(epoch_info_df)
    for i, rec in epoch_info_df.reset_index(drop=True).iterrows():
        cond = str(rec["condition"])
        ep = int(rec["epoch_id"])
        label = f"{cond}, epoch {ep} (n={int(rec['n_trials'])})"
        x = epoch_traj_use[i][0].astype(float)
        y = epoch_traj_use[i][1].astype(float)
        z = epoch_traj_use[i][2].astype(float)

        if smooth_sigma and smooth_sigma > 0 and x.size >= 3:
            x = gaussian_filter1d(x, sigma=smooth_sigma)
            y = gaussian_filter1d(y, sigma=smooth_sigma)
            z = gaussian_filter1d(z, sigma=smooth_sigma)

        color = _epoch_condition_color_for(style_ctx, cond, ep)
        ls = _epoch_condition_linestyle_for(cond_linestyle, cond)
        marker = cond_marker.get(cond, "o")

        ax2d.plot(x, y, color=color, linestyle=ls, linewidth=2.1, alpha=0.9, label=label)
        _plot_mode3_highlight(ax2d, style_ctx, x, y, zt, condition=cond, epoch_id=ep, linewidth=2.1, alpha=0.9, linestyle=ls)
        ax2d.scatter(x, y, s=10, color=color, marker=marker, alpha=0.30, linewidths=0)

        ax3d.plot(x, y, z, color=color, linestyle=ls, linewidth=2.0, alpha=0.9, label=label)
        _plot_mode3_highlight(ax3d, style_ctx, x, y, zt, z=z, condition=cond, epoch_id=ep, linewidth=2.0, alpha=0.9, linestyle=ls)
        ax3d.scatter(x, y, z, s=8, color=color, marker=marker, alpha=0.25, linewidths=0)

        ax3t.plot(x, y, zt, color=color, linestyle=ls, linewidth=2.0, alpha=0.9, label=label)
        _plot_mode3_highlight(ax3t, style_ctx, x, y, zt, z=zt, condition=cond, epoch_id=ep, linewidth=2.0, alpha=0.9, linestyle=ls)
        ax3t.scatter(x, y, zt, s=8, color=color, marker=marker, alpha=0.25, linewidths=0)

    ax2d.set_xlabel(_pc_axis_label(1, n_pc_available=n_pc_available))
    ax2d.set_ylabel(_pc_axis_label(2, n_pc_available=n_pc_available))
    ax3d.set_xlabel(_pc_axis_label(1, n_pc_available=n_pc_available))
    ax3d.set_ylabel(_pc_axis_label(2, n_pc_available=n_pc_available))
    ax3d.set_zlabel(_pc_axis_label(3, n_pc_available=n_pc_available))
    ax3t.set_xlabel(_pc_axis_label(1, n_pc_available=n_pc_available))
    ax3t.set_ylabel(_pc_axis_label(2, n_pc_available=n_pc_available))
    ax3t.set_zlabel("Time from event (s)")
    if show_legend:
        _legend_unique(ax3t, fontsize=7)


def _normalize_group_family_name(name: str) -> str:
    key = str(name).strip().lower()
    alias = {
        "base": "base_trial",
        "base_trial": "base_trial",
        "plot_epoch_condition_group_base_trial": "base_trial",
        "epoch_mean": "epochmean",
        "epochmean": "epochmean",
        "plot_epoch_condition_group_epochmean": "epochmean",
        "trial_time": "trial_time",
        "trialtime": "trial_time",
        "plot_epoch_condition_group_trial_time": "trial_time",
        "epoch_population": "epoch_population",
        "epochpopulation": "epoch_population",
        "plot_epoch_condition_group_epoch_population": "epoch_population",
    }
    out = alias.get(key, key)
    valid = {"base_trial", "epochmean", "trial_time", "epoch_population"}
    if out not in valid:
        raise ValueError(f"Unknown family '{name}'. Valid: {sorted(valid)}")
    return out


def _group_family_meta(family: str) -> tuple[int, list[str], str, str]:
    fam = _normalize_group_family_name(family)
    if fam == "epochmean":
        return (
            4,
            [
                "2D: Epoch mean PC1 vs PC2",
                "3D: Epoch mean PC1-PC2-PC3",
                "3D: Epoch mean PC1-PC2-Time points",
                "3D: Epoch mean PC1-PC2-Time lines",
            ],
            "Grouped_ByRegionGrid_EpochMean",
            "group_epochmean_by_region_grid",
        )
    if fam == "base_trial":
        return (
            3,
            ["2D: PC1 vs PC2", "3D: PC1-PC2-PC3", "3D: PC1-PC2-Time"],
            "Grouped_ByRegionGrid_BaseTrial",
            "group_base_trial_by_region_grid",
        )
    if fam == "trial_time":
        return (
            3,
            [
                "2D: Per-trial time-bin lines (PC1-PC2)",
                "3D: Per-trial time-bin lines (PC1-PC2-PC3)",
                "3D: Per-trial time-bin lines (PC1-PC2-Time)",
            ],
            "Grouped_ByRegionGrid_TrialTime",
            "group_trial_time_by_region_grid",
        )
    return (
        3,
        [
            "2D: One line per epoch (PC1-PC2)",
            "3D: One line per epoch (PC1-PC2-PC3)",
            "3D: One line per epoch (PC1-PC2-Time)",
        ],
        "Grouped_ByRegionGrid_EpochPopulation",
        "group_epoch_population_by_region_grid",
    )


def plot_epoch_condition_group_family_by_region(
    probe,
    merged_dic,
    event_meta,
    *,
    family="base_trial",
    roi_filter=None,
    kslabel_filter="both",
    bc_label_filter=None,
    include_conditions=None,
    n_components=12,
    max_tensor_gb=8.0,
    win_start_s=-1.0,
    win_end_s=1.0,
    bin_size_s=0.025,
    smooth_sigma=1.2,
    brain_regions: list[str] | None = None,
    title_prefix="",
    subtitle="",
    save_root: str | Path = "master/results",
    show_plots=False,
):
    """
    Build one large figure for a probe where each row is a brain region and each row contains
    the grouped PCA subplots for one family.
    """
    fam = _normalize_group_family_name(family)
    n_cols, col_titles, subfolder, file_tag = _group_family_meta(fam)

    if brain_regions is None:
        regions = _list_probe_brain_regions(
            merged_dic=merged_dic,
            probe=probe,
            roi_filter=roi_filter,
            kslabel_filter=kslabel_filter,
            bc_label_filter=bc_label_filter,
        )
    else:
        regions = [_normalize_brain_region_value(r) for r in brain_regions]

    if len(regions) == 0:
        raise ValueError(f"Probe {probe}: no brain regions to plot.")

    n_rows = len(regions)
    fig_w = 7.8 * n_cols
    fig_h = max(4.0 * n_rows, 7.5)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.25, hspace=0.42)

    skipped_regions = []
    for r, br in enumerate(regions):
        if n_cols == 4:
            axes = [
                fig.add_subplot(gs[r, 0]),
                fig.add_subplot(gs[r, 1], projection="3d"),
                fig.add_subplot(gs[r, 2], projection="3d"),
                fig.add_subplot(gs[r, 3], projection="3d"),
            ]
        else:
            axes = [
                fig.add_subplot(gs[r, 0]),
                fig.add_subplot(gs[r, 1], projection="3d"),
                fig.add_subplot(gs[r, 2], projection="3d"),
            ]

        try:
            em, trials, tbins, Xp, _ = run_epoch_condition_pca_for_probe(
                probe=probe,
                merged_dic=merged_dic,
                event_meta=event_meta,
                roi_filter=roi_filter,
                kslabel_filter=kslabel_filter,
                bc_label_filter=bc_label_filter,
                include_conditions=include_conditions,
                n_components=max(3, n_components),
                max_tensor_gb=max_tensor_gb,
                win_start_s=win_start_s,
                win_end_s=win_end_s,
                bin_size_s=bin_size_s,
                brain_region_filter=br,
            )
            n_units = int(trials.shape[1])
            n_events = int(len(em))

            if fam == "base_trial":
                _plot_group_base_trial_on_axes(axes[0], axes[1], axes[2], Xp, em, time_col="start_time", show_legend=True)
            elif fam == "epochmean":
                _plot_group_epochmean_on_axes(axes[0], axes[1], axes[2], axes[3], Xp, em, smooth_sigma=smooth_sigma, show_legend=True)
            elif fam == "trial_time":
                scores, _ = pca_trial_timebin_scores(trials, n_components=max(3, n_components))
                _plot_group_trial_time_on_axes(axes[0], axes[1], axes[2], scores, em, bin_time=tbins, smooth_sigma=smooth_sigma, show_legend=True)
            else:
                epoch_info, epoch_traj, _ = pca_epoch_population_timebin_trajectories(trials, em, n_components=max(3, n_components))
                _plot_group_epoch_population_on_axes(axes[0], axes[1], axes[2], epoch_info, epoch_traj, bin_time=tbins, smooth_sigma=smooth_sigma, show_legend=True)

            row_text = f"BR={br}\nunits={n_units}\nevents={n_events}"
            axes[0].text(
                -0.38,
                0.5,
                row_text,
                transform=axes[0].transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=8.5,
            )
        except Exception as e:
            skipped_regions.append((br, str(e)))
            for ax in axes:
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    f"BR={br}\nSkipped\n{e}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    wrap=True,
                )

        if r == 0:
            for c, t in enumerate(col_titles):
                axes[c].set_title(t, fontsize=11)

    base_title = title_prefix if str(title_prefix).strip() else f"Probe {probe}"
    family_title = {
        "base_trial": "Base Trial-Level Epoch Plots",
        "epochmean": "Epoch-Mean Family",
        "trial_time": "Per-Trial Time-Bin Family",
        "epoch_population": "One-Line-Per-Epoch Population Family",
    }[fam]
    fig.suptitle(f"{base_title} | Brain-Region Grid | {family_title}", fontsize=16, y=0.995)

    subtitle_text = subtitle if str(subtitle).strip() else probe_brain_region_label(
        merged_dic=merged_dic,
        probe=probe,
        roi_filter=roi_filter,
        kslabel_filter=kslabel_filter,
        bc_label_filter=bc_label_filter,
    )
    fig.text(0.5, 0.975, subtitle_text, ha="center", va="center", fontsize=10)

    plt.tight_layout(rect=[0.03, 0.02, 0.98, 0.965])

    save_path = Path(save_root) / subfolder
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"probe_{_sanitize_name(probe)}_{file_tag}.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

    return {
        "out_path": str(out),
        "family": fam,
        "probe": str(probe),
        "regions_requested": int(len(regions)),
        "regions_skipped": int(len(skipped_regions)),
        "skipped_regions": skipped_regions,
    }


def plot_epoch_condition_group_all_families_by_region(
    probe,
    merged_dic,
    event_meta,
    *,
    families: Iterable[str] | None = None,
    roi_filter=None,
    kslabel_filter="both",
    bc_label_filter=None,
    include_conditions=None,
    n_components=12,
    max_tensor_gb=8.0,
    win_start_s=-1.0,
    win_end_s=1.0,
    bin_size_s=0.025,
    smooth_sigma=1.2,
    brain_regions: list[str] | None = None,
    title_prefix="",
    subtitle="",
    save_root: str | Path = "master/results",
    show_plots=False,
):
    """
    Convenience wrapper that builds the 4 large by-region grouped figures for one probe.
    """
    fams = ["base_trial", "epochmean", "trial_time", "epoch_population"] if families is None else list(families)
    out = {}
    for fam in fams:
        out[_normalize_group_family_name(fam)] = plot_epoch_condition_group_family_by_region(
            probe=probe,
            merged_dic=merged_dic,
            event_meta=event_meta,
            family=fam,
            roi_filter=roi_filter,
            kslabel_filter=kslabel_filter,
            bc_label_filter=bc_label_filter,
            include_conditions=include_conditions,
            n_components=n_components,
            max_tensor_gb=max_tensor_gb,
            win_start_s=win_start_s,
            win_end_s=win_end_s,
            bin_size_s=bin_size_s,
            smooth_sigma=smooth_sigma,
            brain_regions=brain_regions,
            title_prefix=title_prefix,
            subtitle=subtitle,
            save_root=save_root,
            show_plots=show_plots,
        )
    return out

def _group_family_merge_meta(family: str) -> tuple[str, str]:
    fam = _normalize_group_family_name(family)
    mapping = {
        "base_trial": ("Grouped_BaseTrial", "group_base_trial"),
        "epochmean": ("Grouped_EpochMean", "group_epochmean"),
        "trial_time": ("Grouped_TrialTime", "group_trial_time"),
        "epoch_population": ("Grouped_EpochPopulation", "group_epoch_population"),
    }
    return mapping[fam]


def merge_group_family_pngs_for_probe(
    save_root: str | Path,
    probe,
    *,
    family="epoch_population",
    n_cols=3,
    out_subfolder=None,
    show_plots=False,
):
    """
    Merge existing per-brain-region grouped PNGs into one grid PNG for a probe/family.
    Keeps original individual PNGs unchanged.
    """
    src_folder_name, suffix = _group_family_merge_meta(family)
    ptag = _sanitize_name(probe)
    src_folder = Path(save_root) / src_folder_name
    pattern = f"probe_{ptag}_region_*_{suffix}.png"
    paths = sorted(src_folder.glob(pattern))
    if len(paths) == 0:
        raise ValueError(f"No files found to merge for probe={probe}, family={family} in {src_folder}")

    n = len(paths)
    n_cols = max(1, int(n_cols))
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2 * n_cols, 5.4 * n_rows))
    axes_arr = np.array(axes).reshape(-1)

    region_re = re.compile(rf"^probe_{re.escape(ptag)}_region_(.+)_{re.escape(suffix)}$")
    for ax, p in zip(axes_arr, paths):
        img = plt.imread(p)
        ax.imshow(img)
        ax.axis("off")
        stem = p.stem
        m = region_re.match(stem)
        region_label = m.group(1) if m else p.stem
        ax.set_title(f"BR={region_label}", fontsize=10)

    for ax in axes_arr[len(paths):]:
        ax.axis("off")

    fam = _normalize_group_family_name(family)
    fam_title = {
        "base_trial": "Grouped Base Trial",
        "epochmean": "Grouped Epoch Mean",
        "trial_time": "Grouped Trial Time",
        "epoch_population": "Grouped Epoch Population",
    }[fam]
    fig.suptitle(f"Probe {probe} | {fam_title} | merged brain-region panels", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.97])

    out_folder = Path(save_root) / (out_subfolder or f"{src_folder_name}_Merged")
    out_folder.mkdir(parents=True, exist_ok=True)
    out_path = out_folder / f"probe_{ptag}_{suffix}_merged.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"Merged PCA grid saved to {out_path}")
    return str(out_path)


def merge_group_family_pngs_for_probe_all(
    save_root: str | Path,
    probe,
    *,
    families: Iterable[str] | None = None,
    n_cols=3,
    show_plots=False,
):
    """
    Merge all 4 grouped family folders into probe-level merged PNGs.
    """
    fams = ["base_trial", "epochmean", "trial_time", "epoch_population"] if families is None else list(families)
    out = {}
    for fam in fams:
        key = _normalize_group_family_name(fam)
        out[key] = merge_group_family_pngs_for_probe(
            save_root=save_root,
            probe=probe,
            family=key,
            n_cols=n_cols,
            show_plots=show_plots,
        )
    return out

def plot_pca_scree(
    evr,
    title_prefix="",
    subtitle="",
    probe="unknown",
    brain_region="unknown",
    run_label="",
    cumulative=True,
    max_components=None,
    save_root: str | Path = "master/results",
    show_plots=False,
):
    """
    Plot and save a scree chart for any PCA run using explained variance ratios.

    Parameters
    ----------
    evr : array-like
        Explained variance ratio per component (e.g., returned as `evr` from PCA run helpers).
    cumulative : bool
        If True, overlays cumulative explained variance (%).
    max_components : int | None
        If set, only the first N components are plotted.
    run_label : str
        Optional suffix to distinguish multiple scree plots for the same probe/region.
    """
    evr_arr = np.asarray(evr, dtype=float).ravel()
    evr_arr = evr_arr[np.isfinite(evr_arr)]
    if evr_arr.size == 0:
        raise ValueError("evr is empty after removing non-finite values.")

    if max_components is not None:
        n_keep = int(max_components)
        if n_keep <= 0:
            raise ValueError("max_components must be >= 1 when provided.")
        evr_arr = evr_arr[:n_keep]

    pcs = np.arange(1, evr_arr.size + 1, dtype=int)
    evr_pct = evr_arr * 100.0
    cum_pct = np.cumsum(evr_arr) * 100.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(pcs, evr_pct, color="#4C72B0", alpha=0.85, label="Per-PC EVR (%)")
    ax.plot(pcs, evr_pct, color="#1F3A5F", marker="o", linewidth=1.6, markersize=4)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_xticks(pcs)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)

    if cumulative:
        ax2 = ax.twinx()
        ax2.plot(pcs, cum_pct, color="#DD8452", marker="s", linewidth=2.0, markersize=4, label="Cumulative EVR (%)")
        ax2.set_ylabel("Cumulative Explained Variance (%)")
        ax2.set_ylim(0, 102)

    main_title = "PCA Scree Plot" if title_prefix == "" else f"{title_prefix} | PCA Scree Plot"
    fig.suptitle(main_title, fontsize=14, y=0.98)
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=10)

    n_show = min(3, evr_arr.size)
    top_text = f"Top {n_show} cumulative EVR: {cum_pct[n_show - 1]:.2f}%"
    fig.text(0.5, 0.90 if subtitle else 0.93, top_text, ha="center", va="center", fontsize=10)

    handles, labels = ax.get_legend_handles_labels()
    if cumulative:
        h2, l2 = ax2.get_legend_handles_labels()
        handles = handles + h2
        labels = labels + l2
    if len(handles) > 0:
        ax.legend(handles, labels, frameon=False, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    save_path = Path(save_root)
    save_path.mkdir(parents=True, exist_ok=True)
    run_suffix = _sanitize_name(run_label).strip("_")
    fname = f"probe_{_sanitize_name(probe)}_region_{_sanitize_name(brain_region)}_pca_scree"
    if run_suffix:
        fname = f"{fname}_{run_suffix}"
    out = save_path / f"{fname}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA scree plot saved to {out}")

# === END GROUPED EPOCH PLOT SETS ===



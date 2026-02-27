from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
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


def pca_get_probe_units_df(merged_dic, probe, roi_filter=None, kslabel_filter="both"):
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


def probe_brain_region_label(merged_dic, probe, roi_filter=None, kslabel_filter="both", max_regions=6):
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
    epochs = sorted(event_meta["epoch_id"].dropna().astype(int).unique().tolist())
    pal = sns.color_palette("husl", max(3, len(epochs)))
    epoch_color = {ep: pal[i % len(pal)] for i, ep in enumerate(epochs)}
    cond_marker = {"baseline": "o", "stimulation": "^", "washout": "s"}

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
                    color=epoch_color[ep], edgecolors="black", linewidths=0.8,
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
    out = _ensure_dir(Path(save_root) / "PCA_by_Epoch" / "2D_scatter") / f"probe_{_sanitize_name(probe)}_epoch_condition_scatter_2D.png"
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
    epochs = sorted(event_meta["epoch_id"].dropna().astype(int).unique().tolist())
    pal = sns.color_palette("husl", max(3, len(epochs)))
    epoch_color = {ep: pal[i % len(pal)] for i, ep in enumerate(epochs)}
    cond_marker = {"baseline": "o", "stimulation": "^", "washout": "s"}

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
                color=epoch_color[ep], edgecolors="black", linewidths=0.6,
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
    out = _ensure_dir(Path(save_root) / "PCA_by_Epoch" / "3D_scatter") / f"probe_{_sanitize_name(probe)}_epoch_condition_scatter_3D.png"
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

    epochs = sorted(em["epoch_id"].astype(int).unique().tolist())
    pal = sns.color_palette("husl", max(3, len(epochs)))
    epoch_color = {ep: pal[i % len(pal)] for i, ep in enumerate(epochs)}
    cond_marker = {"baseline": "o", "stimulation": "^", "washout": "s"}
    cond_linestyle = {"baseline": "-", "stimulation": "--", "washout": ":"}

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    for cond in em["condition"].astype(str).unique():
        em_cond = em[em["condition"].astype(str) == cond]
        marker = cond_marker.get(cond, "o")
        ls = cond_linestyle.get(cond, "-")
        for ep in sorted(em_cond["epoch_id"].astype(int).unique()):
            idx = em_cond.index[em_cond["epoch_id"].astype(int) == ep].to_numpy()
            if idx.size < 2:
                continue
            order = np.argsort(em.loc[idx, "time_rel_s"].to_numpy(dtype=float))
            idx = idx[order]
            ax.plot(
                Xp[0, idx], Xp[1, idx], em.loc[idx, "time_rel_s"].to_numpy(dtype=float),
                color=epoch_color[ep], linestyle=ls, linewidth=2.2,
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
    out = _ensure_dir(Path(save_root) / "PCA_by_Epoch" / "3D_PC12_Time") / f"probe_{_sanitize_name(probe)}_epoch_condition_line_PC12_Time.png"
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
    epochs = sorted(event_meta["epoch_id"].dropna().astype(int).unique().tolist())
    pal = sns.color_palette("husl", max(3, len(epochs)))
    epoch_color = {ep: pal[i % len(pal)] for i, ep in enumerate(epochs)}
    cond_marker = {"baseline": "o", "stimulation": "^", "washout": "s"}

    projections = [(0, 1), (1, 2), (0, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, (i, j) in zip(axes, projections):
        for cond, ep, idx in groups:
            p1 = float(np.nanmean(Xp[i, idx]))
            p2 = float(np.nanmean(Xp[j, idx]))
            ax.scatter(
                [p1], [p2], s=220, alpha=1.0,
                marker=cond_marker.get(cond, "o"), color=epoch_color.get(ep, "gray"),
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
    out = _ensure_dir(Path(save_root) / "PCA_by_Epoch" / "2D_scatter_epoch_avg") / f"probe_{_sanitize_name(probe)}_epoch_avg_scatter_2D.png"
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
    epochs = sorted(event_meta["epoch_id"].dropna().astype(int).unique().tolist())
    pal = sns.color_palette("husl", max(3, len(epochs)))
    epoch_color = {ep: pal[i % len(pal)] for i, ep in enumerate(epochs)}
    cond_marker = {"baseline": "o", "stimulation": "^", "washout": "s"}

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    for cond, ep, idx in groups:
        x = float(np.nanmean(Xp[0, idx]))
        y = float(np.nanmean(Xp[1, idx]))
        z = float(np.nanmean(Xp[2, idx]))
        ax.scatter(
            [x], [y], [z], s=160, alpha=1.0,
            marker=cond_marker.get(cond, "o"), color=epoch_color.get(ep, "gray"),
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
    out = _ensure_dir(Path(save_root) / "PCA_by_Epoch" / "3D_scatter_epoch_avg") / f"probe_{_sanitize_name(probe)}_epoch_avg_scatter_3D.png"
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

    epochs = sorted(event_meta["epoch_id"].dropna().astype(int).unique().tolist())
    pal = sns.color_palette("husl", max(3, len(epochs)))
    epoch_color = {ep: pal[i % len(pal)] for i, ep in enumerate(epochs)}
    cond_marker = {"baseline": "o", "stimulation": "^", "washout": "s"}
    cond_linestyle = {"baseline": "-", "stimulation": "--", "washout": ":"}

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
            color=epoch_color.get(ep, "gray"),
            linestyle=cond_linestyle.get(cond, "-"),
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
    out = _ensure_dir(Path(save_root) / "PCA_by_Epoch" / "3D_PC12_Time_epoch_avg") / f"probe_{_sanitize_name(probe)}_epoch_avg_line_PC12_Time.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close()
        print(f"PCA plot saved to {out}")

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from pynwb import NWBHDF5IO
except Exception:  # pragma: no cover - optional dependency at import time
    NWBHDF5IO = None


class NWBLoader:
    """Light wrapper around NWB I/O used in the notebook workflow."""

    def __init__(self, nwb_path: str | Path):
        if NWBHDF5IO is None:
            raise ImportError("pynwb is required to load NWB files.")
        self.nwb_path = str(nwb_path)
        self.io = None
        self.nwb = None
        self.load_nwb()

    def load_nwb(self):
        self.io = NWBHDF5IO(self.nwb_path, "r", load_namespaces=True)
        self.nwb = self.io.read()
        return self.nwb

    def trials(self) -> pd.DataFrame:
        return self.nwb.trials.to_dataframe() if self.nwb.trials is not None else pd.DataFrame()

    def units(self) -> pd.DataFrame:
        return self.nwb.units.to_dataframe() if self.nwb.units is not None else pd.DataFrame()

    def optogenetics_states(self) -> pd.DataFrame:
        if "optogenetics_states" in self.nwb.intervals:
            return self.nwb.intervals["optogenetics_states"].to_dataframe()
        return pd.DataFrame()

    def close(self):
        try:
            if self.io is not None:
                self.io.close()
        except Exception:
            pass


def resolve_nwb_path(p: str | Path) -> Path:
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"NWB path not found: {p}")
    if p.is_file():
        return p
    nwb_files = sorted(list(p.rglob("*.nwb")))
    if len(nwb_files) == 1:
        return nwb_files[0]
    if len(nwb_files) > 1:
        raise ValueError("Multiple NWB files found. Point NWB_PATH to one file.")
    files = [x for x in p.iterdir() if x.is_file()]
    if len(files) == 1:
        return files[0]
    raise ValueError("Could not resolve a unique NWB file path.")


def load_nwb_tables(nwb_path: str | Path) -> dict[str, pd.DataFrame]:
    """Load common NWB tables used by the PCA workflow."""
    resolved = resolve_nwb_path(nwb_path)
    loader = NWBLoader(resolved)
    try:
        return {
            "df_trials": loader.trials().reset_index(drop=True),
            "df_units": loader.units().reset_index(drop=True),
            "df_opto_states": loader.optogenetics_states().reset_index(drop=True),
        }
    finally:
        loader.close()


def choose_event_time_col(df: pd.DataFrame) -> str:
    for c in ["start_time", "time", "event_time", "timestamps", "event_time_s"]:
        if c in df.columns:
            return c
    raise ValueError(f"No event-time column found. Columns: {list(df.columns)}")


def normalize_kslabel(v: Any):
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


def infer_is_opto(df_trials: pd.DataFrame) -> pd.Series:
    if "optogenetics_LED_state" in df_trials.columns:
        c = df_trials["optogenetics_LED_state"]
        if pd.api.types.is_numeric_dtype(c):
            return pd.to_numeric(c, errors="coerce").fillna(0) > 0
        return c.astype(str).str.lower().isin(["1", "true", "on", "high"])
    if "stimulus" in df_trials.columns:
        s = df_trials["stimulus"].astype(str).str.lower()
        return s.str.contains("opto|laser|led|stim", regex=True)
    return pd.Series([False] * len(df_trials), index=df_trials.index)


def build_block_labels(is_opto: pd.Series) -> pd.DataFrame:
    is_opto = is_opto.astype(bool).reset_index(drop=True)
    block_id = (is_opto != is_opto.shift(1, fill_value=is_opto.iloc[0])).cumsum()

    mapping: dict[int, str] = {}
    seen_opto = False
    opto_k = 1
    wash_k = 1
    for b in block_id.unique():
        state = bool(is_opto[block_id == b].iloc[0])
        if state:
            mapping[b] = f"opto_epoch_{opto_k}"
            opto_k += 1
            seen_opto = True
        else:
            mapping[b] = "baseline" if not seen_opto else f"washout_epoch_{wash_k}"
            if seen_opto:
                wash_k += 1

    return pd.DataFrame({"block_id": block_id, "is_opto": is_opto, "block_label": block_id.map(mapping)})


def find_probe_col(df: pd.DataFrame):
    for c in ["probe", "probe_name", "probe_id", "probe_letter", "electrode_group"]:
        if c in df.columns:
            return c
    return None


def find_kslabel_col(df: pd.DataFrame):
    for c in ["KSlabel", "KSLabel", "kslabel", "ks_label", "label", "quality"]:
        if c in df.columns:
            return c
    return None


def pick_region_col(df: pd.DataFrame):
    for c in ["brain_region", "location", "region", "acronym", "structure", "ccf_acronym"]:
        if c in df.columns:
            return c
    return None


def build_units_probe_dict(df_units: pd.DataFrame, probe_col: str | None = None) -> dict[str, pd.DataFrame]:
    if probe_col is None:
        probe_col = find_probe_col(df_units)
    if probe_col is None:
        raise ValueError(f"No probe column found in df_units. Columns: {list(df_units.columns)}")

    out: dict[str, pd.DataFrame] = {}
    probes = df_units[probe_col].dropna().astype(str).unique().tolist()
    for probe in sorted(probes):
        out[str(probe)] = df_units[df_units[probe_col].astype(str) == str(probe)].copy().reset_index(drop=True)
    return out


def merge_units_with_metrics(
    df_units_dic: dict[str, pd.DataFrame],
    qm_dic: dict[str, pd.DataFrame] | None = None,
    cluster_dic: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Merge probe-unit tables with optional quality-metrics and cluster tables.
    """
    merged_dic: dict[str, pd.DataFrame] = {}
    for probe, u0 in df_units_dic.items():
        u = u0.copy().reset_index(drop=True)
        if "cluster_id" in u.columns:
            u["cluster_id"] = pd.to_numeric(u["cluster_id"], errors="coerce")

        if qm_dic is None and cluster_dic is None:
            merged_dic[probe] = u
            continue

        m = u.copy()

        if qm_dic is not None and probe in qm_dic:
            qm = qm_dic[probe].copy().reset_index(drop=True)
            if "cluster_id" not in qm.columns and "cluster_id" in m.columns:
                qm["cluster_id"] = m["cluster_id"].values
            if "cluster_id" in qm.columns:
                qm["cluster_id"] = pd.to_numeric(qm["cluster_id"], errors="coerce")
                keep_qm = [c for c in ["cluster_id", "nSpikes", "maxDriftEstimate", "maxChannels"] if c in qm.columns]
                if len(keep_qm) > 1 and "cluster_id" in m.columns:
                    m = m.merge(qm[keep_qm], on="cluster_id", how="left")

        if cluster_dic is not None and probe in cluster_dic:
            cl = cluster_dic[probe].copy().reset_index(drop=True)
            if "cluster_id" not in cl.columns and "cluster_id" in m.columns:
                cl["cluster_id"] = m["cluster_id"].values
            if "cluster_id" in cl.columns:
                cl["cluster_id"] = pd.to_numeric(cl["cluster_id"], errors="coerce")
                keep_cl = [c for c in ["cluster_id", "bc_classificationReason", "bc_ROI", "Brain_Region"] if c in cl.columns]
                if len(keep_cl) > 1 and "cluster_id" in m.columns:
                    m = m.merge(cl[keep_cl], on="cluster_id", how="left")
                    m = m.rename(
                        columns={
                            "bc_classificationReason": "bc_label",
                            "bc_ROI": "in_brainRegion",
                            "Brain_Region": "brain_region",
                        }
                    )

        merged_dic[str(probe)] = m.reset_index(drop=True)
    return merged_dic


def build_stim_df(df_trials: pd.DataFrame, event_time_col: str | None = None) -> pd.DataFrame:
    if event_time_col is None:
        event_time_col = choose_event_time_col(df_trials)
    stim_df = df_trials.copy()
    stim_df["event_time_s"] = pd.to_numeric(stim_df[event_time_col], errors="coerce")
    stim_df = stim_df.dropna(subset=["event_time_s"]).sort_values("event_time_s").reset_index(drop=True)
    blk = build_block_labels(infer_is_opto(stim_df))
    stim_df = pd.concat([stim_df, blk], axis=1)
    stim_df["trial_index"] = np.arange(len(stim_df), dtype=int)
    if "stimulus" in stim_df.columns:
        stim_df["label"] = stim_df["stimulus"].astype(str)
    else:
        stim_df["label"] = stim_df["block_label"].astype(str)
    return stim_df


def pca_select_events(
    stim_df: pd.DataFrame,
    event_time_col: str = "timestamp",
    event_label_col: str | None = None,
    event_filter_col: str | None = None,
    event_filter_values: list[str] | None = None,
    exclude_event_names: list[str] | str | None = None,
    max_events: int | None = None,
    subsample_mode: str = "uniform",
):
    if event_time_col not in stim_df.columns:
        for alt in ["timestamp", "event_time_s", "start_time", "time"]:
            if alt in stim_df.columns:
                event_time_col = alt
                break
    if event_time_col not in stim_df.columns:
        raise ValueError(f"{event_time_col} not in stim_df columns: {list(stim_df.columns)}")

    out = stim_df.copy()
    out[event_time_col] = pd.to_numeric(out[event_time_col], errors="coerce")
    out = out.dropna(subset=[event_time_col]).sort_values(event_time_col).reset_index(drop=True)

    if exclude_event_names is not None:
        if isinstance(exclude_event_names, str):
            exclude_event_names = [exclude_event_names]
        exclude_norm = {str(x).strip().lower() for x in exclude_event_names}
    else:
        exclude_norm = set()

    if "stimulus" in out.columns and exclude_norm:
        stim_norm = out["stimulus"].astype(str).str.strip().str.lower()
        out = out[~stim_norm.isin(exclude_norm)].reset_index(drop=True)

    if event_filter_col is not None and event_filter_col in out.columns and exclude_norm:
        col_norm = out[event_filter_col].astype(str).str.strip().str.lower()
        out = out[~col_norm.isin(exclude_norm)].reset_index(drop=True)

    if event_filter_values is not None:
        keep_norm = {str(v).strip().lower() for v in event_filter_values}
        if "stimulus" in out.columns:
            stim_norm = out["stimulus"].astype(str).str.strip().str.lower()
            out = out[stim_norm.isin(keep_norm)].reset_index(drop=True)
        elif event_filter_col is not None and event_filter_col in out.columns:
            col_norm = out[event_filter_col].astype(str).str.strip().str.lower()
            out = out[col_norm.isin(keep_norm)].reset_index(drop=True)

    n_before = len(out)
    if max_events is not None and n_before > max_events:
        if subsample_mode == "first":
            idx = np.arange(max_events)
        else:
            idx = np.linspace(0, n_before - 1, max_events, dtype=int)
        out = out.iloc[idx].reset_index(drop=True)

    if len(out) == 0:
        raise ValueError("No events remain after filtering.")

    if event_label_col is not None and event_label_col in out.columns:
        labels = out[event_label_col].astype(str).to_numpy()
    elif "stimulus" in out.columns:
        labels = out["stimulus"].astype(str).to_numpy()
    else:
        labels = np.array(["all_events"] * len(out), dtype=object)

    return out, out[event_time_col].to_numpy(dtype=float), labels


def _flatten_idx(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        return [int(x)]
    out = []
    for item in x:
        if isinstance(item, (list, tuple, np.ndarray)):
            out.extend(_flatten_idx(item))
        else:
            out.append(int(item))
    return out


def _normalize_trial_indices(idx_nested, n_trials: int):
    if isinstance(idx_nested, np.ndarray):
        idx_nested = idx_nested.tolist()
    if not isinstance(idx_nested, (list, tuple)):
        idx_nested = [idx_nested]

    epochs = []
    for ep in idx_nested:
        if isinstance(ep, (list, tuple, np.ndarray)):
            epochs.append([int(v) for v in _flatten_idx(ep)])
        else:
            epochs.append([int(ep)])

    all_vals = [v for ep in epochs for v in ep]
    if len(all_vals) == 0:
        return [[] for _ in epochs]

    max_v = max(all_vals)
    min_v = min(all_vals)
    one_based = (max_v <= n_trials) and (min_v >= 1)

    norm = []
    for ep in epochs:
        ep0 = [v - 1 for v in ep] if one_based else [v for v in ep]
        ep0 = [v for v in ep0 if 0 <= v < n_trials]
        norm.append(sorted(list(set(ep0))))
    return norm


def build_epoch_event_meta(
    all_trial_start_times: np.ndarray | list[float],
    baseline_trials_idx,
    optoicalStim_trials_idx,
    washout_trials_idx,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    all_trial_start_times = np.asarray(all_trial_start_times, dtype=float)
    n_trials_total = len(all_trial_start_times)

    baseline_idx_epochs = _normalize_trial_indices(baseline_trials_idx, n_trials_total)
    stim_idx_epochs = _normalize_trial_indices(optoicalStim_trials_idx, n_trials_total)
    wash_idx_epochs = _normalize_trial_indices(washout_trials_idx, n_trials_total)

    rows = []
    for _, ep_idx in enumerate(baseline_idx_epochs, start=1):
        for tidx in ep_idx:
            rows.append(
                {
                    "trial_index0": int(tidx),
                    "trial_number": int(tidx + 1),
                    "start_time": float(all_trial_start_times[tidx]),
                    "condition": "baseline",
                    "epoch_id": int(0),
                    "condition_epoch": "baseline_epoch",
                }
            )

    for ep_i, ep_idx in enumerate(stim_idx_epochs, start=1):
        for tidx in ep_idx:
            rows.append(
                {
                    "trial_index0": int(tidx),
                    "trial_number": int(tidx + 1),
                    "start_time": float(all_trial_start_times[tidx]),
                    "condition": "stimulation",
                    "epoch_id": int(ep_i),
                    "condition_epoch": f"stimulation_epoch_{ep_i}",
                }
            )

    for ep_i, ep_idx in enumerate(wash_idx_epochs, start=1):
        for tidx in ep_idx:
            rows.append(
                {
                    "trial_index0": int(tidx),
                    "trial_number": int(tidx + 1),
                    "start_time": float(all_trial_start_times[tidx]),
                    "condition": "washout",
                    "epoch_id": int(ep_i),
                    "condition_epoch": f"washout_epoch_{ep_i}",
                }
            )

    pca_event_meta = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["trial_index0"])
        .sort_values("trial_index0")
        .reset_index(drop=True)
    )
    stimulation_trials_start_times = pca_event_meta.loc[pca_event_meta["condition"] == "stimulation", "start_time"].to_numpy(dtype=float)
    washout_trials_start_times = pca_event_meta.loc[pca_event_meta["condition"] == "washout", "start_time"].to_numpy(dtype=float)
    return pca_event_meta, stimulation_trials_start_times, washout_trials_start_times


def save_processed_bundle(
    out_dir: str | Path,
    merged_dic: dict[str, pd.DataFrame],
    stim_df: pd.DataFrame,
    pca_event_meta: pd.DataFrame,
    extras: dict[str, Any] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "merged_dic.pkl", "wb") as f:
        pickle.dump(merged_dic, f, protocol=pickle.HIGHEST_PROTOCOL)

    stim_df.to_pickle(out / "stim_df.pkl")
    pca_event_meta.to_pickle(out / "pca_event_meta.pkl")

    if extras is None:
        extras = {}
    with open(out / "extras.pkl", "wb") as f:
        pickle.dump(extras, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "created_at": datetime.now().isoformat(),
        "files": ["merged_dic.pkl", "stim_df.pkl", "pca_event_meta.pkl", "extras.pkl"],
        "n_probes": len(merged_dic),
        "n_events_stim_df": int(len(stim_df)),
        "n_events_pca_meta": int(len(pca_event_meta)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def load_processed_bundle(out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    if not out.exists():
        raise FileNotFoundError(f"Processed bundle directory not found: {out}")

    with open(out / "merged_dic.pkl", "rb") as f:
        merged_dic = pickle.load(f)
    stim_df = pd.read_pickle(out / "stim_df.pkl")
    pca_event_meta = pd.read_pickle(out / "pca_event_meta.pkl")

    extras = {}
    extras_path = out / "extras.pkl"
    if extras_path.exists():
        with open(extras_path, "rb") as f:
            extras = pickle.load(f)

    meta = {}
    meta_path = out / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return {
        "merged_dic": merged_dic,
        "stim_df": stim_df,
        "pca_event_meta": pca_event_meta,
        "extras": extras,
        "meta": meta,
        "bundle_dir": out,
    }


def build_and_save_processed_bundle(
    out_dir: str | Path,
    df_units: pd.DataFrame,
    df_trials: pd.DataFrame,
    qm_dic: dict[str, pd.DataFrame] | None = None,
    cluster_dic: dict[str, pd.DataFrame] | None = None,
    all_trial_start_times: np.ndarray | list[float] | None = None,
    baseline_trials_idx=None,
    optoicalStim_trials_idx=None,
    washout_trials_idx=None,
    extras: dict[str, Any] | None = None,
) -> Path:
    df_units_dic = build_units_probe_dict(df_units)
    merged_dic = merge_units_with_metrics(df_units_dic, qm_dic=qm_dic, cluster_dic=cluster_dic)
    stim_df = build_stim_df(df_trials)

    if all_trial_start_times is not None and baseline_trials_idx is not None and optoicalStim_trials_idx is not None and washout_trials_idx is not None:
        pca_event_meta, stimulation_trials_start_times, washout_trials_start_times = build_epoch_event_meta(
            all_trial_start_times=all_trial_start_times,
            baseline_trials_idx=baseline_trials_idx,
            optoicalStim_trials_idx=optoicalStim_trials_idx,
            washout_trials_idx=washout_trials_idx,
        )
        if extras is None:
            extras = {}
        extras = dict(extras)
        extras["stimulation_trials_start_times"] = stimulation_trials_start_times
        extras["washout_trials_start_times"] = washout_trials_start_times
    else:
        pca_event_meta = pd.DataFrame(columns=["trial_index0", "trial_number", "start_time", "condition", "epoch_id", "condition_epoch"])

    return save_processed_bundle(
        out_dir=out_dir,
        merged_dic=merged_dic,
        stim_df=stim_df,
        pca_event_meta=pca_event_meta,
        extras=extras,
    )

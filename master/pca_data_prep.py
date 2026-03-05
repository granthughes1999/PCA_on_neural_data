from __future__ import annotations

import json
import pickle
import re
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


def normalize_probe_value(v: Any) -> str | None:
    """
    Normalize probe identifiers to a stable key.
    Examples:
      "A" -> "A"
      "probeA" -> "A"
      "kilosort4_B" -> "B"
    """
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s == "":
        return None
    s_up = s.upper()
    # Direct single-letter probe labels.
    if re.fullmatch(r"[A-Z]", s_up):
        return s_up

    # Common explicit patterns.
    patterns = [
        r"KILOSORT\s*4[_\-\s]*([A-Z])(?:$|[_\-\s])",
        r"PROBE[_\-\s]*([A-Z])(?:$|[_\-\s])",
        r"^([A-Z])[_\-\s]*PROBE$",
        r"PROBE([A-Z])$",
    ]
    for pat in patterns:
        m = re.search(pat, s_up)
        if m:
            return m.group(1)

    # Token fallback: if any separator-delimited token is a single letter.
    for tok in re.split(r"[_\-\s]+", s_up):
        if re.fullmatch(r"[A-Z]", tok):
            return tok

    # No unambiguous probe label found.
    return None


def extract_probe_letters(df_units: pd.DataFrame, probe_col: str | None = None) -> list[str]:
    if probe_col is None:
        probe_col = find_probe_col(df_units)
    if probe_col is None:
        return []
    vals = df_units[probe_col].map(normalize_probe_value).dropna().astype(str)
    vals = vals[vals.str.len() > 0]
    return sorted(vals.unique().tolist())


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
    probe_key = df_units[probe_col].map(normalize_probe_value)
    probes = probe_key.dropna().astype(str).unique().tolist()
    for probe in sorted(probes):
        out[str(probe)] = df_units[probe_key.astype(str) == str(probe)].copy().reset_index(drop=True)
    return out

def load_or_build_processed_bundle(
    *,
    processed_bundle_dir: str | Path | None = None,
    nwb_path_for_auto_build: str | Path = "",
    bombcell_root_for_auto_build: str | Path = "",
    use_bombcell_if_available: bool = True,
    auto_build_bundle_if_missing: bool = True,
    auto_rebuild_if_bombcell_missing: bool = True,
    required_filenames: tuple[str, ...] = ("merged_dic.pkl", "stim_df.pkl", "pca_event_meta.pkl"),
    verbose: bool = True,
) -> tuple[
    dict[str, Any],  # bundle
    dict[str, pd.DataFrame],  # merged_dic
    pd.DataFrame,  # stim_df
    pd.DataFrame,  # df_stim (alias)
    pd.DataFrame,  # pca_event_meta
    dict[str, Any],  # extras
    dict[str, Any],  # meta
    Path,  # processed_bundle_dir
]:
    """
    Load an existing processed bundle, or build it from NWB if missing / rebuild required.

    Notes
    -----
    - Bundle files are expected to match save_processed_bundle/load_processed_bundle:
      merged_dic.pkl, stim_df.pkl, pca_event_meta.pkl (and optionally extras.pkl, meta.json).
    - If auto_rebuild_if_bombcell_missing=True, will rebuild when merged_dic lacks:
      in_brainRegion and brain_region, but only if a Bombcell root is provided.
    """

    if processed_bundle_dir is None:
        processed_bundle_dir = Path("processed_data") / "bundle_latest"
    processed_bundle_dir = Path(processed_bundle_dir)

    bombcell_root_str = str(bombcell_root_for_auto_build).strip()
    nwb_path_str = str(nwb_path_for_auto_build).strip()

    if use_bombcell_if_available and bombcell_root_str == "" and verbose:
        print(
            "Bombcell root not set. Set bombcell_root_for_auto_build to merge "
            "in_brainRegion/brain_region/bc_label."
        )

    required_files = [processed_bundle_dir / fn for fn in required_filenames]
    bundle_exists = all(f.exists() for f in required_files)
    rebuild_required = (not bundle_exists)

    # If bundle exists but lacks Bombcell columns, optionally rebuild
    if bundle_exists and auto_rebuild_if_bombcell_missing:
        try:
            _bundle_tmp = load_processed_bundle(processed_bundle_dir)
            _md = _bundle_tmp["merged_dic"]
            _probe0 = sorted(list(_md.keys()))[0]
            _cols0 = set(_md[_probe0].columns)

            _has_bombcell_cols = ("in_brainRegion" in _cols0) and ("brain_region" in _cols0)
            if (not _has_bombcell_cols) and use_bombcell_if_available and bombcell_root_str != "":
                if verbose:
                    print("Existing bundle missing Bombcell columns. Rebuild requested.")
                rebuild_required = True
        except Exception as e:
            if verbose:
                print("Could not inspect existing bundle; rebuild requested:", e)
            rebuild_required = True

    # Build if required
    if rebuild_required and auto_build_bundle_if_missing:
        if nwb_path_str == "":
            raise ValueError(
                "Processed bundle is missing/rebuild requested and nwb_path_for_auto_build is empty. "
                "Set nwb_path_for_auto_build to your recording .nwb path (or parent folder) and re-run."
            )

        if verbose:
            print("Building processed bundle from NWB:", nwb_path_for_auto_build)

        tables = load_nwb_tables(nwb_path_for_auto_build)
        df_units = tables["df_units"]
        df_trials = tables["df_trials"]

        if df_units.empty:
            raise ValueError("NWB units table is empty; cannot build processed bundle.")
        if df_trials.empty:
            raise ValueError("NWB trials table is empty; cannot build processed bundle.")

        qm_dic = None
        cluster_dic = None
        bombcell_report = None

        # Optional Bombcell merge
        if use_bombcell_if_available and bombcell_root_str != "":
            probe_col = find_probe_col(df_units)
            if probe_col is None:
                if verbose:
                    print("Bombcell merge skipped: no probe column found in df_units.")
            else:
                probes = extract_probe_letters(df_units, probe_col=probe_col)
                if len(probes) == 0:
                    if verbose:
                        print("Bombcell merge skipped: could not infer probe labels from df_units probe column.")
                else:
                    try:
                        qm_dic, cluster_dic, bombcell_report = load_bombcell_metrics(
                            bombcell_root=bombcell_root_for_auto_build,
                            probes=probes,
                        )
                        if verbose:
                            print("Bombcell loaded qmetrics probes:", sorted(list(qm_dic.keys())) if qm_dic else [])
                            print("Bombcell loaded cluster probes:", sorted(list(cluster_dic.keys())) if cluster_dic else [])
                            if bombcell_report is not None and len(bombcell_report.get("missing", [])) > 0:
                                print("Bombcell missing entries:", bombcell_report["missing"][:10])
                    except Exception as e:
                        if verbose:
                            print("Bombcell load failed; continuing without Bombcell merge:", e)
                        qm_dic = None
                        cluster_dic = None
                        bombcell_report = {"error": str(e)}

        build_and_save_processed_bundle(
            out_dir=processed_bundle_dir,
            df_units=df_units,
            df_trials=df_trials,
            qm_dic=qm_dic,
            cluster_dic=cluster_dic,
            all_trial_start_times=None,
            baseline_trials_idx=None,
            optoicalStim_trials_idx=None,
            washout_trials_idx=None,
            extras={
                "auto_built_from_nwb": str(nwb_path_for_auto_build),
                "bombcell_root": bombcell_root_str if bombcell_root_str != "" else None,
                "bombcell_report": bombcell_report,
            },
        )

        if verbose:
            print("Built processed bundle:", processed_bundle_dir)

    # Load bundle
    bundle = load_processed_bundle(processed_bundle_dir)
    merged_dic = bundle["merged_dic"]
    stim_df = bundle["stim_df"]
    df_stim = stim_df  # alias
    pca_event_meta = bundle["pca_event_meta"]
    extras = bundle.get("extras", {})
    meta = bundle.get("meta", {})

    if verbose:
        print("Loaded bundle:", processed_bundle_dir)
        print("Meta:", meta)
        print("pca_event_meta rows:", len(pca_event_meta))

    return bundle, merged_dic, stim_df, df_stim, pca_event_meta, extras, meta, processed_bundle_dir

def load_bombcell_metrics(
    bombcell_root: str | Path,
    probes: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, list[str]]]:
    """
    Load Bombcell quality_metrics.csv and cluster classification TSV files by probe.

    Expected layout under bombcell_root:
      - .../kilosort4_A/quality_metrics.csv
      - .../kilosort4_A/cluster_bc_classificationReason.tsv
      (same pattern for B/C/...)
    """
    root = Path(bombcell_root)
    if not root.exists():
        raise FileNotFoundError(f"Bombcell root not found: {root}")

    if probes is None:
        probes = [chr(x) for x in range(ord("A"), ord("F") + 1)]
    probes = [str(p).strip().upper() for p in probes]

    qm_dic: dict[str, pd.DataFrame] = {}
    cluster_dic: dict[str, pd.DataFrame] = {}
    report: dict[str, list[str]] = {"loaded_qm": [], "loaded_cluster": [], "missing": []}

    # Pre-index potential probe dirs once
    ks_probe_dirs = [d for d in root.rglob("*") if d.is_dir() and d.name.lower().startswith("kilosort4_")]
    dir_map = {}
    for d in ks_probe_dirs:
        suffix = d.name.lower().replace("kilosort4_", "").strip()
        if len(suffix) > 0:
            dir_map[suffix[0].upper()] = d

    for probe in probes:
        pdir = dir_map.get(probe)
        if pdir is None:
            report["missing"].append(f"{probe}: probe folder (kilosort4_{probe})")
            continue

        # quality metrics (support common case variants)
        qm_candidates = [
            pdir / "bombcell" / f"probe_{probe}_quality_metrics.csv",
            pdir / "bombcell" / f"Probe_{probe}_quality_metrics.csv",
        ]
        qm_path = next((x for x in qm_candidates if x.exists()), None)
        if qm_path is not None:
            qm = pd.read_csv(qm_path)
            qm_dic[probe] = qm
            report["loaded_qm"].append(probe)
        else:
            report["missing"].append(f"{probe}: probe_{probe}_quality_metrics.csv")

        # cluster classification TSV (support a few filename variants)
        cluster_candidates = [
            pdir / "cluster_bc_classificationReason.tsv",
            pdir / "cluster_bc_classificationreason.tsv",
            pdir / "cluster_bc_classification_reason.tsv",
        ]
        cpath = next((x for x in cluster_candidates if x.exists()), None)
        if cpath is not None:
            cl = pd.read_csv(cpath, sep="\t")
            cluster_dic[probe] = cl
            report["loaded_cluster"].append(probe)
        else:
            report["missing"].append(f"{probe}: cluster_bc_classificationReason.tsv")

    return qm_dic, cluster_dic, report


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

def build_pca_event_meta_and_event_times(
    stim_df,
    baseline_trials_idx,
    optoicalStim_trials_idx,
    washout_trials_idx,
    *,
    drop_frame_events=True,
    frame_event_labels=("frame_events_timestamps", "frame_events_timestamp"),
    trigger_stimulus="reachInit_stimROI_timestamps",
    stimROI_stimulus="stimROI_timestamps",
    optical_stimulus="optical_timestamps",
    tone2_stimulus="tone2_timestamps",
    tone1_stimulus="tone1_timestamps",
):
    """
    Returns
    -------
    pca_event_meta : pd.DataFrame
        Per-trial metadata aligned to trigger_stimulus start times.
    tone1_start_times, tone2_start_times, stimROI_start_times, optical_start_times, all_stimROI_triggers_start_times : np.ndarray
        Raw event start times extracted from stim_df (after optional frame-event removal).
    """

    import numpy as np
    import pandas as pd

    if "stimulus" not in stim_df.columns:
        raise ValueError("stim_df has no 'stimulus' column")
    if "start_time" not in stim_df.columns:
        raise ValueError("stim_df has no 'start_time' column")

    # Optionally drop frame-events rows
    df_stim = stim_df.copy()
    if drop_frame_events:
        df_stim = df_stim[
            ~df_stim["stimulus"].astype(str).str.strip().str.lower().isin(
                [s.strip().lower() for s in frame_event_labels]
            )
        ].reset_index(drop=True)

    # Extract start times for all relevant stimulus events
    all_stimROI_triggers = df_stim[df_stim["stimulus"] == trigger_stimulus]
    stim_ROI_df = df_stim[df_stim["stimulus"] == stimROI_stimulus]
    optical_df = df_stim[df_stim["stimulus"] == optical_stimulus]
    tone2_df = df_stim[df_stim["stimulus"] == tone2_stimulus]
    tone1_df = df_stim[df_stim["stimulus"] == tone1_stimulus]

    tone1_start_times = tone1_df["start_time"].to_numpy()
    tone2_start_times = tone2_df["start_time"].to_numpy()
    stimROI_start_times = stim_ROI_df["start_time"].to_numpy()
    optical_start_times = optical_df["start_time"].to_numpy()
    all_stimROI_triggers_start_times = all_stimROI_triggers["start_time"].to_numpy()

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

    def _normalize_trial_indices(idx_nested, n_trials):
        """
        Accept nested trial index/number groups and normalize to 0-based integer indices.
        Handles 1-based trial numbers automatically.
        Returns list[list[int]] preserving epoch nesting.
        """
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

        # 1-based if max <= n_trials and min >= 1
        one_based = (max_v <= n_trials) and (min_v >= 1)

        norm = []
        for ep in epochs:
            ep0 = [v - 1 for v in ep] if one_based else [v for v in ep]
            ep0 = [v for v in ep0 if 0 <= v < n_trials]
            norm.append(sorted(list(set(ep0))))
        return norm

    if len(all_stimROI_triggers_start_times) == 0:
        raise ValueError(f"No events found for trigger_stimulus='{trigger_stimulus}'")

    all_stimROI_triggers_start_times = np.asarray(all_stimROI_triggers_start_times, dtype=float)
    n_trials_total = len(all_stimROI_triggers_start_times)

    baseline_idx_epochs = _normalize_trial_indices(baseline_trials_idx, n_trials_total)
    stim_idx_epochs = _normalize_trial_indices(optoicalStim_trials_idx, n_trials_total)
    wash_idx_epochs = _normalize_trial_indices(washout_trials_idx, n_trials_total)

    # Build metadata rows
    rows = []

    # baseline epoch_id=0
    for ep_idx in baseline_idx_epochs:
        for tidx in ep_idx:
            rows.append(
                {
                    "trial_index0": int(tidx),
                    "trial_number": int(tidx + 1),
                    "start_time": float(all_stimROI_triggers_start_times[tidx]),
                    "condition": "baseline",
                    "epoch_id": 0,
                    "condition_epoch": "baseline_epoch",
                }
            )

    for ep_i, ep_idx in enumerate(stim_idx_epochs, start=1):
        for tidx in ep_idx:
            rows.append(
                {
                    "trial_index0": int(tidx),
                    "trial_number": int(tidx + 1),
                    "start_time": float(all_stimROI_triggers_start_times[tidx]),
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
                    "start_time": float(all_stimROI_triggers_start_times[tidx]),
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

    return (
        pca_event_meta,
        tone1_start_times,
        tone2_start_times,
        stimROI_start_times,
        optical_start_times,
        all_stimROI_triggers_start_times,
    )


def _normalize_event_time_source_name(name: str) -> str:
    key = str(name).strip().lower()
    alias = {
        "start_time": "start_time",
        "reachinit": "all_stimROI_triggers_start_times",
        "reachinit_stimroi": "all_stimROI_triggers_start_times",
        "reachinit_stimroi_timestamps": "all_stimROI_triggers_start_times",
        "trigger": "all_stimROI_triggers_start_times",
        "all_stimroi_triggers_start_times": "all_stimROI_triggers_start_times",
        "all_stimroi_triggers": "all_stimROI_triggers_start_times",
        "tone1": "tone1_start_times",
        "tone1_start_times": "tone1_start_times",
        "tone2": "tone2_start_times",
        "tone2_start_times": "tone2_start_times",
        "stimroi": "stimROI_start_times",
        "stimroi_start_times": "stimROI_start_times",
        "optical": "optical_start_times",
        "optical_start_times": "optical_start_times",
        "custom": "custom_event_start_times",
        "custom_event": "custom_event_start_times",
        "custom_event_start_times": "custom_event_start_times",
    }
    return alias.get(key, str(name).strip())


def _nearest_event_time_lookup(ref_times: np.ndarray, candidate_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if candidate_times.size == 0:
        out = np.full(ref_times.shape, np.nan, dtype=float)
        d = np.full(ref_times.shape, np.nan, dtype=float)
        return out, d

    cand = np.asarray(candidate_times, dtype=float)
    cand = cand[np.isfinite(cand)]
    if cand.size == 0:
        out = np.full(ref_times.shape, np.nan, dtype=float)
        d = np.full(ref_times.shape, np.nan, dtype=float)
        return out, d

    cand_sorted = np.sort(cand)
    ref = np.asarray(ref_times, dtype=float)
    out = np.full(ref.shape, np.nan, dtype=float)
    d = np.full(ref.shape, np.nan, dtype=float)

    valid = np.isfinite(ref)
    if not np.any(valid):
        return out, d

    x = ref[valid]
    right = np.searchsorted(cand_sorted, x, side="left")
    left = np.clip(right - 1, 0, cand_sorted.size - 1)
    right = np.clip(right, 0, cand_sorted.size - 1)

    left_val = cand_sorted[left]
    right_val = cand_sorted[right]
    choose_right = np.abs(right_val - x) < np.abs(left_val - x)
    picked = np.where(choose_right, right_val, left_val)
    dist = np.abs(picked - x)

    out[valid] = picked
    d[valid] = dist
    return out, d


def align_pca_event_meta_start_times(
    pca_event_meta: pd.DataFrame,
    *,
    align_to: str = "all_stimROI_triggers_start_times",
    tone1_start_times: np.ndarray | list[float] | None = None,
    tone2_start_times: np.ndarray | list[float] | None = None,
    stimROI_start_times: np.ndarray | list[float] | None = None,
    optical_start_times: np.ndarray | list[float] | None = None,
    all_stimROI_triggers_start_times: np.ndarray | list[float] | None = None,
    custom_event_start_times: np.ndarray | list[float] | None = None,
    mismatch: str = "index_then_nearest",
    max_delta_s: float | None = None,
    drop_unmatched: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Return a copy of pca_event_meta with start_time remapped to a selected event source.

    Parameters
    ----------
    align_to:
      One of:
      - 'start_time' (keep existing values)
      - 'tone1_start_times'
      - 'tone2_start_times'
      - 'stimROI_start_times'
      - 'optical_start_times'
      - 'all_stimROI_triggers_start_times' (reach-init trigger)
      Common aliases are accepted (e.g. 'tone1', 'stimROI', 'optical', 'reachInit').

    mismatch:
      - 'index'              : strict trial_index0 -> source[index]
      - 'nearest'            : nearest source event to existing start_time
      - 'index_then_nearest' : index mapping first, then nearest for missing rows
    """
    if "trial_index0" not in pca_event_meta.columns:
        raise ValueError("pca_event_meta must contain trial_index0.")
    if "start_time" not in pca_event_meta.columns:
        raise ValueError("pca_event_meta must contain start_time.")

    mode = str(mismatch).strip().lower()
    valid_modes = {"index", "nearest", "index_then_nearest"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mismatch='{mismatch}'. Use one of {sorted(valid_modes)}.")

    source_name = _normalize_event_time_source_name(align_to)
    source_map = {
        "tone1_start_times": tone1_start_times,
        "tone2_start_times": tone2_start_times,
        "stimROI_start_times": stimROI_start_times,
        "optical_start_times": optical_start_times,
        "all_stimROI_triggers_start_times": all_stimROI_triggers_start_times,
        "custom_event_start_times": custom_event_start_times,
    }

    em = pca_event_meta.copy().reset_index(drop=True)

    if source_name == "start_time":
        em["start_time"] = pd.to_numeric(em["start_time"], errors="coerce")
        em["start_time_source"] = "start_time"
        em["start_time_align_method"] = "existing"
        em["start_time_align_abs_delta_s"] = 0.0
        report = {
            "source": "start_time",
            "method": "existing",
            "input_rows": int(len(pca_event_meta)),
            "output_rows": int(len(em)),
            "unmatched_rows": 0,
        }
        return em, report

    if source_name not in source_map:
        raise ValueError(
            f"Unknown align_to='{align_to}'. "
            "Expected start_time, tone1, tone2, stimROI, optical, custom_event_start_times, "
            "or all_stimROI_triggers_start_times."
        )

    src = source_map[source_name]
    if src is None:
        raise ValueError(
            f"align_to='{source_name}' requested, but corresponding array was not provided."
        )
    src_arr = np.asarray(src, dtype=float).ravel()
    if src_arr.size == 0:
        raise ValueError(f"Source array '{source_name}' is empty.")

    trial_idx = pd.to_numeric(em["trial_index0"], errors="coerce").to_numpy(dtype=float)
    ref_time = pd.to_numeric(em["start_time"], errors="coerce").to_numpy(dtype=float)

    aligned = np.full(len(em), np.nan, dtype=float)
    method = np.array(["unmatched"] * len(em), dtype=object)
    delta = np.full(len(em), np.nan, dtype=float)

    used_index = False
    if mode in {"index", "index_then_nearest"}:
        valid_idx = np.isfinite(trial_idx)
        idx_int = np.zeros(len(em), dtype=int)
        idx_int[valid_idx] = trial_idx[valid_idx].astype(int)
        in_range = valid_idx & (idx_int >= 0) & (idx_int < src_arr.size)
        aligned[in_range] = src_arr[idx_int[in_range]]
        method[in_range] = "index"
        delta[in_range] = np.abs(aligned[in_range] - ref_time[in_range])
        used_index = True

    if mode in {"nearest", "index_then_nearest"}:
        need = np.isnan(aligned)
        need_idx = np.flatnonzero(need)
        if need_idx.size > 0:
            nearest_val, nearest_dist = _nearest_event_time_lookup(ref_time[need], src_arr)
            aligned[need] = nearest_val
            delta[need] = nearest_dist
            matched_idx = need_idx[np.isfinite(nearest_val)]
            method[matched_idx] = "nearest"

    if mode == "index" and np.isnan(aligned).any():
        n_bad = int(np.isnan(aligned).sum())
        raise ValueError(
            f"Index alignment failed for {n_bad} rows. "
            f"Source '{source_name}' length is {src_arr.size}, but some trial_index0 are out of range. "
            "Use mismatch='index_then_nearest' or mismatch='nearest' to fill by nearest event time."
        )

    if max_delta_s is not None:
        max_delta_s = float(max_delta_s)
        too_far = np.isfinite(delta) & (delta > max_delta_s)
        aligned[too_far] = np.nan
        method[too_far] = "delta_exceeded"

    em["start_time"] = aligned
    em["start_time_source"] = source_name
    em["start_time_align_method"] = method
    em["start_time_align_abs_delta_s"] = delta

    if drop_unmatched:
        em = em.dropna(subset=["start_time"]).reset_index(drop=True)

    report = {
        "source": source_name,
        "method": mode,
        "used_index": bool(used_index),
        "input_rows": int(len(pca_event_meta)),
        "output_rows": int(len(em)),
        "unmatched_rows": int(np.isnan(aligned).sum()),
    }
    return em, report


def map_source_events_to_pca_trials(
    pca_event_meta: pd.DataFrame,
    *,
    source_times: np.ndarray | list[float],
    source_name: str,
    drop_before_first_trial: bool = False,
    sort_source_times: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Expand trial-level metadata to one row per source event time.

    Mapping rule:
      source_time maps to the nearest trial start_time in pca_event_meta.
      This is robust to event streams that occur slightly before or after
      the trial trigger used to build trial start_time.
    """
    if "start_time" not in pca_event_meta.columns:
        raise ValueError("pca_event_meta must contain start_time.")

    em_base = pca_event_meta.copy().reset_index(drop=True)
    em_base["start_time"] = pd.to_numeric(em_base["start_time"], errors="coerce")
    em_base = em_base.dropna(subset=["start_time"]).reset_index(drop=True)
    if len(em_base) == 0:
        raise ValueError("pca_event_meta has no valid start_time values.")

    if "trial_index0" in em_base.columns:
        em_base = em_base.sort_values("trial_index0").reset_index(drop=True)
    else:
        em_base = em_base.sort_values("start_time").reset_index(drop=True)

    src_arr = np.asarray(source_times, dtype=float).ravel()
    n_source_input = int(src_arr.size)
    if n_source_input == 0:
        raise ValueError(f"Source array '{source_name}' is empty.")

    finite_mask = np.isfinite(src_arr)
    src_finite = src_arr[finite_mask]
    src_idx = np.flatnonzero(finite_mask)
    n_non_finite = int(n_source_input - src_finite.size)
    if src_finite.size == 0:
        raise ValueError(f"Source array '{source_name}' has no finite values.")

    if bool(sort_source_times):
        order = np.argsort(src_finite, kind="mergesort")
        src_finite = src_finite[order]
        src_idx = src_idx[order]

    trial_starts = pd.to_numeric(em_base["start_time"], errors="coerce").to_numpy(dtype=float)
    right = np.searchsorted(trial_starts, src_finite, side="left")
    left = np.clip(right - 1, 0, len(trial_starts) - 1)
    right_clipped = np.clip(right, 0, len(trial_starts) - 1)

    left_val = trial_starts[left]
    right_val = trial_starts[right_clipped]
    choose_right = np.abs(src_finite - right_val) <= np.abs(src_finite - left_val)
    trial_lookup_idx = np.where(choose_right, right_clipped, left)

    before_first = src_finite < trial_starts[0]
    n_before_first = int(before_first.sum())

    if bool(drop_before_first_trial):
        keep_mask = ~before_first
        src_used = src_finite[keep_mask]
        src_used_idx = src_idx[keep_mask]
        trial_lookup_idx = trial_lookup_idx[keep_mask]
    else:
        trial_lookup_idx = np.clip(trial_lookup_idx, 0, len(em_base) - 1)
        src_used = src_finite
        src_used_idx = src_idx

    mapped = em_base.iloc[trial_lookup_idx].copy().reset_index(drop=True)
    trial_start_time = pd.to_numeric(mapped["start_time"], errors="coerce").to_numpy(dtype=float)

    mapped["trial_start_time"] = trial_start_time
    mapped["start_time"] = src_used
    mapped["start_time_source"] = str(source_name)
    mapped["start_time_align_method"] = "nearest_trial_start"
    mapped["start_time_align_abs_delta_s"] = np.abs(src_used - trial_start_time)
    mapped["source_event_index"] = src_used_idx.astype(int)

    report = {
        "source": str(source_name),
        "method": "nearest_trial_start",
        "trial_rows_input": int(len(em_base)),
        "source_count_input": n_source_input,
        "source_count_finite": int(src_finite.size),
        "source_count_non_finite": n_non_finite,
        "dropped_before_first_trial": n_before_first if bool(drop_before_first_trial) else 0,
        "output_rows": int(len(mapped)),
        "sorted_source_times": bool(sort_source_times),
    }
    return mapped, report


def apply_runner_post_alignment(
    em_aligned: pd.DataFrame,
    align_to: str,
    *,
    count_mode: str = "match_source_count",
    enforce_source_condition: bool = True,
    base_event_meta: pd.DataFrame | None = None,
    tone1_start_times: np.ndarray | list[float] | None = None,
    tone2_start_times: np.ndarray | list[float] | None = None,
    stimROI_start_times: np.ndarray | list[float] | None = None,
    optical_start_times: np.ndarray | list[float] | None = None,
    all_stimROI_triggers_start_times: np.ndarray | list[float] | None = None,
    custom_event_start_times: np.ndarray | list[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply runner-level alignment patches used by PCA_master_runner notebooks.

    This mirrors the runner notebook logic:
    1) optional count-mode processing,
    2) source-condition filtering,
    3) optional stimulation-sequence remap for stimROI_start_times.
    """
    out = em_aligned.copy().reset_index(drop=True)
    source_name = _normalize_event_time_source_name(align_to)
    source_map = {
        "tone1_start_times": tone1_start_times,
        "tone2_start_times": tone2_start_times,
        "stimROI_start_times": stimROI_start_times,
        "optical_start_times": optical_start_times,
        "all_stimROI_triggers_start_times": all_stimROI_triggers_start_times,
        "custom_event_start_times": custom_event_start_times,
    }

    cm = str(count_mode).strip().lower()
    if cm not in {"preserve_aligned_rows", "match_source_count", "use_all_source_events"}:
        raise ValueError(
            f"Invalid count_mode={count_mode}. "
            "Use 'preserve_aligned_rows', 'match_source_count', or 'use_all_source_events'."
        )

    expand_report: dict[str, Any] = {}
    rows_before_count_mode = int(len(out))
    if cm == "use_all_source_events" and source_name != "start_time":
        src_vals = source_map.get(source_name, None)
        if src_vals is None:
            raise ValueError(
                f"count_mode='use_all_source_events' requires source array for '{source_name}', but it was not provided."
            )
        base_for_expand = base_event_meta if base_event_meta is not None else out
        out, expand_report = map_source_events_to_pca_trials(
            pca_event_meta=base_for_expand,
            source_times=src_vals,
            source_name=source_name,
            drop_before_first_trial=False,
            sort_source_times=False,
        )
    elif cm == "match_source_count" and source_name != "start_time" and "start_time_align_method" in out.columns:
        out = out[out["start_time_align_method"].astype(str).eq("index")].reset_index(drop=True)
    rows_after_count_mode = int(len(out))

    source_condition_map = {"stimROI_start_times": "stimulation"}
    source_condition = source_condition_map.get(source_name, None)
    rows_before_source_condition = int(len(out))
    if bool(enforce_source_condition) and (source_condition is not None) and ("condition" in out.columns):
        out = out[out["condition"].astype(str).str.lower().eq(str(source_condition).lower())].reset_index(drop=True)
    rows_after_source_condition = int(len(out))

    stim_sequence_source_count = 0
    stim_sequence_stimulation_rows = 0
    stim_sequence_rows_mapped = 0
    if (
        source_name == "stimROI_start_times"
        and cm != "use_all_source_events"
        and bool(enforce_source_condition)
        and base_event_meta is not None
        and ("condition" in base_event_meta.columns)
    ):
        stim_src = np.asarray(stimROI_start_times if stimROI_start_times is not None else [], dtype=float).ravel()
        stim_rows_all = base_event_meta[
            base_event_meta["condition"].astype(str).str.lower().eq("stimulation")
        ].sort_values("trial_index0").reset_index(drop=True)
        n_map = int(min(len(stim_rows_all), stim_src.size))
        if n_map > 0:
            stim_rows = stim_rows_all.iloc[:n_map].copy().reset_index(drop=True)
            orig_start = pd.to_numeric(stim_rows["start_time"], errors="coerce").to_numpy(dtype=float)
            new_start = stim_src[:n_map]
            stim_rows["start_time"] = new_start
            stim_rows["start_time_source"] = source_name
            stim_rows["start_time_align_method"] = "stimulation_sequence_index"
            stim_rows["start_time_align_abs_delta_s"] = np.abs(new_start - orig_start)
            out = stim_rows
        stim_sequence_source_count = int(stim_src.size)
        stim_sequence_stimulation_rows = int(len(stim_rows_all))
        stim_sequence_rows_mapped = int(n_map)

    patch_report = {
        "count_mode": cm,
        "source_condition_filter": source_condition if bool(enforce_source_condition) else None,
        "output_rows_before_count_mode": rows_before_count_mode,
        "output_rows_after_count_mode": rows_after_count_mode,
        "output_rows_before_source_condition_filter": rows_before_source_condition,
        "output_rows_after_source_condition_filter": int(len(out)),
        "output_rows_after_source_condition_only": rows_after_source_condition,
        "expanded_from_source": bool(cm == "use_all_source_events" and source_name != "start_time"),
        "expand_report": expand_report,
        "stim_sequence_source_count": stim_sequence_source_count,
        "stim_sequence_stimulation_rows": stim_sequence_stimulation_rows,
        "stim_sequence_rows_mapped": stim_sequence_rows_mapped,
    }
    return out, patch_report


def configure_epoch_pca_runner(
    *,
    plots: Any,
    merged_dic: dict[str, pd.DataFrame],
    pca_event_meta: pd.DataFrame,
    tone1_start_times: np.ndarray | list[float] | None,
    tone2_start_times: np.ndarray | list[float] | None,
    stimROI_start_times: np.ndarray | list[float] | None,
    optical_start_times: np.ndarray | list[float] | None,
    all_stimROI_triggers_start_times: np.ndarray | list[float] | None,
    EPOCH_PCA_PROBE: str,
    EPOCH_PCA_BRAIN_REGION: str | None,
    EPOCH_PCA_ROI_FILTER: str | None,
    EPOCH_PCA_KSLABEL_FILTER: str,
    EPOCH_PCA_BC_LABEL_FILTER: str | list[str] | tuple[str, ...] | set[str],
    EPOCH_PCA_FILTER_TYPE: str,
    EVENT_TIME_ALIGN_TO: str,
    EVENT_TIME_ALIGN_MISMATCH: str,
    EVENT_TIME_ALIGN_COUNT_MODE: str,
    EVENT_TIME_ALIGN_MAX_DELTA_S: float | None,
    EVENT_TIME_DROP_UNMATCHED: bool,
    EVENT_TIME_ENFORCE_SOURCE_CONDITION: bool = True,
    PLOT_COLOR_MODE: int = 2,
    STIMULATION_LINESTYLE: str = ":",
    WASHOUT_LINESTYLE: str = "-",
    BASELINE_LINESTYLE: str = "-",
    BASELINE_COLOR: str = "orange",
    MODE3_BASE_COLOR: str = "gray",
    MODE3_STIMULATION_COLOR: str = "purple",
    MODE3_WASHOUT_COLOR: str = "green",
    MODE3_BASELINE_COLOR: str = "orange",
    MODE3_HIGHLIGHT_COLOR: str = "green",
    MODE3_HIGHLIGHT_WINDOW_S: float = 0.5,
    REGION_PANEL_PC123_ELEV: float = 25.0,
    REGION_PANEL_PC123_AZIM: float = -60.0,
    REGION_PANEL_PC12TIME_ELEV: float = 20.0,
    REGION_PANEL_PC12TIME_AZIM: float = -60.0,
    PLOTS_SAVE_ROOT: str | Path = "master/results",
    ALLOWED_BRAIN_REGIONS: list[str] | tuple[str, ...] | None = None,
    SHOW_PLOTS_SINGLE_DEFAULT: bool = True,
    SHOW_PLOTS_BATCH_DEFAULT: bool = False,
    print_checks: bool = True,
    custom_event_start_times: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    """
    Apply PCA notebook config logic and return computed runtime variables.

    This keeps notebook cells minimal by centralizing style setup, event-time
    alignment, filter normalization, summary checks, and save-path setup.
    """
    if ALLOWED_BRAIN_REGIONS is None:
        ALLOWED_BRAIN_REGIONS = ["PG", "SIM", "IP", "VaL", "MoP", "SnR", "RN"]
    else:
        ALLOWED_BRAIN_REGIONS = [str(x) for x in ALLOWED_BRAIN_REGIONS]

    plots.set_epoch_plot_style(
        color_mode=PLOT_COLOR_MODE,
        stimulation_linestyle=STIMULATION_LINESTYLE,
        washout_linestyle=WASHOUT_LINESTYLE,
        baseline_linestyle=BASELINE_LINESTYLE,
        baseline_color=BASELINE_COLOR,
        mode3_base_color=MODE3_BASE_COLOR,
        mode3_stimulation_color=MODE3_STIMULATION_COLOR,
        mode3_washout_color=MODE3_WASHOUT_COLOR,
        mode3_baseline_color=MODE3_BASELINE_COLOR,
        mode3_highlight_color=MODE3_HIGHLIGHT_COLOR,
        mode3_highlight_window_s=MODE3_HIGHLIGHT_WINDOW_S,
    )

    _filter_type_norm = str(EPOCH_PCA_FILTER_TYPE).strip().lower()
    if _filter_type_norm not in {"bombcell", "kilosort"}:
        raise ValueError(
            f"Invalid EPOCH_PCA_FILTER_TYPE={EPOCH_PCA_FILTER_TYPE}. Use 'bombcell' or 'kilosort'."
        )

    if _filter_type_norm == "bombcell":
        EPOCH_PCA_EFFECTIVE_KSLABEL_FILTER = "both"
        EPOCH_PCA_EFFECTIVE_BC_LABEL_FILTER = EPOCH_PCA_BC_LABEL_FILTER
        save_filter_tag = "bc"
    else:
        EPOCH_PCA_EFFECTIVE_KSLABEL_FILTER = EPOCH_PCA_KSLABEL_FILTER
        EPOCH_PCA_EFFECTIVE_BC_LABEL_FILTER = "all"
        save_filter_tag = "ks"

    if print_checks:
        print(
            f"Filter type={EPOCH_PCA_FILTER_TYPE} | effective KS={EPOCH_PCA_EFFECTIVE_KSLABEL_FILTER} "
            f"| effective BC={EPOCH_PCA_EFFECTIVE_BC_LABEL_FILTER}"
        )

    _allowed_region_map = {r.lower(): r for r in ALLOWED_BRAIN_REGIONS}

    def _allowed_regions_only(
        _regions: list[str] | tuple[str, ...] | np.ndarray,
        _allowed_map: dict[str, str] = _allowed_region_map,
        _allowed_order: list[str] = ALLOWED_BRAIN_REGIONS,
    ) -> list[str]:
        _canon: list[str] = []
        for _r in _regions:
            _k = str(_r).strip().lower()
            if _k in _allowed_map:
                _v = _allowed_map[_k]
                if _v not in _canon:
                    _canon.append(_v)
        return [r for r in _allowed_order if r in _canon]

    pca_event_meta_aligned, EVENT_TIME_ALIGN_REPORT = align_pca_event_meta_start_times(
        pca_event_meta=pca_event_meta,
        align_to=EVENT_TIME_ALIGN_TO,
        tone1_start_times=tone1_start_times,
        tone2_start_times=tone2_start_times,
        stimROI_start_times=stimROI_start_times,
        optical_start_times=optical_start_times,
        all_stimROI_triggers_start_times=all_stimROI_triggers_start_times,
        custom_event_start_times=custom_event_start_times,
        mismatch=EVENT_TIME_ALIGN_MISMATCH,
        max_delta_s=EVENT_TIME_ALIGN_MAX_DELTA_S,
        drop_unmatched=EVENT_TIME_DROP_UNMATCHED,
    )

    _event_time_source_map = {
        "start_time": None,
        "tone1_start_times": tone1_start_times,
        "tone2_start_times": tone2_start_times,
        "stimROI_start_times": stimROI_start_times,
        "optical_start_times": optical_start_times,
        "all_stimROI_triggers_start_times": all_stimROI_triggers_start_times,
        "custom_event_start_times": custom_event_start_times,
    }
    _event_time_source_name = _normalize_event_time_source_name(EVENT_TIME_ALIGN_TO)
    _event_time_source_vals = _event_time_source_map.get(_event_time_source_name, None)
    if _event_time_source_vals is None:
        EVENT_TIME_SOURCE_COUNT = int(len(pca_event_meta))
    else:
        EVENT_TIME_SOURCE_COUNT = int(np.asarray(_event_time_source_vals, dtype=float).ravel().size)

    pca_event_meta_aligned, _post_align_report = apply_runner_post_alignment(
        pca_event_meta_aligned,
        EVENT_TIME_ALIGN_TO,
        count_mode=EVENT_TIME_ALIGN_COUNT_MODE,
        enforce_source_condition=EVENT_TIME_ENFORCE_SOURCE_CONDITION,
        base_event_meta=pca_event_meta,
        tone1_start_times=tone1_start_times,
        tone2_start_times=tone2_start_times,
        stimROI_start_times=stimROI_start_times,
        optical_start_times=optical_start_times,
        all_stimROI_triggers_start_times=all_stimROI_triggers_start_times,
        custom_event_start_times=custom_event_start_times,
    )
    _count_mode = _post_align_report["count_mode"]
    _enforce_source_condition = bool(EVENT_TIME_ENFORCE_SOURCE_CONDITION)
    _source_condition = _post_align_report["source_condition_filter"]

    EVENT_TIME_ALIGN_REPORT["count_mode"] = _count_mode
    EVENT_TIME_ALIGN_REPORT["source_count"] = EVENT_TIME_SOURCE_COUNT
    EVENT_TIME_ALIGN_REPORT["output_rows_before_count_mode"] = _post_align_report["output_rows_before_count_mode"]
    EVENT_TIME_ALIGN_REPORT["output_rows_after_count_mode"] = _post_align_report["output_rows_after_count_mode"]
    EVENT_TIME_ALIGN_REPORT["source_condition_filter"] = _post_align_report["source_condition_filter"]
    EVENT_TIME_ALIGN_REPORT["output_rows_before_source_condition_filter"] = _post_align_report["output_rows_before_source_condition_filter"]
    EVENT_TIME_ALIGN_REPORT["output_rows_after_source_condition_filter"] = _post_align_report["output_rows_after_source_condition_filter"]
    EVENT_TIME_ALIGN_REPORT["output_rows_after_source_condition_only"] = _post_align_report["output_rows_after_source_condition_only"]
    EVENT_TIME_ALIGN_REPORT["expanded_from_source"] = _post_align_report["expanded_from_source"]
    EVENT_TIME_ALIGN_REPORT["expand_report"] = _post_align_report["expand_report"]
    EVENT_TIME_ALIGN_REPORT["stim_sequence_source_count"] = _post_align_report["stim_sequence_source_count"]
    EVENT_TIME_ALIGN_REPORT["stim_sequence_stimulation_rows"] = _post_align_report["stim_sequence_stimulation_rows"]
    EVENT_TIME_ALIGN_REPORT["stim_sequence_rows_mapped"] = _post_align_report["stim_sequence_rows_mapped"]

    if print_checks:
        print("event-time alignment:", EVENT_TIME_ALIGN_REPORT)

    _summary_regions_raw = plots._list_probe_brain_regions(
        merged_dic=merged_dic,
        probe=EPOCH_PCA_PROBE,
        roi_filter=EPOCH_PCA_ROI_FILTER,
        kslabel_filter="both",
        bc_label_filter="all",
    )
    _summary_regions = _allowed_regions_only(_summary_regions_raw)
    _summary_selected_br = (
        EPOCH_PCA_BRAIN_REGION
        if EPOCH_PCA_BRAIN_REGION is not None
        else (_summary_regions[0] if len(_summary_regions) > 0 else None)
    )

    _units_df_cfg = plots.pca_get_probe_units_df(
        merged_dic=merged_dic,
        probe=EPOCH_PCA_PROBE,
        roi_filter=EPOCH_PCA_ROI_FILTER,
        kslabel_filter=EPOCH_PCA_EFFECTIVE_KSLABEL_FILTER,
        bc_label_filter=EPOCH_PCA_EFFECTIVE_BC_LABEL_FILTER,
    )
    _norm_br = getattr(plots, "_normalize_brain_region_value", lambda x: str(x).strip())
    _br_counts: dict[str, int] = {}
    if ("brain_region" in _units_df_cfg.columns) and (len(_units_df_cfg) > 0):
        for _v in _units_df_cfg["brain_region"].tolist():
            if _v is None:
                continue
            _name = str(_norm_br(_v)).strip()
            if _name == "" or _name.lower() == "nan":
                continue
            _br_counts[_name] = int(_br_counts.get(_name, 0)) + 1

    _br_print_order = list(ALLOWED_BRAIN_REGIONS)
    for _k in sorted(_br_counts.keys()):
        if _k not in _br_print_order:
            _br_print_order.append(_k)

    def _angle_tag(v: float) -> str:
        return f"{float(v):g}".replace("-", "m").replace(".", "p")

    _align_source_name = _normalize_event_time_source_name(EVENT_TIME_ALIGN_TO)
    _align_token = str(_align_source_name).strip().lower()
    if _align_token.endswith("_start_times"):
        _align_token = _align_token[: -len("_start_times")]
    _align_token = re.sub(r"[^a-z0-9_]+", "_", _align_token).strip("_")
    if _align_token == "":
        _align_token = "start_time"
    EVENT_TIME_ALIGN_FOLDER = f"aligned_{_align_token}"

    ANGLE_FOLDER_TAG = (
        f"e{_angle_tag(REGION_PANEL_PC123_ELEV)}_a{_angle_tag(REGION_PANEL_PC123_AZIM)}_"
        f"e{_angle_tag(REGION_PANEL_PC12TIME_ELEV)}_a{_angle_tag(REGION_PANEL_PC12TIME_AZIM)}"
    )

    PLOTS_SAVE_PATH = (
        Path(PLOTS_SAVE_ROOT)
        / "byEpoch"
        / f"mode_{PLOT_COLOR_MODE}_{save_filter_tag}"
        / EVENT_TIME_ALIGN_FOLDER
        / ANGLE_FOLDER_TAG
    )

    if print_checks:
        print("\n========== PCA Plot Summary ==========")
        print(f"Probe: {EPOCH_PCA_PROBE}")
        print(f"Filter type: {EPOCH_PCA_FILTER_TYPE}")
        print(f"Effective KS filter: {EPOCH_PCA_EFFECTIVE_KSLABEL_FILTER}")
        print(f"Effective BC filter: {EPOCH_PCA_EFFECTIVE_BC_LABEL_FILTER}")
        print(f"Event aligned to: {EVENT_TIME_ALIGN_TO}")
        print(f"Alignment count mode: {_count_mode}")
        print(f"Source condition enforcement: {_enforce_source_condition} | source condition: {_source_condition}")
        print(f"Event source count: {EVENT_TIME_SOURCE_COUNT}")
        print(f"Total aligned events: {len(pca_event_meta_aligned)}")
        print(f"Brain regions ({len(_summary_regions)}): {_summary_regions}")
        print(f"Selected brain region: {_summary_selected_br}")
        print("Unit counts by brain region (current filters):")
        for _br in _br_print_order:
            print(f"  {_br}: {int(_br_counts.get(_br, 0))}")
        print(f"Total units after current filters: {len(_units_df_cfg)}")
        print("======================================\n")
        print("plots will be saved to:", PLOTS_SAVE_PATH)

    return {
        "EPOCH_PCA_EFFECTIVE_KSLABEL_FILTER": EPOCH_PCA_EFFECTIVE_KSLABEL_FILTER,
        "EPOCH_PCA_EFFECTIVE_BC_LABEL_FILTER": EPOCH_PCA_EFFECTIVE_BC_LABEL_FILTER,
        "save_filter_tag": save_filter_tag,
        "ALLOWED_BRAIN_REGIONS": ALLOWED_BRAIN_REGIONS,
        "_allowed_regions_only": _allowed_regions_only,
        "pca_event_meta_aligned": pca_event_meta_aligned,
        "EVENT_TIME_ALIGN_REPORT": EVENT_TIME_ALIGN_REPORT,
        "EVENT_TIME_SOURCE_COUNT": EVENT_TIME_SOURCE_COUNT,
        "_count_mode": _count_mode,
        "_summary_regions": _summary_regions,
        "_summary_selected_br": _summary_selected_br,
        "EVENT_TIME_ALIGN_FOLDER": EVENT_TIME_ALIGN_FOLDER,
        "ANGLE_FOLDER_TAG": ANGLE_FOLDER_TAG,
        "PLOTS_SAVE_PATH": PLOTS_SAVE_PATH,
        "SHOW_PLOTS_SINGLE": bool(SHOW_PLOTS_SINGLE_DEFAULT),
        "SHOW_PLOTS_BATCH": bool(SHOW_PLOTS_BATCH_DEFAULT),
    }


def merge_units_with_metrics(
    df_units_dic: dict[str, pd.DataFrame],
    qm_dic: dict[str, pd.DataFrame] | None = None,
    cluster_dic: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Merge probe-unit tables with optional quality-metrics and cluster tables.
    """
    def _ensure_cluster_id(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().reset_index(drop=True)
        if "cluster_id" in out.columns:
            out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="coerce")
            return out

        for c in ["phy_clusterID", "unit_id", "id", "cluster", "clusterId"]:
            if c not in out.columns:
                continue
            vals = pd.to_numeric(out[c], errors="coerce")
            if vals.notna().any():
                out["cluster_id"] = vals
                return out

        # Fallback used by some NWB exports where unit rows are ordered by cluster id.
        out["cluster_id"] = np.arange(len(out), dtype=int)
        return out

    merged_dic: dict[str, pd.DataFrame] = {}
    for probe, u0 in df_units_dic.items():
        u = u0.copy().reset_index(drop=True)

        if qm_dic is None and cluster_dic is None:
            merged_dic[probe] = u
            continue

        m = _ensure_cluster_id(u)

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


def build_pca_event_meta_from_stim_df(stim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a fallback pca_event_meta directly from stim_df block labels.
    This is used when explicit trial-index epoch lists are not provided.
    """
    if "event_time_s" not in stim_df.columns:
        raise ValueError("stim_df must contain event_time_s.")
    if "trial_index" not in stim_df.columns:
        raise ValueError("stim_df must contain trial_index.")
    if "block_label" not in stim_df.columns:
        raise ValueError("stim_df must contain block_label.")

    out = stim_df.copy().reset_index(drop=True)
    out["start_time"] = pd.to_numeric(out["event_time_s"], errors="coerce")
    out = out.dropna(subset=["start_time"]).reset_index(drop=True)

    def _cond_epoch(lbl: Any, is_opto_val: Any) -> tuple[str, int, str]:
        s = str(lbl).strip().lower()
        if s.startswith("opto_epoch_"):
            try:
                ep = int(s.split("_")[-1])
            except Exception:
                ep = 1
            return "stimulation", ep, f"stimulation_epoch_{ep}"
        if s.startswith("washout_epoch_"):
            try:
                ep = int(s.split("_")[-1])
            except Exception:
                ep = 1
            return "washout", ep, f"washout_epoch_{ep}"
        if s == "baseline":
            return "baseline", 0, "baseline_epoch"

        # Fallback if block_label is unexpected
        is_opto_bool = bool(is_opto_val)
        if is_opto_bool:
            return "stimulation", 1, "stimulation_epoch_1"
        return "baseline", 0, "baseline_epoch"

    conds = []
    epochs = []
    cond_epochs = []
    for _, r in out.iterrows():
        cond, ep, ce = _cond_epoch(r.get("block_label", ""), r.get("is_opto", False))
        conds.append(cond)
        epochs.append(ep)
        cond_epochs.append(ce)

    pca_event_meta = pd.DataFrame(
        {
            "trial_index0": pd.to_numeric(out["trial_index"], errors="coerce").astype("Int64"),
            "trial_number": pd.to_numeric(out["trial_index"], errors="coerce").astype("Int64") + 1,
            "start_time": out["start_time"].astype(float),
            "condition": conds,
            "epoch_id": epochs,
            "condition_epoch": cond_epochs,
        }
    )
    pca_event_meta = (
        pca_event_meta.dropna(subset=["trial_index0", "start_time"])
        .astype({"trial_index0": int, "trial_number": int, "epoch_id": int})
        .drop_duplicates(subset=["trial_index0"])
        .sort_values("trial_index0")
        .reset_index(drop=True)
    )
    return pca_event_meta


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


def append_custom_event_arrays(
    events_df: pd.DataFrame,
    time_col: str,
    label_col: str,
    custom_event_arrays: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Append custom numpy/list timestamp arrays as additional event rows.
    Useful for overlay plotting with arrays like opto trigger start_times.
    """
    out = events_df.copy()
    if custom_event_arrays is None:
        custom_event_arrays = {}

    rows = []
    for label, arr in custom_event_arrays.items():
        if arr is None:
            continue
        vals = np.asarray(arr, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append(pd.DataFrame({label_col: [str(label)] * len(vals), time_col: vals}))

    if len(rows) > 0:
        out = pd.concat([out] + rows, ignore_index=True, sort=False)

    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return out


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
        pca_event_meta = build_pca_event_meta_from_stim_df(stim_df)

    return save_processed_bundle(
        out_dir=out_dir,
        merged_dic=merged_dic,
        stim_df=stim_df,
        pca_event_meta=pca_event_meta,
        extras=extras,
    )

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
from os import listdir
from os.path import isfile, join
import re
import MasterPSD as PSD
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import glob
import os, re, csv
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, List, Optional
# Accepts ..._<STAMP>... with STAMP = 20251124T192840Z or 20251124T192840+0000
STAMP_ANY = re.compile(r"_(\d{8}T\d{6}(?:Z|[+-]\d{4}))")

def _parse_stamp(s: str) -> Optional[datetime]:
    try:
        if s.endswith("Z"):
            return datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return datetime.strptime(s, "%Y%m%dT%H%M%S%z")
    except Exception:
        return None

def _load_meta(meta_path: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    if not os.path.isfile(meta_path): return meta
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line: continue
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()
    return meta

def _load_from_npz(npz_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with np.load(npz_path, allow_pickle=False) as z:
        for k in z.files:
            out[k] = z[k]
    return out

def _load_from_csv(csv_path: str) -> Dict[str, Any]:
    if not os.path.isfile(csv_path): return {}
    times, counts, voa, pm, reset = [], [], [], [], []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        has_reset = False
        if header:
            hl = [h.strip().lower() for h in header]
            has_reset = ("reset_bin" in hl)
        for row in r:
            if not row: continue
            times.append(np.datetime64(row[0]))
            counts.append(int(float(row[1])))
            voa.append(float(row[2]))
            pm.append(float(row[3]))
            if has_reset and len(row) > 4:
                try: reset.append(bool(int(row[4])))
                except ValueError: reset.append(row[4].strip().lower() in ("true","t","yes","y"))
    data: Dict[str, Any] = {
        "timestamps": np.array(times, dtype="datetime64[us]"),
        "counts":     np.array(counts, dtype=np.uint32),
        "voa_volts":  np.array(voa, dtype=float),
        "pm_volts":   np.array(pm,  dtype=float),
    }
    if reset:
        data["reset_mask_bins"] = np.array(reset, dtype=bool)
    return data

def load_measurements(base_dir: str, date_str: str, subfolder: str, verbose: bool=False) -> List[Dict[str, Any]]:
    """
    Scans base_dir/YYYY-MM-DD/subfolder for triplets:
      <base>_<STAMP>*.npz
      <base>_<STAMP>*.csv
      <base>_<STAMP>*_meta.txt
    Uses the *last* stamp occurrence in the filename. Returns newest→oldest.
    """
    root = os.path.join(base_dir, date_str, subfolder)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Folder not found: {root}")

    recs: Dict[tuple, Dict[str, Any]] = {}
    for fn in os.listdir(root):
        full = os.path.join(root, fn)
        if not os.path.isfile(full): 
            if verbose: print("[skip dir]", fn)
            continue

        matches = list(STAMP_ANY.finditer(fn))
        if not matches:
            if verbose: print("[no stamp]", fn)
            continue
        m = matches[-1]                      # use the last stamp in the name
        stamp_str = m.group(1)
        stamp_dt  = _parse_stamp(stamp_str)
        if not stamp_dt:
            if verbose: print("[bad stamp]", fn, "|", stamp_str)
            continue

        base = fn[:m.start()]                # everything before the stamp
        kind = None
        if fn.endswith("_meta.txt"): kind = "meta"
        else:
            _, ext = os.path.splitext(fn)
            ext = ext.lower()
            if ext == ".npz": kind = "npz"
            elif ext == ".csv": kind = "csv"
        if not kind:
            if verbose: print("[unknown ext]", fn)
            continue

        key = (base, stamp_dt)
        rec = recs.setdefault(key, {"base": base.rstrip("_"), "stamp": stamp_dt, "npz": None, "csv": None, "meta": None})
        rec[kind] = full

    entries = sorted(recs.values(), key=lambda d: d["stamp"], reverse=True)

    out: List[Dict[str, Any]] = []
    for g in entries:
        data: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}
        if g["npz"]:
            try: data = _load_from_npz(g["npz"])
            except Exception as e:
                if verbose: print("[npz load fail]", g["npz"], e)
        if not data and g["csv"]:
            try: data = _load_from_csv(g["csv"])
            except Exception as e:
                if verbose: print("[csv load fail]", g["csv"], e)
        if g["meta"]:
            try: meta = _load_meta(g["meta"])
            except Exception as e:
                if verbose: print("[meta load fail]", g["meta"], e)

        out.append({
            "name": g["base"],
            "timestamp": g["stamp"],   # tz-aware datetime
            "npz_path": g["npz"],
            "csv_path": g["csv"],
            "meta_path": g["meta"],
            "data": data,
            "meta": meta,
        })
    if verbose:
        print(f"Loaded {len(out)} measurements from {root}")
    return out

def timestamps_to_elapsed_seconds(timestamps: np.ndarray) -> np.ndarray:
    """
    Convert an array of absolute timestamps (numpy datetime64)
    into time since start of measurement, in seconds.
    """
    ts = np.asarray(timestamps)

    if ts.dtype.kind != 'M':  # not a datetime64
        raise TypeError(f"Expected datetime64 array, got {ts.dtype}")

    # time offset from first sample
    dt = ts - ts[0]

    # convert timedeltas to float seconds
    elapsed_sec = dt / np.timedelta64(1, 's')
    return elapsed_sec.astype(float)

# ----------------- helpers -----------------

def _timestr_to_seconds(tstr: str) -> float:
    h, m, s, us = tstr.split(':')
    return (int(h) * 3600) + (int(m) * 60) + int(s) + int(us) / 1e6

def _unwrap_seconds(sec: np.ndarray) -> np.ndarray:
    if sec.size <= 1:
        return sec.astype(float)
    out = sec.astype(float).copy()
    add = 0.0
    for i in range(1, sec.size):
        if sec[i] < sec[i-1]:
            add += 86400.0  # add a day after a rollover
        out[i] += add
    return out

def _visibility_from_block_sums(block_sums: np.ndarray) -> float:
    if block_sums.size == 0:
        return np.nan
    Nmin = float(np.min(block_sums))
    Nmax = float(np.max(block_sums))
    den = Nmax + Nmin
    return (Nmax - Nmin) / den if den > 0 else np.nan

# ------------- main function -------------

def point_binned_visibility(
    ts,
    counts,
    n_taus: int = 40,
    tau_max: float = None,
    log_bins: bool = True,
    overlap: Optional[float] = None,   # <-- NEW: None or fraction in [0,1)
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Point-binning visibility with analytic Poisson errors and optional window overlap.

    Parameters
    ----------
    overlap : None or float in [0,1)
        If None, use non-overlapping windows (stride = K).
        If given, use stride S = round(K*(1-overlap)).
        e.g., overlap=0.5 -> 50% overlap (S ≈ K/2).

    Returns
    -------
    taus, V, QBER, V_err, QBER_err, tau_err, diagnostics
    """
    # ---- prepare data
    ts = np.asarray(ts)
    x  = np.clip(np.nan_to_num(np.asarray(counts), nan=0.0), 0, None)
    n = min(ts.size, x.size)
    ts = ts[:n]; x = x[:n]
    if n < 2:
        raise ValueError("Need at least 2 samples.")

    secs = ts
    duration = float(secs[-1] - secs[0])
    
    dts = np.diff(secs)
    dt_pos = dts[dts > 0]
    if dt_pos.size == 0:
        raise ValueError("Timestamps are not strictly advancing.")
    dt_mean = float(np.mean(dt_pos))
    dt_std  = float(np.std(dt_pos, ddof=1)) if dt_pos.size > 1 else 0.0
    dt_cv   = dt_std / dt_mean if dt_mean > 0 else np.nan

    if tau_max is None:
        tau_max = 0.5 * duration
    tau_max = max(float(tau_max), dt_mean)

    # ---- choose K values (samples per bin)
    Kmax = max(1, int(np.floor(tau_max / dt_mean)))
    if n_taus <= 1:
        Ks = np.array([1], dtype=int)
    else:
        if log_bins:
            Ks = np.round(np.logspace(0, np.log10(Kmax), num=n_taus)).astype(int)
        else:
            Ks = np.round(np.linspace(1, Kmax, num=n_taus)).astype(int)
        Ks = np.unique(np.clip(Ks, 1, Kmax))

    taus = Ks * dt_mean
    V = np.empty_like(taus, dtype=float)
    QBER = np.empty_like(taus, dtype=float)
    V_err = np.full_like(taus, np.nan, dtype=float)
    QBER_err = np.full_like(taus, np.nan, dtype=float)
    tau_err = np.full_like(taus, np.nan, dtype=float)

    # Validate overlap
    if overlap is not None:
        if not (0.0 <= overlap < 1.0):
            raise ValueError("overlap must be in [0, 1) or None.")

    # Diagnostics we’ll collect per K
    per_tau_diag: List[Dict[str, Any]] = []

    # ---- compute V(τ), analytic errors, and bin-size consistency
    for i, K in enumerate(Ks):
        # stride from overlap
        if overlap is None:
            stride = K
            overlap_eff = 0.0
        else:
            stride = max(1, int(round(K * (1.0 - overlap))))
            stride = min(stride, K)  # never exceed K
            overlap_eff = 1.0 - (stride / float(K))

        # starting indices; drop any final partial window (<K)
        if n < K:
            starts = np.array([], dtype=int)
        else:
            last_start = n - K
            starts = np.arange(0, last_start + 1, stride, dtype=int)

        n_blocks = int(starts.size)
        if n_blocks == 0:
            V[i] = np.nan
            QBER[i] = np.nan
            tau_err[i] = np.nan
            per_tau_diag.append({
                "K": int(K), "tau_target": K * dt_mean, "n_blocks": 0,
                "stride": int(stride), "overlap_eff": float(overlap_eff),
                "tau_mean": np.nan, "tau_std": np.nan, "rel_std": np.nan,
                "rmse_vs_target": np.nan, "max_abs_err": np.nan
            })
            continue

        # Efficient sliding sums via cumulative sum
        cs = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
        ends = starts + K
        block_sums = cs[ends] - cs[starts]

        # Central V and QBER
        V[i] = _visibility_from_block_sums(block_sums)
        QBER[i] = 0.5 * (1.0 - np.abs(V[i])) if np.isfinite(V[i]) else np.nan

        # Analytic errors from A=Nmax, B=Nmin (Poisson)
        A = float(np.max(block_sums))
        B = float(np.min(block_sums))
        den = A + B
        if den > 0 and np.isfinite(V[i]):
            varV = 4.0 * A * B / (den ** 3)
            V_err[i] = np.sqrt(varV)
            QBER_err[i] = 0.5 * V_err[i]

        # Actual time-span of each window (first to last sample in window)
        durations = secs[ends - 1] - secs[starts]
        tau_target = K * dt_mean

        # Consistency metrics (and τ error)
        tau_mean = float(np.mean(durations)) if durations.size else np.nan
        tau_std  = float(np.std(durations, ddof=1)) if durations.size > 1 else 0.0
        tau_err[i] = tau_std
        rel_std  = (tau_std / tau_mean) if (durations.size and tau_mean > 0) else np.nan
        rmse     = float(np.sqrt(np.mean((durations - tau_target) ** 2))) if durations.size else np.nan
        max_err  = float(np.max(np.abs(durations - tau_target))) if durations.size else np.nan

        per_tau_diag.append({
            "K": int(K),
            "stride": int(stride),
            "overlap_eff": float(overlap_eff),   # actual overlap used
            "tau_target": tau_target,
            "n_blocks": n_blocks,
            "tau_mean": tau_mean,
            "tau_std": tau_std,
            "rel_std": rel_std,
            "rmse_vs_target": rmse,
            "max_abs_err": max_err,
        })

    diagnostics = {
        "sample_spacing": {
            "n_samples": int(n),
            "duration_s": duration,
            "dt_mean_s": dt_mean,
            "dt_std_s": dt_std,
            "dt_cv": dt_cv,
            "dt_min_s": float(np.min(dt_pos)),
            "dt_med_s": float(np.median(dt_pos)),
            "dt_max_s": float(np.max(dt_pos)),
            "pct_1_99_s": (float(np.percentile(dt_pos, 1)), float(np.percentile(dt_pos, 99)))
        },
        "per_tau": per_tau_diag
    }

    if verbose:
        ss = diagnostics["sample_spacing"]
        print("\n[Sample spacing diagnostics]")
        print(f"  n = {ss['n_samples']}, duration = {ss['duration_s']:.6f} s")
        print(f"  mean Δt = {ss['dt_mean_s']:.6e} s, std = {ss['dt_std_s']:.6e} s, CV = {ss['dt_cv']:.3g}")
        print(f"  min/median/max Δt = {ss['dt_min_s']:.6e} / {ss['dt_med_s']:.6e} / {ss['dt_max_s']:.6e} s")
        p1, p99 = ss["pct_1_99_s"]
        print(f"  1–99% Δt = {p1:.6e} … {p99:.6e} s")

        print("\n[Bin-size consistency per τ (point binning)]")
        print("  K  stride  overlap  τ_target(s)  n_blocks  τ_mean(s)   τ_std(s)   rel_std   RMSE(s)   max|err|(s)")
        for r in per_tau_diag:
            print(f"{r['K']:>3d}  {r['stride']:>6d}  {r['overlap_eff']:.2f}  {r['tau_target']:.6e}  "
                  f"{r['n_blocks']:>8d}  {r['tau_mean']:.6e}  {r['tau_std']:.6e}  "
                  f"{(r['rel_std'] if np.isfinite(r['rel_std']) else np.nan):>7.3g}  "
                  f"{r['rmse_vs_target']:.6e}  {r['max_abs_err']:.6e}")

    return taus, V, QBER, V_err, QBER_err, tau_err, diagnostics

# ---------- time helpers ----------
def _timestr_to_seconds(tstr: str) -> float:
    h, m, s, us = tstr.split(':')
    return (int(h) * 3600) + (int(m) * 60) + int(s) + int(us) / 1e6

def _unwrap_seconds(sec: np.ndarray) -> np.ndarray:
    if sec.size <= 1:
        return sec.astype(float)
    out = sec.astype(float).copy()
    add = 0.0
    for i in range(1, sec.size):
        if sec[i] < sec[i-1]:
            add += 86400.0  # day rollover
        out[i] += add
    return out

def _seconds_to_timestr(t: float) -> str:
    # format seconds (can exceed 24h); we fold modulo 86400 for the string
    t_mod = t % 86400.0
    h = int(t_mod // 3600)
    m = int((t_mod % 3600) // 60)
    s = int(t_mod % 60)
    us = int(round((t_mod - int(t_mod)) * 1e6))
    # handle rounding to next second
    if us == 1_000_000:
        us = 0; s += 1
        if s == 60:
            s = 0; m += 1
            if m == 60:
                m = 0; h = (h + 1) % 24
    return f"{h:02d}:{m:02d}:{s:02d}:{us:06d}"

# ---------- main ----------
def bin_by_tau_point(
    ts,
    counts,
    tau: float,
    ts_mode: str = "start",          # 'start' | 'end' | 'center'
    include_partial: str = "drop"    # 'drop' | 'keep' | 'normalize'
):
    """
    Bin by a single τ using point-binning (K samples per bin, K ≈ tau / mean Δt).

    Parameters
    ----------
    ts : array-like of str ('HH:MM:SS:ffffff')
    counts : array-like of non-negative numbers
    tau : float, target bin size in seconds
    ts_mode : which timestamp to emit per bin:
        'start' -> first sample's original timestamp (default)
        'end'   -> last sample's original timestamp
        'center'-> mean time of samples in the bin (formatted back to string)
    include_partial : how to treat the final short bin (<K samples):
        'drop'      -> drop it (same as earlier code) [default]
        'keep'      -> keep raw sum (short bin often becomes Nmin)
        'normalize' -> scale the short-bin sum by K/L to be comparable to full bins

    Returns
    -------
    ts_binned : np.ndarray of str, one timestamp per bin
    counts_binned : np.ndarray of float, sum per bin (normalized if requested)
    info : dict with metadata (K, tau_effective, n_bins, dropped_samples, etc.)
    """
    ts = np.asarray(ts)
    x  = np.clip(np.nan_to_num(np.asarray(counts), nan=0.0), 0, None)
    n = min(ts.size, x.size)
    ts = ts[:n]; x = x[:n]
    if n < 1:
        return np.array([], dtype=str), np.array([], dtype=float), {
            "K": 0, "tau_effective": np.nan, "n_bins": 0, "dropped_samples": 0
        }

    # Parse / unwrap time, compute mean Δt and K from tau
    secs = ts
    if n < 2:
        # only one point -> one-bin result if include_partial!='drop'
        if include_partial == "drop":
            return np.array([], dtype=str), np.array([], dtype=float), {
                "K": 1, "tau_effective": np.nan, "n_bins": 0, "dropped_samples": 1
            }
        # keep/normalize: single bin with that point
        ts_b = np.array([ts[0]])
        cnts = np.array([x[0]], dtype=float)
        return ts_b, cnts, {"K": 1, "tau_effective": 0.0, "n_bins": 1, "dropped_samples": 0}

    dts = np.diff(secs)
    dt_pos = dts[dts > 0]
    if dt_pos.size == 0:
        raise ValueError("Timestamps are not strictly advancing.")
    dt_mean = float(np.mean(dt_pos))

    K = max(1, int(np.rint(float(tau) / dt_mean)))
    tau_eff = K * dt_mean

    # Build bins of K samples
    if include_partial == "drop":
        n_full = (n // K) * K
        starts = np.arange(0, n_full, K, dtype=int)
        lens = np.full_like(starts, K)
        dropped = n - n_full
    else:
        n_bins = int(np.ceil(n / K))
        starts = np.arange(0, n_bins * K, K, dtype=int)
        lens = np.minimum(K, n - starts)
        lens = lens[lens > 0]
        starts = starts[:lens.size]
        dropped = 0

    # Sums per bin
    sums = np.add.reduceat(x, starts)
    sums = sums[:lens.size]  # guard
    if include_partial == "normalize":
        # scale short bins to K-sample equivalent
        scale = (K / lens.astype(float))
        sums = sums * scale

    # Timestamp per bin
    if ts_mode == "start":
        ts_binned = ts[starts]
    elif ts_mode == "end":
        ts_binned = ts[starts + lens - 1]
    elif ts_mode == "center":
        centers = []
        for s0, L in zip(starts, lens):
            t_mean = float(np.mean(secs[s0:s0+L]))
        ts_binned = np.array(t_mean, dtype=object)
    else:
        raise ValueError("ts_mode must be one of {'start','end','center'}")

    info = {
        "K": int(K),
        "tau_effective": float(tau_eff),
        "requested_tau": float(tau),
        "n_bins": int(len(sums)),
        "dropped_samples": int(dropped),
        "include_partial": include_partial,
        "ts_mode": ts_mode,
    }
    return ts_binned, sums.astype(float), info

def rolling_visibility_segments(
    ts: np.ndarray,
    counts: np.ndarray,
    phase_length: float,
    overlap: Optional[float] = None,
    min_points: int = 2,   # require at least this many samples per segment
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide a window of length `phase_length` (s) across the data (optionally overlapped),
    and compute per-segment visibility and its analytic 1σ error assuming Poisson counts.

    Visibility in a segment:
        A = max(counts in segment)
        B = min(counts in segment)
        V = (A - B) / (A + B)
        σ_V = sqrt( 4 A B / (A + B)^3 )

    Parameters
    ----------
    ts : array-like of float
        Timestamps in seconds (nondecreasing).
    counts : array-like of float
        Non-negative counts per sample (same length/order as ts).
    phase_length : float
        Segment duration (seconds), must be > 0.
    overlap : None or float in [0,1)
        If None: non-overlapping segments (stride = phase_length).
        Else: stride = phase_length * (1 - overlap). Example: 0.5 -> 50% overlap.
    min_points : int
        Minimum number of samples required in a segment to compute visibility.

    Returns
    -------
    t_centers : np.ndarray (float)
        Center time of each segment (start + phase_length/2).
    V : np.ndarray (float)
        Visibility per segment (NaN when undefined).
    V_err : np.ndarray (float)
        Analytic 1σ error per segment (NaN when undefined).
    """
    ts = np.asarray(ts, dtype=float)
    x  = np.asarray(counts, dtype=float)

    if ts.size != x.size:
        raise ValueError("ts and counts must have the same length.")
    if ts.size == 0:
        return np.array([], float), np.array([], float), np.array([], float)
    if phase_length <= 0:
        raise ValueError("phase_length must be > 0.")

    # stride from overlap
    if overlap is None:
        stride = phase_length
    else:
        if not (0.0 <= overlap < 1.0):
            raise ValueError("overlap must be in [0, 1) or None.")
        stride = phase_length * (1.0 - overlap)
        if stride <= 0:
            stride = np.finfo(float).eps  # guard

    # sanitize counts (non-negative, finite)
    x = np.clip(np.nan_to_num(x, nan=0.0), 0.0, None)

    t0, t1 = ts[0], ts[-1]
    if (t1 - t0) < phase_length:
        return np.array([], float), np.array([], float), np.array([], float)

    # build segment starts and centers
    n_segs = int(np.floor((t1 - t0 - phase_length) / stride)) + 1
    starts  = t0 + np.arange(n_segs, dtype=float) * stride
    centers = starts + 0.5 * phase_length

    V     = np.full(n_segs, np.nan, dtype=float)
    V_err = np.full(n_segs, np.nan, dtype=float)

    # two-pointer sweep
    i_start = 0
    i_end   = 0
    N = ts.size

    for k, s in enumerate(starts):
        seg_end = s + phase_length
        # advance to segment bounds
        while i_start < N and ts[i_start] < s:
            i_start += 1
        if i_end < i_start:
            i_end = i_start
        while i_end < N and ts[i_end] < seg_end:
            i_end += 1

        L = i_end - i_start
        if L >= min_points:
            seg_vals = x[i_start:i_end]
            A = float(np.max(seg_vals))
            B = float(np.min(seg_vals))
            den = A + B
            if den > 0:
                V[k] = (A - B) / den
                # analytic 1σ (delta-method, treating A,B independent Poisson)
                V_err[k] = np.sqrt(4.0 * A * B / (den ** 3))
            else:
                V[k] = np.nan
                V_err[k] = np.nan
        else:
            V[k] = np.nan
            V_err[k] = np.nan

    return centers, V, V_err
def point_binned_visibility_over_timeframes(
    ts,
    counts,
    frame_duration_s: float,
    n_taus: int = 40,
    tau_max: float = None,
    log_bins: bool = True,
    overlap: Optional[float] = None,   # overlap between τ-windows (not frames)
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each τ, partition the data into time frames of approximately `frame_duration_s`
    and compute a visibility V for each frame. Then return, for each τ, the maximum,
    minimum, and mean visibility across all frames.

    Returns
    -------
    taus : np.ndarray, shape (M,)
        τ values in seconds (one per K).
    maxvis : np.ndarray, shape (M,)
        For each τ, the maximum visibility over all frames (NaN if no frames valid).
    minvis : np.ndarray, shape (M,)
        For each τ, the minimum visibility over all frames (NaN if no frames valid).
    meanvis : np.ndarray, shape (M,)
        For each τ, the mean visibility over all frames (NaN if no frames valid).
    """
    # ---- prepare data
    ts = np.asarray(ts)
    x  = np.clip(np.nan_to_num(np.asarray(counts), nan=0.0), 0, None)
    n = min(ts.size, x.size)
    ts = ts[:n]; x = x[:n]
    if n < 2:
        raise ValueError("Need at least 2 samples.")

    secs = ts
    duration = float(secs[-1] - secs[0])

    dts = np.diff(secs)
    dt_pos = dts[dts > 0]
    if dt_pos.size == 0:
        raise ValueError("Timestamps are not strictly advancing.")
    dt_mean = float(np.mean(dt_pos))

    if tau_max is None:
        tau_max = 0.5 * duration
    tau_max = max(float(tau_max), dt_mean)

    # ---- choose K values (samples per τ window)
    Kmax = max(1, int(np.floor(tau_max / dt_mean)))
    if n_taus <= 1:
        Ks = np.array([1], dtype=int)
    else:
        if log_bins:
            Ks = np.round(np.logspace(0, np.log10(Kmax), num=n_taus)).astype(int)
        else:
            Ks = np.round(np.linspace(1, Kmax, num=n_taus)).astype(int)
        Ks = np.unique(np.clip(Ks, 1, Kmax))

    taus = Ks * dt_mean

    # ---- validate overlap
    if overlap is not None:
        if not (0.0 <= overlap < 1.0):
            raise ValueError("overlap must be in [0, 1) or None.")

    # ---- define time frames in samples
    if frame_duration_s <= 0:
        raise ValueError("frame_duration_s must be > 0.")

    frame_K = max(1, int(round(frame_duration_s / dt_mean)))

    if duration <= frame_duration_s:
        frame_starts = np.array([0], dtype=int)
        frame_ends   = np.array([n], dtype=int)
    else:
        last_start = n - frame_K
        if last_start < 0:
            frame_starts = np.array([0], dtype=int)
            frame_ends   = np.array([n], dtype=int)
        else:
            frame_starts = np.arange(0, last_start + 1, frame_K, dtype=int)
            frame_ends   = np.minimum(frame_starts + frame_K, n)

    n_frames = frame_starts.size
    if verbose:
        print(f"[Frames] Using {n_frames} frame(s) with nominal duration "
              f"{frame_duration_s:.3f} s (≈ {frame_K} samples per frame).")

    # ---- output arrays
    maxvis  = np.full_like(taus, np.nan, dtype=float)
    minvis  = np.full_like(taus, np.nan, dtype=float)
    meanvis = np.full_like(taus, np.nan, dtype=float)

    # ---- main loop over τ (over K)
    for i, K in enumerate(Ks):
        # stride for τ-windows
        if overlap is None:
            stride = K
            overlap_eff = 0.0
        else:
            stride = max(1, int(round(K * (1.0 - overlap))))
            stride = min(stride, K)
            overlap_eff = 1.0 - (stride / float(K))

        vis_per_frame: List[float] = []

        for f in range(n_frames):
            fs = frame_starts[f]
            fe = frame_ends[f]

            seg_x = x[fs:fe]
            seg_n = seg_x.size
            if seg_n < K:
                continue

            last_start_local = seg_n - K
            starts_local = np.arange(0, last_start_local + 1, stride, dtype=int)
            if starts_local.size == 0:
                continue

            cs_local = np.concatenate(([0.0], np.cumsum(seg_x, dtype=float)))
            ends_local = starts_local + K
            block_sums = cs_local[ends_local] - cs_local[starts_local]

            V_frame = _visibility_from_block_sums(block_sums)
            if np.isfinite(V_frame):
                vis_per_frame.append(float(V_frame))

        if vis_per_frame:
            vis_arr = np.asarray(vis_per_frame, dtype=float)
            maxvis[i]  = float(np.max(vis_arr))
            minvis[i]  = float(np.min(vis_arr))
            meanvis[i] = float(np.mean(vis_arr))
        else:
            maxvis[i]  = np.nan
            minvis[i]  = np.nan
            meanvis[i] = np.nan

        if verbose:
            print(
                f"τ[{i}] = {taus[i]:.3e} s (K={K:>4d}, overlap={overlap_eff:.2f}) -> "
                f"frames used: {len(vis_per_frame):>3d}, "
                f"maxV={maxvis[i]: .4f}, minV={minvis[i]: .4f}, meanV={meanvis[i]: .4f}"
            )

    return taus, maxvis, minvis, meanvis

def timestamps_to_elapsed_seconds(timestamps: np.ndarray) -> np.ndarray:
    """
    Convert an array of absolute timestamps (numpy datetime64)
    into time since start of measurement, in seconds.
    """
    ts = np.asarray(timestamps)

    if ts.dtype.kind != 'M':  # not a datetime64
        raise TypeError(f"Expected datetime64 array, got {ts.dtype}")

    # time offset from first sample
    dt = ts - ts[0]

    # convert timedeltas to float seconds
    elapsed_sec = dt / np.timedelta64(1, 's')
    return elapsed_sec.astype(float)
def AnalyseMeasurement(ms,i,n_taus,tau,phase_length,firsthalf=True,secondhalf=True):
    ts=timestamps_to_elapsed_seconds(ms[i]['data']['timestamps'])
    cts=ms[i]['data']['counts']
    taus, Vmax, Vmin, Vmean = point_binned_visibility_over_timeframes(ts, cts, frame_duration_s=phase_length , n_taus=n_taus ,tau_max=1, log_bins=True,overlap=0.5, verbose=False)
    if firsthalf==True:
        fig1,axes1 = plt.subplots(ncols=2,figsize=(14, 3))
        axes1[0].semilogx(taus,Vmax,label='Vis (max)')
        axes1[0].semilogx(taus,Vmin,label='Vis (min)')
        axes1[0].semilogx(taus,Vmean,label='Vis (mean)')
        axes1[0].set_title('Plot of Visibility V vs Tau')
        axes1[0].set_ylabel('Visibility')
        axes1[0].set_xlabel('Averaging Time Tau')
        axes1[0].axvline(tau, color='gray', linestyle='--')
        axes1[0].legend()
        axes1[0].set_ylim(0,1.05)
    ts_b, counts_b, meta = bin_by_tau_point(ts, cts, tau, ts_mode="start", include_partial="normalize")
    if firsthalf==True:
        axes1[1].plot(ts_b,counts_b,color='tab:orange')
        axes1[1].set_title('Plot of Counts vs Time, binned at Tau')
        axes1[1].set_ylabel('Counts')
        axes1[1].set_xlabel('Time (s)')
        axes1[1].set_ylim(0,np.max(counts_b))
    tsrv, Vrv, Vrver=rolling_visibility_segments(ts_b, counts_b,phase_length=phase_length, overlap=None)
    QBERrv=(1-Vrv)/2
    QBERrver=0.5*Vrver
    if secondhalf==True:
        fig2,axes2 = plt.subplots(ncols=2,figsize=(14, 3))
        axes2[0].scatter(tsrv,Vrv,color='tab:blue')
        axes2[0].errorbar(tsrv,Vrv,yerr=Vrver,color='tab:green', alpha=0.1)
        axes2[0].set_title('Plot of Visibility vs Time, binned at Tau')
        axes2[0].set_ylabel('Visibility')
        axes2[0].set_xlabel('Time (s)')
        axes2[0].set_ylim(0.2,1.05)
        axes2[1].errorbar(tsrv,QBERrv,yerr=QBERrver,color='red', alpha=0.1)
        axes2[1].scatter(tsrv,QBERrv,color='tab:green')
        axes2[1].set_title('Plot of QBER vs Time, binned at Tau')
        axes2[1].set_ylabel('QBER')
        axes2[1].set_xlabel('Time (s)')
        axes2[1].set_ylim(0,0.5)
    pb_vis=[taus, Vmax, Vmin]
    bn_b_tau=[ts_b, counts_b, meta]
    roll_vis=[tsrv, Vrv, Vrver,QBERrv,QBERrver]
    values=[pb_vis,bn_b_tau,roll_vis]
    return(values)
    
def lockin_visibility_vs_time(
    ts,
    counts,
    f_ramp: float,
    tau: float,
    step: Optional[float] = None,
    phase_offset: float = 0.0,
    n_min: int = 100,      # minimum total counts per window to accept
    clip: bool = True,     # clip V to [0,1]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate time-dependent visibility V(t) using a sinusoidal fit
    for a known linear phase ramp, and propagate errors to V_err(t).

    Model in each window:
        y(t) ≈ a + b cos(phi(t)) + c sin(phi(t)),
        V = sqrt(b^2 + c^2) / a

    Phase model
    -----------
    phi(t) = 2*pi*f_ramp*(t - t0) + phase_offset    (mod 2*pi)

    Parameters
    ----------
    ts : array_like
        Timestamps in seconds, 1D, monotonically increasing.
    counts : array_like
        Photon counts for each timestamp, same length as ts.
    f_ramp : float
        Phase ramp frequency in Hz (0→2π sweeps per second).
    tau : float
        Window length in seconds for estimating V(t).
    step : float or None, optional
        Step between consecutive windows in seconds.
        If None, use tau/2 (50% overlap).
    phase_offset : float, optional
        Additional constant phase offset in radians.
    n_min : int, optional
        Minimum total counts in a window to accept a visibility estimate.
        Windows with fewer counts get V = NaN, V_err = NaN.
    clip : bool, optional
        If True, clip the visibility into [0, 1].

    Returns
    -------
    t_centers : np.ndarray
        Center time of each window.
    V : np.ndarray
        Estimated visibility in each window (NaN where not enough counts).
    V_err : np.ndarray
        Approximate 1σ error on V in each window (NaN where not enough counts).
    """
    ts = np.asarray(ts, dtype=float)
    y  = np.asarray(counts, dtype=float)

    if ts.ndim != 1 or y.ndim != 1:
        raise ValueError("ts and counts must be 1D arrays.")
    if ts.size != y.size:
        raise ValueError("ts and counts must have the same length.")
    if ts.size < 2:
        raise ValueError("Need at least 2 samples.")
    if tau <= 0:
        raise ValueError("tau must be > 0.")
    if f_ramp <= 0:
        raise ValueError("f_ramp must be > 0.")

    n = ts.size

    # Relative times (only differences matter for the phase)
    t0 = ts[0]
    trel = ts - t0

    # Approximate mean dt for index-based windows
    dts = np.diff(ts)
    dt_pos = dts[dts > 0]
    if dt_pos.size == 0:
        raise ValueError("Timestamps must be strictly increasing.")
    dt_mean = float(np.mean(dt_pos))

    # Number of samples in a tau-window
    K_win = max(1, int(round(tau / dt_mean)))

    # Step in samples
    if step is None:
        step_s = tau / 2.0  # 50% overlap
    else:
        if step <= 0:
            raise ValueError("step must be > 0.")
        step_s = float(step)
    K_step = max(1, int(round(step_s / dt_mean)))

    if n < K_win:
        return np.array([]), np.array([]), np.array([])

    # Compute phase at each sample
    phi = 2.0 * np.pi * f_ramp * trel + phase_offset
    cphi = np.cos(phi)
    sphi = np.sin(phi)

    # Window starts/ends in index space
    last_start = n - K_win
    starts = np.arange(0, last_start + 1, K_step, dtype=int)
    ends = starts + K_win

    t_centers = np.empty_like(starts, dtype=float)
    V = np.full_like(starts, np.nan, dtype=float)
    V_err = np.full_like(starts, np.nan, dtype=float)

    for i, (s, e) in enumerate(zip(starts, ends)):
        y_win = y[s:e]
        if y_win.size == 0:
            continue

        N_tot = np.sum(y_win)
        t_centers[i] = 0.5 * (ts[s] + ts[e - 1])

        if N_tot < n_min:
            # Not enough counts -> unreliable; leave NaNs
            continue

        c_win = cphi[s:e]
        s_win = sphi[s:e]

        # Design matrix: [1, cos(phi), sin(phi)]
        X = np.column_stack((np.ones_like(c_win), c_win, s_win))

        # Least-squares fit: y ≈ a + b cos + c sin
        beta, residuals, rank, svals = np.linalg.lstsq(X, y_win, rcond=None)
        a, b, c = beta

        # Require positive DC level
        if a <= 0:
            continue

        # Visibility
        amp = np.sqrt(b * b + c * c)
        v = amp / a
        if clip:
            v = max(0.0, min(1.0, v))
        V[i] = v

        # --- Error propagation ---
        # Estimate noise variance sigma^2 from residuals
        # residuals is sum of squared residuals if rank == 3, else empty
        if residuals.size > 0 and (y_win.size > X.shape[1]):
            dof = y_win.size - X.shape[1]
            sigma2 = float(residuals[0] / dof)
        else:
            # Fallback: assume sigma^2 ~ mean count (Poisson-ish)
            sigma2 = float(np.mean(y_win)) if np.mean(y_win) > 0 else 1.0

        # Covariance of [a,b,c] for homoscedastic noise
        # Cov = sigma^2 * (X^T X)^(-1)
        XT_X = X.T @ X
        try:
            Cov = sigma2 * np.linalg.inv(XT_X)
        except np.linalg.LinAlgError:
            # Ill-conditioned; skip error estimate
            continue

        # Gradient of V wrt (a,b,c)
        # r = sqrt(b^2 + c^2)
        r2 = b * b + c * c
        if r2 <= 0:
            continue
        r = np.sqrt(r2)
        V_unclipped = r / a  # use unclipped for derivatives

        dV_da = -V_unclipped / a
        dV_db = V_unclipped * b / r2
        dV_dc = V_unclipped * c / r2

        grad = np.array([dV_da, dV_db, dV_dc])

        # Var(V) ≈ grad^T Cov grad
        var_V = float(grad @ Cov @ grad)
        if var_V >= 0:
            V_err[i] = np.sqrt(var_V)

    return t_centers, V, V_err
def lockin_visibility_vs_tau(
    ts,
    counts,
    f_ramp: float,
    n_taus: int = 40,
    tau_max: float = None,
    log_bins: bool = True,
    overlap: Optional[float] = None,   # same semantics as point_binned_visibility
    phase_offset: float = 0.0,
    n_min: int = 100,                  # min counts per window for a valid V
    verbose: bool = True,
    clip=True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For a series of τ values, compute the maximum, minimum, and mean visibility
    across all lock-in windows for that τ.

    This uses the same τ grid logic as point_binned_visibility, but instead of
    max/min over *counts* blocks, it runs the lock-in / sinusoid-fit estimator
    in sliding windows of length τ and then summarizes the resulting V(t).

    Parameters
    ----------
    ts : array_like
        Timestamps in seconds (monotonic, not necessarily uniform).
    counts : array_like
        Counts associated with each timestamp.
    f_ramp : float
        Phase ramp frequency in Hz (0→2π sweeps per second), used by the
        lock-in visibility estimator.
    n_taus, tau_max, log_bins, overlap, verbose :
        Same meaning as in point_binned_visibility.
    phase_offset : float
        Extra constant phase offset (radians) for the lock-in model.
    n_min : int
        Minimum total counts in a window for lockin_visibility_vs_time to
        accept a visibility estimate (windows below this become NaN).

    Returns
    -------
    taus : np.ndarray, shape (M,)
        τ values in seconds.
    vmax : np.ndarray, shape (M,)
        For each τ, the maximum visibility over all windows (NaN if none).
    vmin : np.ndarray, shape (M,)
        For each τ, the minimum visibility over all windows (NaN if none).
    vmean : np.ndarray, shape (M,)
        For each τ, the mean visibility over all finite windows (NaN if none).
    """
    # ---- prepare data (same style as point_binned_visibility)
    ts = np.asarray(ts, dtype=float)
    x  = np.clip(np.nan_to_num(np.asarray(counts, dtype=float), nan=0.0), 0, None)
    n = min(ts.size, x.size)
    ts = ts[:n]; x = x[:n]
    if n < 2:
        raise ValueError("Need at least 2 samples.")

    secs = ts
    duration = float(secs[-1] - secs[0])

    dts = np.diff(secs)
    dt_pos = dts[dts > 0]
    if dt_pos.size == 0:
        raise ValueError("Timestamps are not strictly advancing.")
    dt_mean = float(np.mean(dt_pos))
    dt_std  = float(np.std(dt_pos, ddof=1)) if dt_pos.size > 1 else 0.0
    dt_cv   = dt_std / dt_mean if dt_mean > 0 else np.nan

    if tau_max is None:
        tau_max = 0.5 * duration
    tau_max = max(float(tau_max), dt_mean)

    # ---- choose K values (samples per τ window)
    Kmax = max(1, int(np.floor(tau_max / dt_mean)))
    if n_taus <= 1:
        Ks = np.array([1], dtype=int)
    else:
        if log_bins:
            Ks = np.round(np.logspace(0, np.log10(Kmax), num=n_taus)).astype(int)
        else:
            Ks = np.round(np.linspace(1, Kmax, num=n_taus)).astype(int)
        Ks = np.unique(np.clip(Ks, 1, Kmax))

    taus = Ks * dt_mean

    # ---- validate overlap
    if overlap is not None:
        if not (0.0 <= overlap < 1.0):
            raise ValueError("overlap must be in [0, 1) or None.")

    vmax = np.full_like(taus, np.nan, dtype=float)
    vmin = np.full_like(taus, np.nan, dtype=float)
    vmean = np.full_like(taus, np.nan, dtype=float)

    # simple per-τ diagnostics (if you want to extend later)
    per_tau_diag: List[Dict[str, Any]] = []

    # ---- main loop over τ
    for i, K in enumerate(Ks):
        tau = K * dt_mean

        # choose step from overlap in *samples*, then convert to seconds
        if overlap is None:
            stride_samples = K  # non-overlapping
        else:
            stride_samples = max(1, int(round(K * (1.0 - overlap))))
            stride_samples = min(stride_samples, K)
        step = stride_samples * dt_mean
        overlap_eff = 1.0 - (stride_samples / float(K))

        # use lockin_visibility_vs_time with this τ and step
        t_centers, V_tau, V_err_tau = lockin_visibility_vs_time(
            ts,
            x,
            f_ramp=f_ramp,
            tau=tau,
            step=step,
            phase_offset=phase_offset,
            n_min=n_min,
            clip=clip,
        )

        # summarize visibilities for this τ
        finite = np.isfinite(V_tau)
        n_win = int(np.sum(finite))
        if n_win == 0:
            vmax[i] = np.nan
            vmin[i] = np.nan
            vmean[i] = np.nan
        else:
            V_use = V_tau[finite]
            vmax[i] = float(np.max(V_use))
            vmin[i] = float(np.min(V_use))
            vmean[i] = float(np.mean(V_use))

        if verbose:
            print(f"τ[{i}] = {tau:.3e} s (K={K:>4d}, overlap_eff={overlap_eff:.2f}) -> "
                  f"{n_win:4d} windows, V_max={vmax[i]: .4f}, "
                  f"V_min={vmin[i]: .4f}, V_mean={vmean[i]: .4f}")

        # optional diagnostics collection
        per_tau_diag.append({
            "K": int(K),
            "tau": tau,
            "stride_samples": int(stride_samples),
            "overlap_eff": float(overlap_eff),
            "n_windows": n_win,
        })

    # if you later want diagnostics, you can return them too or stash them globally
    return taus, vmax, vmin, vmean
import numpy as np
from typing import Optional, Tuple, List, Dict, Any


def lockin_visibility_vs_time_fast(
    ts,
    counts,
    f_ramp: float,
    tau: float,
    step: Optional[float] = None,
    phase_offset: float = 0.0,
    n_min: int = 100,       # minimum total counts per window
    clip: bool = True,      # clip V to [0, 1]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast lock-in visibility vs time using precomputed cumulative sums and a 3x3 normal-equation solve
    per window (no per-window array allocations, no np.linalg.lstsq).

    Model per window:
        y_j ≈ a + b cos(phi_j) + c sin(phi_j)
        V = sqrt(b^2 + c^2) / a

    Returns
    -------
    t_centers : np.ndarray
    V         : np.ndarray
    V_err     : np.ndarray  (approx 1σ via delta-method, homoscedastic residual variance)
    """
    ts = np.asarray(ts, dtype=float)
    y  = np.asarray(counts, dtype=float)

    if ts.ndim != 1 or y.ndim != 1:
        raise ValueError("ts and counts must be 1D arrays.")
    if ts.size != y.size:
        raise ValueError("ts and counts must have the same length.")
    if ts.size < 2:
        raise ValueError("Need at least 2 samples.")
    if tau <= 0:
        raise ValueError("tau must be > 0.")
    if f_ramp <= 0:
        raise ValueError("f_ramp must be > 0.")
    if not np.all(np.diff(ts) > 0):
        raise ValueError("times must be strictly increasing.")

    n = ts.size
    t0 = ts[0]
    trel = ts - t0

    dts = np.diff(ts)
    dt_pos = dts[dts > 0]
    if dt_pos.size == 0:
        raise ValueError("Timestamps must be strictly increasing.")
    dt_mean = float(np.mean(dt_pos))

    # Window + step in samples
    K_win = max(1, int(round(tau / dt_mean)))
    if step is None:
        step_s = tau / 2.0
    else:
        if step <= 0:
            raise ValueError("step must be > 0.")
        step_s = float(step)
    K_step = max(1, int(round(step_s / dt_mean)))

    if n < K_win:
        return np.array([]), np.array([]), np.array([])

    # Phase reference
    phi = 2.0 * np.pi * f_ramp * trel + phase_offset
    c = np.cos(phi)
    s = np.sin(phi)

    # Precompute cumulative sums for sufficient statistics
    def cumsum0(a: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(a, dtype=float)))

    Cy   = cumsum0(y)
    Cy2  = cumsum0(y * y)
    Cc   = cumsum0(c)
    Cs   = cumsum0(s)
    Ccc  = cumsum0(c * c)
    Css  = cumsum0(s * s)
    Ccs  = cumsum0(c * s)
    Cyc  = cumsum0(y * c)
    Cys  = cumsum0(y * s)

    # Windows
    last_start = n - K_win
    starts = np.arange(0, last_start + 1, K_step, dtype=int)
    ends = starts + K_win

    # Vectorized centers (no loop)
    t_centers = 0.5 * (ts[starts] + ts[ends - 1])

    V = np.full(starts.size, np.nan, dtype=float)
    V_err = np.full(starts.size, np.nan, dtype=float)

    p = 3  # parameters a,b,c
    I3 = np.eye(3, dtype=float)

    for i, (st, en) in enumerate(zip(starts, ends)):
        # Fast window sums via prefix sums
        sumy  = Cy[en]  - Cy[st]
        if sumy < n_min:
            continue

        m     = float(en - st)  # number of samples in window
        sumc  = Cc[en]  - Cc[st]
        sums  = Cs[en]  - Cs[st]
        sumcc = Ccc[en] - Ccc[st]
        sumss = Css[en] - Css[st]
        sumcs = Ccs[en] - Ccs[st]
        sumyc = Cyc[en] - Cyc[st]
        sumys = Cys[en] - Cys[st]
        sumy2 = Cy2[en] - Cy2[st]

        # Normal equations: G beta = h
        # G = X^T X, h = X^T y, where X = [1, c, s]
        G = np.array([[m,    sumc,  sums],
                      [sumc, sumcc, sumcs],
                      [sums, sumcs, sumss]], dtype=float)
        h = np.array([sumy, sumyc, sumys], dtype=float)

        # Solve for beta = [a,b,c]
        try:
            beta = np.linalg.solve(G, h)
        except np.linalg.LinAlgError:
            continue

        a, b, cc_ = beta
        if a <= 0:
            continue

        r2 = b * b + cc_ * cc_
        if r2 <= 0:
            continue
        r = float(np.sqrt(r2))

        v_unclipped = r / a
        v = v_unclipped
        if clip:
            v = max(0.0, min(1.0, v))
        V[i] = v

        # --- Error (approx): homoscedastic residual variance + delta method ---
        # SSE = y^T y - beta^T (X^T y) = sumy2 - beta·h
        dof = m - p
        if dof > 0:
            SSE = float(sumy2 - beta.dot(h))
            # numerical guard
            if SSE < 0:
                SSE = 0.0
            sigma2 = SSE / dof

            # grad of V wrt [a,b,c]
            # Use unclipped V for derivatives
            dV_da = -v_unclipped / a
            dV_db = v_unclipped * b / r2
            dV_dc = v_unclipped * cc_ / r2
            grad = np.array([dV_da, dV_db, dV_dc], dtype=float)

            # var(V) = sigma^2 * grad^T (G^-1) grad
            # compute z = G^-1 grad via solve (faster than explicit inverse)
            try:
                z = np.linalg.solve(G, grad)
                varV = float(sigma2 * grad.dot(z))
                if varV >= 0:
                    V_err[i] = float(np.sqrt(varV))
            except np.linalg.LinAlgError:
                pass

    return t_centers, V, V_err


def lockin_visibility_vs_tau_fast(
    ts,
    counts,
    f_ramp: float,
    n_taus: int = 40,
    tau_max: float = None,
    log_bins: bool = True,
    overlap: Optional[float] = None,   # same semantics as point_binned_visibility
    phase_offset: float = 0.0,
    n_min: int = 100,                  # min counts per window for a valid V
    verbose: bool = True,
    clip: bool = True,      # clip V to [0, 1]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Same as lockin_visibility_vs_tau, but uses lockin_visibility_vs_time_fast.

    Returns
    -------
    taus, vmax, vmin, vmean
    """
    ts = np.asarray(ts, dtype=float)
    x  = np.clip(np.nan_to_num(np.asarray(counts, dtype=float), nan=0.0), 0, None)
    n = min(ts.size, x.size)
    ts = ts[:n]; x = x[:n]
    if n < 2:
        raise ValueError("Need at least 2 samples.")

    secs = ts
    duration = float(secs[-1] - secs[0])

    dts = np.diff(secs)
    dt_pos = dts[dts > 0]
    if dt_pos.size == 0:
        raise ValueError("Timestamps are not strictly advancing.")
    dt_mean = float(np.mean(dt_pos))

    if tau_max is None:
        tau_max = 0.5 * duration
    tau_max = max(float(tau_max), dt_mean)

    # Choose K values (samples per τ window)
    Kmax = max(1, int(np.floor(tau_max / dt_mean)))
    if n_taus <= 1:
        Ks = np.array([1], dtype=int)
    else:
        if log_bins:
            Ks = np.round(np.logspace(0, np.log10(Kmax), num=n_taus)).astype(int)
        else:
            Ks = np.round(np.linspace(1, Kmax, num=n_taus)).astype(int)
        Ks = np.unique(np.clip(Ks, 1, Kmax))

    taus = Ks * dt_mean

    if overlap is not None and not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1) or None.")

    vmax  = np.full_like(taus, np.nan, dtype=float)
    vmin  = np.full_like(taus, np.nan, dtype=float)
    vmean = np.full_like(taus, np.nan, dtype=float)

    for i, K in enumerate(Ks):
        tau = float(K * dt_mean)

        if overlap is None:
            stride_samples = K
        else:
            stride_samples = max(1, int(round(K * (1.0 - overlap))))
            stride_samples = min(stride_samples, K)

        step = float(stride_samples * dt_mean)
        overlap_eff = 1.0 - (stride_samples / float(K))

        t_centers, V_tau, _ = lockin_visibility_vs_time_fast(
            ts,
            x,
            f_ramp=f_ramp,
            tau=tau,
            step=step,
            phase_offset=phase_offset,
            n_min=n_min,
            clip=clip
        )

        finite = np.isfinite(V_tau)
        n_win = int(np.sum(finite))
        if n_win > 0:
            V_use = V_tau[finite]
            vmax[i]  = float(np.max(V_use))
            vmin[i]  = float(np.min(V_use))
            vmean[i] = float(np.mean(V_use))

        if verbose:
            print(
                f"τ[{i}] = {tau:.3e} s (K={K:>4d}, overlap_eff={overlap_eff:.2f}) -> "
                f"{n_win:4d} windows, V_max={vmax[i]: .4f}, "
                f"V_min={vmin[i]: .4f}, V_mean={vmean[i]: .4f}"
            )

    return taus, vmax, vmin, vmean
def VisLockAnalysis(ms, i, 
                    f_ramp,
                    n_taus,
                    tau_max,
                    log_bins,
                    overlap,
                    phase_offset,
                    n_min,
                    verbose):
    print(i,ms[i]['name'])
    ts=timestamps_to_elapsed_seconds(ms[i]['data']['timestamps'])
    counts=ms[i]['data']['counts']
    taus, vmax, vmin, vmean = lockin_visibility_vs_tau(
        ts,
        counts,
        f_ramp,
        n_taus,
        tau_max,
        log_bins,
        overlap,
        phase_offset,
        n_min,
        verbose)
    return(taus, vmax, vmin, vmean)
def lockin_bins_and_counts_vs_time(
    ts,
    counts,
    f_ramp: float,          # kept for identical signature (not used)
    tau: float,
    step: Optional[float] = None,
    phase_offset: float = 0.0,  # kept for identical signature (not used)
    n_min: int = 100,
    clip: bool = True,      # kept for identical signature (not used)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the time bins and total counts used by lockin_visibility_vs_time_fast.

    This function uses the *same windowing* as lockin_visibility_vs_time_fast,
    but instead of estimating visibility, it outputs the total counts in each bin.

    Parameters
    ----------
    Same as lockin_visibility_vs_time_fast (unused parameters are accepted
    to keep the signature identical).

    Returns
    -------
    t_centers : np.ndarray
        Center time of each τ-window.
    counts_per_bin : np.ndarray
        Total counts in each window (NaN if below n_min).
    n_samples_per_bin : np.ndarray
        Number of samples contributing to each window.
    """
    ts = np.asarray(ts, dtype=float)
    y  = np.asarray(counts, dtype=float)

    if ts.ndim != 1 or y.ndim != 1:
        raise ValueError("ts and counts must be 1D arrays.")
    if ts.size != y.size:
        raise ValueError("ts and counts must have the same length.")
    if ts.size < 2:
        raise ValueError("Need at least 2 samples.")
    if tau <= 0:
        raise ValueError("tau must be > 0.")
    if not np.all(np.diff(ts) > 0):
        raise ValueError("times must be strictly increasing.")

    n = ts.size

    # Mean dt
    dts = np.diff(ts)
    dt_pos = dts[dts > 0]
    if dt_pos.size == 0:
        raise ValueError("Timestamps must be strictly increasing.")
    dt_mean = float(np.mean(dt_pos))

    # Window + step in samples
    K_win = max(1, int(round(tau / dt_mean)))
    if step is None:
        step_s = tau / 2.0
    else:
        if step <= 0:
            raise ValueError("step must be > 0.")
        step_s = float(step)
    K_step = max(1, int(round(step_s / dt_mean)))

    if n < K_win:
        return np.array([]), np.array([]), np.array([])

    # Prefix sum of counts
    Cy = np.concatenate(([0.0], np.cumsum(y, dtype=float)))

    # Window starts / ends
    last_start = n - K_win
    starts = np.arange(0, last_start + 1, K_step, dtype=int)
    ends = starts + K_win

    # Outputs
    t_centers = 0.5 * (ts[starts] + ts[ends - 1])
    counts_per_bin = np.full(starts.size, np.nan, dtype=float)
    n_samples_per_bin = np.full(starts.size, K_win, dtype=int)

    for i, (st, en) in enumerate(zip(starts, ends)):
        sumy = Cy[en] - Cy[st]
        if sumy >= n_min:
            counts_per_bin[i] = float(sumy)
        else:
            counts_per_bin[i] = np.nan

    return t_centers, counts_per_bin, n_samples_per_bin
def AnalyseMeasurement2(ms,i,n_taus,tau,phase_length, tau_max, overlap, phase_offset, n_min, verbose, firsthalf=True,secondhalf=True, clip=False):
    f_ramp=1/phase_length
    log_bins=True
    print(i,ms[i]['name'])
    ts=timestamps_to_elapsed_seconds(ms[i]['data']['timestamps'])
    cts=ms[i]['data']['counts']
    taus, Vmax, Vmin, Vmean = lockin_visibility_vs_tau(
        ts,
        cts,
        f_ramp,
        n_taus,
        tau_max,
        log_bins,
        overlap,
        phase_offset,
        n_min,
        verbose,
        clip)
    if firsthalf==True:
        fig1,axes1 = plt.subplots(ncols=2,figsize=(14, 3))
        axes1[0].semilogx(taus,Vmax,label='Vis (max)')
        axes1[0].semilogx(taus,Vmin,label='Vis (min)')
        axes1[0].semilogx(taus,Vmean,label='Vis (mean)')
        axes1[0].set_title('Plot of Visibility V vs Tau')
        axes1[0].set_ylabel('Visibility')
        axes1[0].set_xlabel('Averaging Time Tau (s)')
        axes1[0].axvline(tau, color='gray', linestyle='--')
        axes1[0].legend()
        axes1[0].set_ylim(0,1.05)
    ts_b, counts_b, smp_b = lockin_bins_and_counts_vs_time(ts, cts,f_ramp=f_ramp, tau=tau, n_min=n_min, clip=clip)
    if firsthalf==True:
        axes1[1].plot(ts_b,counts_b,color='tab:orange')
        axes1[1].set_title('Plot of Counts vs Time, binned at Tau')
        axes1[1].set_ylabel('Counts')
        axes1[1].set_xlabel('Time (s)')
        axes1[1].set_ylim(0,np.nanmax(counts_b))
    tsrv, Vrv, Vrver=lockin_visibility_vs_time(ts, cts,f_ramp, tau, n_min=n_min, clip=clip)
    QBERrv=(1-Vrv)/2
    QBERrver=0.5*Vrver
    if secondhalf==True:
        fig2,axes2 = plt.subplots(ncols=2,figsize=(14, 3))
        axes2[0].scatter(tsrv,Vrv,color='tab:blue')
        axes2[0].errorbar(tsrv,Vrv,yerr=Vrver,color='tab:green', alpha=0.1)
        axes2[0].set_title('Plot of Visibility vs Time, binned at Tau')
        axes2[0].set_ylabel('Visibility')
        axes2[0].set_xlabel('Time (s)')
        axes2[0].set_ylim(0.2,1.05)
        axes2[1].errorbar(tsrv,QBERrv,yerr=QBERrver,color='red', alpha=0.1)
        axes2[1].scatter(tsrv,QBERrv,color='tab:green')
        axes2[1].set_title('Plot of QBER vs Time, binned at Tau')
        axes2[1].set_ylabel('QBER')
        axes2[1].set_xlabel('Time (s)')
        axes2[1].set_ylim(0,0.5)
    pb_vis=[taus, Vmax, Vmin]
    bn_b_tau=[ts_b, counts_b, smp_b]
    roll_vis=[tsrv, Vrv, Vrver,QBERrv,QBERrver]
    values=[pb_vis,bn_b_tau,roll_vis]
    return(values)
def AnalyseMeasurement2_fast(ms,i,n_taus,tau,phase_length, tau_max, overlap, phase_offset, n_min, verbose, firsthalf=True,secondhalf=True, clip=False):
    f_ramp=1/phase_length
    log_bins=True
    print(i,ms[i]['name'])
    ts=timestamps_to_elapsed_seconds(ms[i]['data']['timestamps'])
    cts=ms[i]['data']['counts']
    taus, Vmax, Vmin, Vmean = lockin_visibility_vs_tau_fast(
        ts,
        cts,
        f_ramp,
        n_taus,
        tau_max,
        log_bins,
        overlap,
        phase_offset,
        n_min,
        verbose,
        clip)
    if firsthalf==True:
        fig1,axes1 = plt.subplots(ncols=2,figsize=(14, 3))
        axes1[0].semilogx(taus,Vmax,label='Vis (max)')
        axes1[0].semilogx(taus,Vmin,label='Vis (min)')
        axes1[0].semilogx(taus,Vmean,label='Vis (mean)')
        axes1[0].set_title('Plot of Visibility V vs Tau')
        axes1[0].set_ylabel('Visibility')
        axes1[0].set_xlabel('Averaging Time Tau (s)')
        axes1[0].axvline(tau, color='gray', linestyle='--')
        axes1[0].legend()
        axes1[0].set_ylim(0,1.05)
    ts_b, counts_b, smp_b = lockin_bins_and_counts_vs_time(ts, cts,f_ramp=f_ramp, tau=tau, n_min=n_min, clip=clip)
    if firsthalf==True:
        axes1[1].plot(ts_b,counts_b,color='tab:orange')
        axes1[1].set_title('Plot of Counts vs Time, binned at Tau')
        axes1[1].set_ylabel('Counts')
        axes1[1].set_xlabel('Time (s)')
        axes1[1].set_ylim(0,np.nanmax(counts_b))
    tsrv, Vrv, Vrver=lockin_visibility_vs_time_fast(ts, cts,f_ramp, tau, n_min=n_min, clip=clip)
    QBERrv=(1-Vrv)/2
    QBERrver=0.5*Vrver
    if secondhalf==True:
        fig2,axes2 = plt.subplots(ncols=2,figsize=(14, 3))
        axes2[0].scatter(tsrv,Vrv,color='tab:blue')
        axes2[0].errorbar(tsrv,Vrv,yerr=Vrver,color='tab:green', alpha=0.1)
        axes2[0].set_title('Plot of Visibility vs Time, binned at Tau')
        axes2[0].set_ylabel('Visibility')
        axes2[0].set_xlabel('Time (s)')
        axes2[0].set_ylim(0.2,1.05)
        axes2[1].errorbar(tsrv,QBERrv,yerr=QBERrver,color='red', alpha=0.1)
        axes2[1].scatter(tsrv,QBERrv,color='tab:green')
        axes2[1].set_title('Plot of QBER vs Time, binned at Tau')
        axes2[1].set_ylabel('QBER')
        axes2[1].set_xlabel('Time (s)')
        axes2[1].set_ylim(0,0.5)
    pb_vis=[taus, Vmax, Vmin]
    bn_b_tau=[ts_b, counts_b, smp_b]
    roll_vis=[tsrv, Vrv, Vrver,QBERrv,QBERrver]
    values=[pb_vis,bn_b_tau,roll_vis]
    return(values)
# ---------------------------
# Utilities
# ---------------------------

def _cumsum0(a: np.ndarray) -> np.ndarray:
    """Cumulative sum with a leading zero for O(1) range-sum queries."""
    return np.concatenate(([0.0], np.cumsum(a, dtype=float)))


def _h2(p: np.ndarray) -> np.ndarray:
    """Binary entropy (base-2)."""
    p = np.asarray(p, float)
    out = np.zeros_like(p)
    m = (p > 0) & (p < 1)
    out[m] = -(p[m]*np.log2(p[m]) + (1-p[m])*np.log2(1-p[m]))
    return out

from dataclasses import dataclass

@dataclass
class VisibilityTauResult:
    taus: np.ndarray
    n_windows: np.ndarray
    V_max: np.ndarray
    V_min: np.ndarray
    V_mean: np.ndarray
    V_median: np.ndarray
    V_p10: np.ndarray
    V_p90: np.ndarray
    expected_counts_per_window: np.ndarray
    meta: Dict[str, Any]


# ---------------------------
# Core: fast lock-in visibility vs time
# ---------------------------

def lockin_visibility_vs_time_fast(
    ts: np.ndarray,
    counts: np.ndarray,
    f_ramp: float,
    tau: float,
    step: Optional[float] = None,
    phase_offset: float = 0.0,
    n_min: float = 0.0,
    clip: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fast lock-in effective visibility in sliding windows.

    Visibility per window:
        V = 2 * sqrt((sum y cos phi)^2 + (sum y sin phi)^2) / (sum y)
    where phi(t) = 2*pi*f_ramp*(t - t0) + phase_offset.

    Returns
    -------
    t_centers : window-center timestamps
    V : visibility estimates per window (NaN if window rejected by n_min or sum y == 0)
    diag : diagnostics dict
    """
    ts = np.asarray(ts, float)
    y  = np.clip(np.nan_to_num(np.asarray(counts, float), nan=0.0), 0, None)
    n = min(ts.size, y.size)
    ts = ts[:n]; y = y[:n]
    if n < 2:
        raise ValueError("Need at least 2 samples.")

    dt = float(np.mean(np.diff(ts)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Bad timestamps / dt.")

    if step is None:
        step = tau / 2.0
    if tau <= 0 or step <= 0:
        raise ValueError("tau and step must be > 0.")

    K = max(2, int(round(tau / dt)))
    S = max(1, int(round(step / dt)))
    if K > n:
        return np.array([]), np.array([]), {"dt": dt, "K": K, "S": S, "n_windows": 0}

    # Reference phase
    t0 = ts[0]
    phi = 2.0*np.pi*f_ramp*(ts - t0) + phase_offset
    c = np.cos(phi)
    s = np.sin(phi)

    # Prefix sums
    Cy  = _cumsum0(y)
    Cyc = _cumsum0(y * c)
    Cys = _cumsum0(y * s)

    starts = np.arange(0, n - K + 1, S, dtype=int)
    ends = starts + K

    sumy  = Cy[ends]  - Cy[starts]
    sumyc = Cyc[ends] - Cyc[starts]
    sumys = Cys[ends] - Cys[starts]

    V = np.full(starts.size, np.nan, dtype=float)

    good = (sumy >= n_min) & (sumy > 0)
    if np.any(good):
        Vg = 2.0*np.sqrt(sumyc[good]**2 + sumys[good]**2) / sumy[good]
        if clip:
            Vg = np.clip(Vg, 0.0, 1.0)
        V[good] = Vg

    t_centers = ts[starts + (K//2)]

    diag = {
        "dt": dt, "K": int(K), "S": int(S),
        "tau": float(K*dt),
        "step": float(S*dt),
        "n_windows_total": int(starts.size),
        "n_windows_used": int(np.sum(np.isfinite(V))),
        "mean_counts_per_window_used": float(np.nanmean(sumy[good])) if np.any(good) else np.nan,
    }
    return t_centers, V, diag


# ---------------------------
# Sweep tau: Vmax / Vmin / Vmean (+ robust stats)
# ---------------------------

def lockin_visibility_summary_vs_tau(
    ts: np.ndarray,
    counts: np.ndarray,
    f_ramp: float,
    taus: Optional[np.ndarray] = None,
    n_taus: int = 25,
    tau_min: Optional[float] = None,
    tau_max: Optional[float] = None,
    log_bins: bool = True,
    cycles_min: int = 10,     # "trustworthy region" lower bound: tau >= cycles_min / f_ramp
    overlap: float = 0.5,     # step = tau*(1-overlap)
    phase_offset: float = 0.0,
    n_min: float = 50.0,
    clip: bool = False,
) -> VisibilityTauResult:
    """
    Sweep over taus, compute V(t; tau) in sliding windows, and summarize per tau:
    max/min/mean/median + 10–90% band and number of windows.

    If `taus` not provided, creates a grid between tau_min and tau_max.
    """
    ts = np.asarray(ts, float)
    y  = np.clip(np.nan_to_num(np.asarray(counts, float), nan=0.0), 0, None)
    n = min(ts.size, y.size)
    ts = ts[:n]; y = y[:n]
    if n < 2:
        raise ValueError("Need at least 2 samples.")

    dt = float(np.mean(np.diff(ts)))
    duration = float(ts[-1] - ts[0])
    rate = float(np.sum(y) / duration)

    if tau_min is None:
        # at least one cycle and at least one sample
        tau_min = max(dt, 1.0 / f_ramp)
    if tau_max is None:
        tau_max = min(0.5 * duration, 0.2)  # you can override; 0.2 s is a sane default for 1 kHz ramps

    tau_min = float(max(tau_min, dt))
    tau_max = float(max(tau_max, tau_min))

    if taus is None:
        if n_taus <= 1:
            taus = np.array([tau_min], dtype=float)
        else:
            if log_bins:
                taus = np.logspace(np.log10(tau_min), np.log10(tau_max), n_taus)
            else:
                taus = np.linspace(tau_min, tau_max, n_taus)
        taus = np.unique(taus.astype(float))

    # Step is set from overlap fraction
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0,1).")
    steps = taus * (1.0 - overlap)

    V_max = np.full(taus.size, np.nan)
    V_min = np.full(taus.size, np.nan)
    V_mean = np.full(taus.size, np.nan)
    V_median = np.full(taus.size, np.nan)
    V_p10 = np.full(taus.size, np.nan)
    V_p90 = np.full(taus.size, np.nan)
    n_windows = np.zeros(taus.size, dtype=int)

    for i, (tau, step) in enumerate(zip(taus, steps)):
        tcent, Vt, _ = lockin_visibility_vs_time_fast(
            ts, y, f_ramp=f_ramp, tau=float(tau), step=float(step),
            phase_offset=phase_offset, n_min=n_min, clip=clip
        )
        Vt = Vt[np.isfinite(Vt)]
        n_windows[i] = int(Vt.size)
        if Vt.size == 0:
            continue
        V_max[i] = float(np.max(Vt))
        V_min[i] = float(np.min(Vt))
        V_mean[i] = float(np.mean(Vt))
        V_median[i] = float(np.median(Vt))
        V_p10[i] = float(np.quantile(Vt, 0.10))
        V_p90[i] = float(np.quantile(Vt, 0.90))

    expected_counts = rate * taus
    meta = {
        "dt": dt,
        "duration_s": duration,
        "rate_counts_per_s": rate,
        "cycles_min": cycles_min,
        "trust_tau_min_s": cycles_min / f_ramp,
        "overlap": overlap,
        "n_min_counts_per_window": n_min,
        "clip": clip,
    }
    return VisibilityTauResult(
        taus=taus,
        n_windows=n_windows,
        V_max=V_max,
        V_min=V_min,
        V_mean=V_mean,
        V_median=V_median,
        V_p10=V_p10,
        V_p90=V_p90,
        expected_counts_per_window=expected_counts,
        meta=meta
    )


# ---------------------------
# Choose operating tau + estimate "knee"
# ---------------------------

def choose_operating_tau(
    res: VisibilityTauResult,
    f_ramp: float,
    cycles_min: int = 10,
    within_frac: float = 0.98,
    use: str = "median",
) -> float:
    """
    Choose smallest tau >= cycles_min/f_ramp with statistic within `within_frac` of plateau (max in that region).
    `use` can be 'median' or 'mean'.
    """
    taus = res.taus
    stat = res.V_median if use == "median" else res.V_mean

    trust = (taus >= (cycles_min / f_ramp)) & np.isfinite(stat)
    if not np.any(trust):
        return float("nan")
    plateau = float(np.max(stat[trust]))
    ok = trust & (stat >= within_frac * plateau)
    if not np.any(ok):
        return float("nan")
    return float(np.min(taus[ok]))


def estimate_knee_tau(
    res: VisibilityTauResult,
    f_ramp: float,
    cycles_min: int = 10,
    drop_frac: float = 0.95,
    use: str = "median",
) -> float:
    """
    Find first tau (>= cycles_min/f_ramp) where statistic falls below drop_frac * plateau.
    """
    taus = res.taus
    stat = res.V_median if use == "median" else res.V_mean

    trust = (taus >= (cycles_min / f_ramp)) & np.isfinite(stat)
    if not np.any(trust):
        return float("nan")
    plateau = float(np.max(stat[trust]))
    thresh = drop_frac * plateau
    for i in np.where(trust)[0]:
        if stat[i] < thresh:
            return float(taus[i])
    return float("nan")


# ---------------------------
# Plotting helpers
# ---------------------------

def plot_visibility_vs_tau(res: VisibilityTauResult, title: str = "Effective Visibility vs Tau") -> None:
    taus = res.taus
    m = np.isfinite(res.V_mean)

    plt.figure(figsize=(9, 4))
    plt.plot(taus[m], res.V_mean[m], label="Vmean")
    plt.plot(taus[m], res.V_median[m], label="Vmedian")
    plt.plot(taus[m], res.V_max[m], label="Vmax")
    plt.plot(taus[m], res.V_min[m], label="Vmin")
    plt.fill_between(taus[m], res.V_p10[m], res.V_p90[m], alpha=0.2, label="10–90% band")
    plt.xscale("log")
    plt.ylim(0, 2)
    plt.xlabel("Tau (s)")
    plt.ylabel("Effective visibility")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_qber_and_secret_proxy(res: VisibilityTauResult, title_prefix: str = "") -> None:
    taus = res.taus

    # QBER mapping from visibility; clip to [0,1] for physical mapping
    Vmax_c = np.clip(res.V_max, 0, 1)
    Vmean_c = np.clip(res.V_mean, 0, 1)
    Vmin_c  = np.clip(res.V_min,  0, 1)
    Qmax = 0.5 * (1 - Vmax_c)
    Qmean = 0.5 * (1 - Vmean_c)
    Qworst = 0.5 * (1 - Vmin_c)

    m = np.isfinite(Qmean)

    plt.figure(figsize=(9, 4))
    plt.plot(taus[m], Qmax[m], label="QBER from Vmax")
    plt.plot(taus[m], Qmean[m], label="QBER from Vmean")
    plt.plot(taus[m], Qworst[m], label="QBER from Vmin")
    plt.xscale("log")
    plt.ylim(0, 0.5)
    plt.xlabel("Tau (s)")
    plt.ylabel("Estimated QBER")
    plt.title((title_prefix + " " if title_prefix else "") + "QBER vs Tau (from visibility)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Illustrative secret fraction proxy (NOT TF-QKD key rate; just sanity proxy)
    s_max  = np.maximum(0.0, 1.0 - 2.0*_h2(Qmax))
    s_mean  = np.maximum(0.0, 1.0 - 2.0*_h2(Qmean))
    s_worst = np.maximum(0.0, 1.0 - 2.0*_h2(Qworst))

    m2 = np.isfinite(s_mean)

    plt.figure(figsize=(9, 4))
    plt.plot(taus[m2], s_max[m2], label="Secret fraction proxy (from Vmax)")
    plt.plot(taus[m2], s_mean[m2], label="Secret fraction proxy (from Vmean)")
    plt.plot(taus[m2], s_worst[m2], label="Secret fraction proxy (from Vmin)")
    plt.xscale("log")
    plt.ylim(0, 1.0)
    plt.xlabel("Tau (s)")
    plt.ylabel("Illustrative secret fraction")
    plt.title((title_prefix + " " if title_prefix else "") + "Secret-fraction proxy vs Tau")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_visibility_and_qber_vs_tau(
    res: "VisibilityTauResult",
    title: str = "Visibility and QBER vs Tau",
    vis_ylim: tuple = (0.0, 1.01),
    qber_ylim: tuple = (0.0, 0.5),
) -> None:
    taus = res.taus

    # ---- Visibility mask
    mV = np.isfinite(res.V_mean)

    # ---- QBER mapping from visibility; clip to [0,1] for physical mapping
    Vmax_c  = np.clip(res.V_max,  0.0, 1.0)
    Vmean_c = np.clip(res.V_mean, 0.0, 1.0)
    Vmin_c  = np.clip(res.V_min,  0.0, 1.0)

    Qmax   = 0.5 * (1.0 - Vmax_c)
    Qmean  = 0.5 * (1.0 - Vmean_c)
    Qworst = 0.5 * (1.0 - Vmin_c)

    mQ = np.isfinite(Qmean)

    # ---- Single figure with two subplots
    fig, (axV, axQ) = plt.subplots(1, 2, figsize=(14, 3), sharex=False)

    # ========== Left: Visibility vs tau ==========
    axV.plot(taus[mV], res.V_mean[mV],   label="Vmean")
    axV.plot(taus[mV], res.V_median[mV], label="Vmedian")
    axV.plot(taus[mV], res.V_max[mV],    label="Vmax")
    axV.plot(taus[mV], res.V_min[mV],    label="Vmin")
    axV.fill_between(taus[mV], res.V_p10[mV], res.V_p90[mV], alpha=0.2, label="10–90% band")

    axV.set_xscale("log")
    axV.set_ylim(*vis_ylim)
    axV.set_xlabel("Tau (s)")
    axV.set_ylabel("Effective visibility")
    axV.set_title("Visibility vs Tau")
    axV.grid(True, alpha=0.25)
    axV.legend()

    # ========== Right: QBER vs tau ==========
    axQ.plot(taus[mQ], Qmax[mQ],   label="QBER from Vmax")
    axQ.plot(taus[mQ], Qmean[mQ],  label="QBER from Vmean")
    axQ.plot(taus[mQ], Qworst[mQ], label="QBER from Vmin")

    axQ.set_xscale("log")
    axQ.set_ylim(*qber_ylim)
    axQ.set_xlabel("Tau (s)")
    axQ.set_ylabel("Estimated QBER")
    axQ.set_title("QBER vs Tau (from visibility)")
    axQ.grid(True, alpha=0.25)
    axQ.legend()

    # ---- Overall title + layout
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

from matplotlib.gridspec import GridSpec

def plot_visibility_time_series_at_tau(
    ts: np.ndarray,
    counts: np.ndarray,
    f_ramp: float,
    tau: float,
    overlap: float = 0.5,
    phase_offset: float = 0.0,
    n_min: float = 50.0,
    clip_for_qber: bool = True,
    max_points: int = 5000,
    hist_bins: int = 60,
) -> List[Any]:
    """
    Single figure:
      - Left:  V(t) with normalized sideways histogram
      - Right: QBER(t) with normalized sideways histogram

    Returns the same list structure as your original:
      [ [tcent, Vt, diag], [tcent_ds, Vt_ds], [tcent_ds, Q_ds] ]
    """
    # step from overlap
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0,1).")
    step = tau * (1.0 - overlap)

    # Compute visibility time series (no clipping here; we clip only for QBER mapping)
    tcent, Vt, diag = lockin_visibility_vs_time_fast(
        ts, counts, f_ramp=f_ramp, tau=float(tau), step=float(step),
        phase_offset=phase_offset, n_min=float(n_min), clip=False
    )

    # Downsample for display
    if tcent.size > max_points:
        ds = int(np.ceil(tcent.size / max_points))
    else:
        ds = 1

    t_plot = tcent[::ds]
    V_plot = Vt[::ds]

    # QBER from visibility
    Vc = np.clip(Vt, 0.0, 1.0) if clip_for_qber else Vt
    Q = 0.5 * (1.0 - Vc)
    Q_plot = Q[::ds]

    # ---- Build one figure: [V time + V hist] [Q time + Q hist]
    fig = plt.figure(figsize=(14, 3))
    gs = GridSpec(
        1, 5,
        width_ratios=[4, 1, 0.6, 4, 1],  # <-- spacer column
        wspace=0.04
    )


    axV  = fig.add_subplot(gs[0, 0])
    axVh = fig.add_subplot(gs[0, 1], sharey=axV)
    
    # gs[0, 2] is intentionally empty (spacer)
    
    axQ  = fig.add_subplot(gs[0, 3])
    axQh = fig.add_subplot(gs[0, 4], sharey=axQ)


    # ---- Visibility time series
    axV.plot(t_plot, V_plot, lw=1)
    axV.set_xlabel("Time (s)")
    axV.set_ylabel(f"V(t) at τ≈{tau*1e3:.2f} ms")
    axV.set_title("Visibility vs time")
    axV.set_ylim(0.0, 1.01)
    axV.grid(True, alpha=0.25)

    # ---- Visibility histogram (normalized)
    finiteV = np.isfinite(V_plot)
    axVh.hist(
        V_plot[finiteV],
        bins=hist_bins,
        orientation="horizontal",
        density=True,     # normalized histogram
        alpha=0.7,
    )
    axVh.set_xlabel("PDF")
    axVh.grid(True, alpha=0.25)
    plt.setp(axVh.get_yticklabels(), visible=False)
    axVh.tick_params(axis="y", length=0)

    # ---- QBER time series
    axQ.plot(t_plot, Q_plot, lw=1)
    axQ.set_xlabel("Time (s)")
    axQ.set_ylabel(f"QBER(t) at τ≈{tau*1e3:.2f} ms")
    axQ.set_title("QBER vs time (from visibility)")
    axQ.set_ylim(0.0, 0.5)
    axQ.grid(True, alpha=0.25)

    # ---- QBER histogram (normalized)
    finiteQ = np.isfinite(Q_plot)
    axQh.hist(
        Q_plot[finiteQ],
        bins=hist_bins,
        orientation="horizontal",
        density=True,     # normalized histogram
        alpha=0.7,
    )
    axQh.set_xlabel("PDF")
    axQh.grid(True, alpha=0.25)
    plt.setp(axQh.get_yticklabels(), visible=False)
    axQh.tick_params(axis="y", length=0)

    fig.suptitle(f"Time series at τ={tau:.3e} s, f_ramp={f_ramp:g} Hz, overlap={overlap:.2f}, n_min={n_min:g}")
    plt.show()

    # ---- Print diagnostics (same as original)
    print("[Selected tau diagnostics]")
    print(diag)
    finite = np.isfinite(Vt)
    if np.any(finite):
        print("V median:", float(np.nanmedian(Vt)))
        print("V mean:  ", float(np.nanmean(Vt)))
        print("V 10–90%:", float(np.nanquantile(Vt, 0.10)), "…", float(np.nanquantile(Vt, 0.90)))

    vals1 = [tcent, Vt, diag]
    vals2 = [t_plot, V_plot]
    vals3 = [t_plot, Q_plot]
    return [vals1, vals2, vals3]

        
def bin_times_counts(ts_s, counts, bin_factor):
    n = (len(counts) // bin_factor) * bin_factor
    ts_s = ts_s[:n]
    counts = counts[:n].reshape(-1, bin_factor).sum(axis=1)
    t_mid = ts_s[:n].reshape(-1, bin_factor)[:, bin_factor//2]
    return t_mid, counts
    
def hybrid_taus(
    tau_min,
    tau_max,
    n_taus_num,
    log_sub=True,
    log_cycles=True):
    """
    Generate τ values such that:
      - τ < 1/f_ramp are arbitrary (sub-cycle)
      - τ >= 1/f_ramp are integer multiples of the ramp period

    Parameters
    ----------
    f_ramp : float
        Ramp frequency in Hz.
    tau_min, tau_max : float
        Overall τ range (seconds).
    n_taus_sub : int
        Number of τ points below one ramp cycle.
    n_taus_cycles : int
        Number of τ points at/above one ramp cycle.
    log_sub, log_cycles : bool
        Whether to space τ logarithmically in each region.

    Returns
    -------
    taus : np.ndarray
        Sorted, unique τ values.
    """
    n_taus_cycles=n_taus_num
    T = 1.0 / 1000

    taus = []

    # ---- Region A: sub-cycle τ (fractional ramp periods)
    if tau_min < T:
        tau_hi = min(T, tau_max)
    
        # Fractions of one ramp period, e.g. 0.1T → 1.0T
        frac_min = max(tau_min / T, 1e-6)
        frac_max = tau_hi / T
    
        fracs = np.arange(
            np.ceil(frac_min * 10) / 10,
            frac_max + 1e-12,
            0.1
        )
    
        taus_sub = fracs * T
        taus.append(taus_sub)

    # ---- Region B: integer-cycle τ
    if tau_max >= T:
        m_min = 1
        m_max = int(np.floor(tau_max * 1000))
        if m_max >= m_min:
            if log_cycles:
                ms = np.unique(
                    np.round(
                        np.logspace(np.log10(m_min), np.log10(m_max), n_taus_cycles)
                    ).astype(int)
                )
            else:
                ms = np.arange(m_min, m_max + 1)
            taus_cycles = ms / 1000
            taus.append(taus_cycles)

    if not taus:
        raise ValueError("No τ values generated — check tau_min/tau_max.")

    taus = np.unique(np.concatenate(taus))
    return taus

    
def AnalyseMeasurement_New(ms,i,n_taus,f_ramp,tau_min,tau_max,tau_op,cycles,overlap=0.5,hist_bins=100,binning=None,binfactor=None):
    if binning==None:
        ts=timestamps_to_elapsed_seconds(ms[i]['data']['timestamps'])
        counts=ms[i]['data']['counts']
    else:
        ts, counts =  bin_times_counts(timestamps_to_elapsed_seconds(ms[i]['data']['timestamps']), ms[i]['data']['counts'], binfactor)

    taus = hybrid_taus(
    tau_min,
    tau_max,
    n_taus,
    log_sub=True,
    log_cycles=True)
    # Run tau sweep
    res = lockin_visibility_summary_vs_tau(ts, counts, f_ramp=f_ramp, taus=taus,overlap=overlap, n_min=0, clip=False)
    # Plot results
    plot_visibility_and_qber_vs_tau(res)
    cycles=cycles
    # Choose operating tau and knee
    if tau_op == None:
        tau_op = choose_operating_tau(res, f_ramp=f_ramp, cycles_min=cycles, within_frac=0.98, use="median")
    #Plot time series at operating tau
    vals=plot_visibility_time_series_at_tau(ts, counts, f_ramp=f_ramp, tau=tau_op, overlap=overlap, n_min=0, hist_bins=hist_bins)
    return([vals,res])

def convert_to_diff_db(etas):
    power_array=etas
    power_array = np.asarray(power_array)

    if np.any(power_array <= 0):
        raise ValueError("All power values must be positive to compute dB.")

    mean_power = np.mean(power_array)
    return 10 * np.log10(power_array / mean_power)

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

def _plot_with_side_hist(x, y, xlabel, ylabel, ylim, bins=50, title=None):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np

    fig = plt.figure(figsize=(10, 4))
    gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)

    ax = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1], sharey=ax)

    ax.plot(x, y, lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)

    finite = np.isfinite(y)
    ax_hist.hist(
        y[finite],
        bins=bins,
        orientation="horizontal",
        density=True,
        alpha=0.7,
    )
    ax_hist.grid(True, alpha=0.25)
    ax_hist.set_xlabel("PDF")

    plt.setp(ax_hist.get_yticklabels(), visible=False)
    ax_hist.tick_params(axis="y", length=0)

    plt.tight_layout()
    plt.show()


def _to_seconds(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts)
    if np.issubdtype(ts.dtype, np.datetime64):
        t0 = ts[0]
        return (ts - t0) / np.timedelta64(1, "s")
    return ts.astype(float) - float(ts[0])


def _phase_from_ramp(t_s: np.ndarray, f_ramp: float, phase_offset: float = 0.0) -> np.ndarray:
    return 2.0 * np.pi * f_ramp * t_s + phase_offset


if NUMBA_OK:
    @njit(cache=True)
    def _prefix_sum(arr):
        out = np.empty(arr.size + 1, dtype=np.float64)
        out[0] = 0.0
        s = 0.0
        for i in range(arr.size):
            s += arr[i]
            out[i + 1] = s
        return out

    @njit(cache=True)
    def _win_sum(ps, a, b):
        return ps[b] - ps[a]

    @njit(cache=True)
    def _poisson_nll(y, lam, lam_floor=1e-12):
        nll = 0.0
        for i in range(y.size):
            li = lam[i]
            if li < lam_floor:
                li = lam_floor
            nll += li - y[i] * np.log(li)
        return nll

    @njit(cache=True)
    def _fit_alpha_beta_for_delta(yw, cw, iters=6, lam_floor=1e-6):
        """
        For fixed delta, we have cw = cos(phi+delta).
        Fit lambda = alpha + beta*cw with Poisson NLL using a few Newton-ish steps
        on (alpha,beta) with constraints alpha>0, beta>=0, and alpha>|beta| to keep lambda positive-ish.

        Returns alpha, beta, nll.
        """
        # Initial guesses
        ymean = 0.0
        for i in range(yw.size):
            ymean += yw[i]
        ymean /= yw.size

        alpha = max(ymean, lam_floor)
        beta  = 0.0

        # Iterative reweighted least squares-like updates for identity mean
        for _ in range(iters):
            # weights w = 1/lambda
            sw0 = 0.0
            sw1 = 0.0
            sw2 = 0.0
            swy0 = 0.0
            swy1 = 0.0

            for i in range(yw.size):
                lam = alpha + beta * cw[i]
                if lam < lam_floor:
                    lam = lam_floor
                w = 1.0 / lam

                x1 = cw[i]
                sw0 += w
                sw1 += w * x1
                sw2 += w * x1 * x1

                yi = yw[i]
                swy0 += w * yi
                swy1 += w * yi * x1

            # Solve 2x2: [[sw0, sw1],[sw1, sw2]] [alpha,beta] = [swy0,swy1]
            det = sw0 * sw2 - sw1 * sw1
            if abs(det) < 1e-18:
                break

            alpha_new = ( sw2 * swy0 - sw1 * swy1) / det
            beta_new  = (-sw1 * swy0 + sw0 * swy1) / det

            # Constraints (keep stable)
            if alpha_new < lam_floor:
                alpha_new = lam_floor
            # nonnegative beta (visibility amplitude)
            if beta_new < 0.0:
                beta_new = 0.0
            # keep alpha >= beta to avoid negative lambda
            if beta_new > alpha_new - lam_floor:
                beta_new = max(0.0, alpha_new - lam_floor)

            alpha, beta = alpha_new, beta_new

        # compute nll
        lam = np.empty(yw.size, dtype=np.float64)
        for i in range(yw.size):
            li = alpha + beta * cw[i]
            if li < lam_floor:
                li = lam_floor
            lam[i] = li

        nll = _poisson_nll(yw, lam, lam_floor=lam_floor)
        return alpha, beta, nll

    @njit(cache=True)
    def _mle_phasegrid_visibility(
        y, cphi, sphi,
        K, S, n_min,
        deltas, iters_ab=6, lam_floor=1e-6
    ):
        """
        Per window: grid search over delta.
        For each delta:
          cw = cos(phi+delta) = cphi*cosd - sphi*sind
          Fit alpha,beta (Poisson) and compute NLL
        Choose best delta => V=beta/alpha
        """
        n = y.size
        nwin = (n - K) // S + 1
        V_out = np.empty(nwin, dtype=np.float64)
        delta_out = np.empty(nwin, dtype=np.float64)

        for w in range(nwin):
            start = w * S
            end = start + K

            # total counts gate
            sy = 0.0
            for i in range(start, end):
                sy += y[i]
            if sy < n_min:
                V_out[w] = np.nan
                delta_out[w] = np.nan
                continue

            best_nll = 1e300
            best_V = np.nan
            best_d = 0.0

            for di in range(deltas.size):
                d = deltas[di]
                cd = np.cos(d)
                sd = np.sin(d)

                # build cw = cos(phi+d) quickly from precomputed cos/sin(phi)
                cw = np.empty(K, dtype=np.float64)
                yw = np.empty(K, dtype=np.float64)
                idx = 0
                for i in range(start, end):
                    cw[idx] = cphi[i] * cd - sphi[i] * sd
                    yw[idx] = y[i]
                    idx += 1

                alpha, beta, nll = _fit_alpha_beta_for_delta(
                    yw, cw, iters=iters_ab, lam_floor=lam_floor
                )

                if nll < best_nll:
                    best_nll = nll
                    best_d = d
                    best_V = beta / alpha if alpha > 0 else np.nan

            V_out[w] = best_V
            delta_out[w] = best_d

        return V_out, delta_out


def lockin_visibility_vs_time_mle_numba(
    ts,
    counts,
    f_ramp: float,
    tau: float,
    step: Optional[float] = None,
    phase_offset: float = 0.0,
    n_min: float = 0.0,
    clip: bool = True,
    n_phase: int = 64,        # phase grid points
    iters_ab: int = 8,        # alpha/beta solver iterations
    lam_floor: float = 1e-6,
    verbose: bool = True,
    make_plots: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Numba-friendly Poisson MLE-like visibility vs time:
      - grid search over delta
      - Poisson fit for alpha,beta per delta
      - choose delta minimizing NLL
      - V = beta/alpha

    Returns t_centers, V(t), QBER(t), diagnostics
    """
    if not NUMBA_OK:
        raise ImportError("numba is not available. Install with: pip install numba")
    t_s = _to_seconds(np.asarray(ts)).astype(np.float64)
    y = np.clip(np.nan_to_num(np.asarray(counts), nan=0.0), 0, None).astype(np.float64)
    n = min(t_s.size, y.size)
    t_s = t_s[:n]
    y = y[:n]
    if n < 3:
        raise ValueError("Need at least 3 samples.")

    dts = np.diff(t_s)
    dt_pos = dts[dts > 0]
    if dt_pos.size == 0:
        raise ValueError("Timestamps not strictly increasing.")
    dt = float(np.mean(dt_pos))

    if step is None:
        step = tau / 2.0
    K = max(3, int(round(tau / dt)))
    S = max(1, int(round(step / dt)))
    if K > n:
        return np.array([]), np.array([]), np.array([]), {"dt": dt, "K": K, "S": S, "n_windows": 0}

    phi = _phase_from_ramp(t_s, f_ramp=f_ramp, phase_offset=phase_offset).astype(np.float64)
    cphi = np.cos(phi).astype(np.float64)
    sphi = np.sin(phi).astype(np.float64)

    deltas = np.linspace(-np.pi, np.pi, int(n_phase), endpoint=False).astype(np.float64)

    V_out, delta_out = _mle_phasegrid_visibility(
        y, cphi, sphi,
        K, S, float(n_min),
        deltas, iters_ab=int(iters_ab), lam_floor=float(lam_floor)
    )

    nwin = V_out.size
    centers_idx = (np.arange(nwin, dtype=np.int64) * S) + (K // 2)
    t_centers = t_s[centers_idx]

    if clip:
        Vq = np.clip(V_out, 0.0, 1.0)
    else:
        Vq = V_out
    QBER = 0.5 * (1.0 - Vq)

    diagnostics = {
        "dt": dt,
        "K_samples": int(K),
        "S_samples": int(S),
        "tau_eff_s": float(K * dt),
        "step_eff_s": float(S * dt),
        "n_windows_total": int(nwin),
        "n_windows_used": int(np.sum(np.isfinite(V_out))),
        "n_phase": int(n_phase),
        "iters_ab": int(iters_ab),
        "lam_floor": float(lam_floor),
        "n_min_counts": float(n_min),
        "clip_for_qber": bool(clip),
    }
    if verbose:
        print("[Numba phase-grid Poisson MLE visibility]")
        for k, v in diagnostics.items():
            print(f"  {k}: {v}")

    if make_plots and t_centers.size:
        _plot_with_side_hist(
            t_centers,
            V_out,
            xlabel="Time (s)",
            ylabel=f"V_MLE_NUMBA(t)  (tau ≈ {K*dt*1e3:.2f} ms)",
            ylim=(-0.05, 1.05 if clip else 1.3),
            bins=60,
            title="Visibility vs Time (Poisson MLE, Numba)",
        )
    
        _plot_with_side_hist(
            t_centers,
            QBER,
            xlabel="Time (s)",
            ylabel="QBER(t)",
            ylim=(0.0, 0.5),
            bins=60,
            title="QBER vs Time (from Visibility, Poisson MLE, Numba)",
        )


    return t_centers, V_out, QBER, diagnostics

def lockin_visibility_summary_vs_tau_mle_numba(
    ts: np.ndarray,
    counts: np.ndarray,
    f_ramp: float,
    taus: Optional[np.ndarray] = None,
    n_taus: int = 25,
    tau_min: Optional[float] = None,
    tau_max: Optional[float] = None,
    log_bins: bool = True,
    cycles_min: int = 10,
    overlap: float = 0.5,
    phase_offset: float = 0.0,
    n_min: float = 50.0,
    clip: bool = False,
    # forwarded to lockin_visibility_vs_time_mle_numba
    n_phase: int = 64,
    iters_ab: int = 8,
    lam_floor: float = 1e-6,
    verbose: bool = False,
) -> "VisibilityTauResult":
    """
    Sweep over taus, compute V(t; tau) in sliding windows using
    lockin_visibility_vs_time_mle_numba, and summarize per tau:
    max/min/mean/median + 10–90% band and number of windows.

    IMPORTANT:
      - Any requested tau is snapped to the sampling grid:
            K = round(tau/dt)  (min 3)
            tau_eff = K*dt
      - Step is also snapped:
            step_req = tau_req*(1-overlap)
            S = round(step_req/dt) (min 1)
            step_eff = S*dt

    Output res.taus are the *effective* taus (tau_eff) actually used.
    The originally requested taus are stored in meta["taus_requested"].
    """

    ts_s = _to_seconds(np.asarray(ts))  # <-- critical: seconds, relative
    ts_s = np.asarray(ts_s, float)
    
    y  = np.clip(np.nan_to_num(np.asarray(counts, float), nan=0.0), 0, None)
    n = min(ts_s.size, y.size)
    ts_s = ts_s[:n]
    y    = y[:n]
    if n < 2:
        raise ValueError("Need at least 2 samples.")
    
    dts = np.diff(ts_s)
    dt_pos = dts[dts > 0]
    if dt_pos.size == 0:
        raise ValueError("Timestamps are not strictly advancing.")
    dt = float(np.mean(dt_pos))
    
    duration = float(ts_s[-1] - ts_s[0])
    rate = float(np.sum(y) / duration)


    # defaults only used when GENERATING taus
    if tau_min is None:
        tau_min_default = max(dt, 1.0 / f_ramp)
    else:
        tau_min_default = float(max(tau_min, dt))

    if tau_max is None:
        tau_max_default = min(0.5 * duration, 0.2)
    else:
        tau_max_default = float(max(tau_max, tau_min_default))

    # choose / clean taus (requested)
    if taus is None:
        tau_min_use = float(max(tau_min_default, dt))
        tau_max_use = float(max(tau_max_default, tau_min_use))

        if n_taus <= 1:
            taus_req = np.array([tau_min_use], dtype=float)
        else:
            if log_bins:
                taus_req = np.logspace(np.log10(tau_min_use), np.log10(tau_max_use), n_taus)
            else:
                taus_req = np.linspace(tau_min_use, tau_max_use, n_taus)
        taus_req = np.unique(taus_req.astype(float))
    else:
        taus_req = np.unique(np.asarray(taus, dtype=float))
        taus_req = taus_req[np.isfinite(taus_req)]
        taus_req = taus_req[taus_req > 0.0]

        # Only apply tau_min/tau_max filtering if user explicitly provided them
        if tau_min is not None:
            taus_req = taus_req[taus_req >= float(max(tau_min, dt))]
        if tau_max is not None:
            taus_req = taus_req[taus_req <= float(tau_max)]

        if taus_req.size == 0:
            raise ValueError("Provided taus are empty after cleaning/filtering.")

    # overlap
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0,1).")

    # ---- snap taus to dt grid (and compute snapped steps)
    # K>=3 ensures the MLE routine behaves sensibly
    Ks = np.maximum(3, np.rint(taus_req / dt).astype(int))
    taus_eff = Ks * dt

    # step snapping using the *requested* tau (matches your prior semantics)
    steps_req = taus_req * (1.0 - overlap)
    Ss = np.maximum(1, np.rint(steps_req / dt).astype(int))
    steps_eff = Ss * dt

    # ---- collapse duplicates created by snapping (critical!)
    # If many taus_req map to the same K, you’d otherwise “repeat” the same tau.
    # We compute each unique K once.
    Ku, idx_first = np.unique(Ks, return_index=True)
    # keep in increasing effective tau order
    order = np.argsort(Ku)
    Ku = Ku[order]
    idx_first = idx_first[order]

    taus_eff_u = Ku * dt
    taus_req_u = taus_req[idx_first]
    Ss_u = Ss[idx_first]
    steps_eff_u = Ss_u * dt

    # outputs for unique taus
    M = taus_eff_u.size
    V_max    = np.full(M, np.nan)
    V_min    = np.full(M, np.nan)
    V_mean   = np.full(M, np.nan)
    V_median = np.full(M, np.nan)
    V_p10    = np.full(M, np.nan)
    V_p90    = np.full(M, np.nan)
    n_windows = np.zeros(M, dtype=int)

    for i in range(M):
        tau_use = float(taus_eff_u[i])
        step_use = float(steps_eff_u[i])

        tcent, Vt, QBER, diag = lockin_visibility_vs_time_mle_numba(
            ts,
            y,
            f_ramp=f_ramp,
            tau=tau_use,
            step=step_use,
            phase_offset=phase_offset,
            n_min=float(n_min),
            clip=bool(clip),
            n_phase=int(n_phase),
            iters_ab=int(iters_ab),
            lam_floor=float(lam_floor),
            verbose=False,
            make_plots=False,
        )

        Vt = Vt[np.isfinite(Vt)]
        n_windows[i] = int(Vt.size)
        if Vt.size == 0:
            continue

        V_max[i]    = float(np.max(Vt))
        V_min[i]    = float(np.min(Vt))
        V_mean[i]   = float(np.mean(Vt))
        V_median[i] = float(np.median(Vt))
        V_p10[i]    = float(np.quantile(Vt, 0.10))
        V_p90[i]    = float(np.quantile(Vt, 0.90))

        if verbose:
            print(f"[τ_req={taus_req_u[i]:.3e}s -> τ_eff={taus_eff_u[i]:.3e}s | "
                  f"step_eff={steps_eff_u[i]:.3e}s] windows={n_windows[i]:d} "
                  f"Vmean={V_mean[i]:.4f}")

    expected_counts = rate * taus_eff_u

    meta = {
        "dt": dt,
        "duration_s": duration,
        "rate_counts_per_s": rate,
        "cycles_min": cycles_min,
        "trust_tau_min_s": cycles_min / f_ramp,
        "overlap": overlap,
        "n_min_counts_per_window": float(n_min),
        "clip": bool(clip),
        "estimator": "lockin_visibility_vs_time_mle_numba",
        "n_phase": int(n_phase),
        "iters_ab": int(iters_ab),
        "lam_floor": float(lam_floor),

        # transparency about snapping / duplicates
        "taus_requested": taus_req,
        "taus_effective_all": taus_eff,
        "Ks_all": Ks,
        "taus_requested_unique": taus_req_u,
        "taus_effective_unique": taus_eff_u,
        "Ku_unique": Ku,
        "steps_requested_all": steps_req,
        "steps_effective_unique": steps_eff_u,
        "Ss_unique": Ss_u,
        "note": "Output taus are effective taus (K*dt) used by the estimator; duplicates after snapping are collapsed.",
    }

    return VisibilityTauResult(
        taus=taus_eff_u,
        n_windows=n_windows,
        V_max=V_max,
        V_min=V_min,
        V_mean=V_mean,
        V_median=V_median,
        V_p10=V_p10,
        V_p90=V_p90,
        expected_counts_per_window=expected_counts,
        meta=meta,
    )

def plot_visibility_time_series_at_tau_MLEnumba(
    ts: np.ndarray,
    counts: np.ndarray,
    f_ramp: float,
    tau: float,
    overlap: float = 0.5,
    phase_offset: float = 0.0,
    n_min: float = 50.0,
    clip_for_qber: bool = True,
    max_points: int = 5000,
    hist_bins: int = 60,
) -> List[Any]:
    """
    Single figure:
      - Left:  V(t) with normalized sideways histogram
      - Right: QBER(t) with normalized sideways histogram

    Returns the same list structure as your original:
      [ [tcent, Vt, diag], [tcent_ds, Vt_ds], [tcent_ds, Q_ds] ]
    """
    # step from overlap
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0,1).")
    step = tau * (1.0 - overlap)

    # Compute visibility time series (no clipping here; we clip only for QBER mapping)
    tcent, Vt, QBERir, diag = lockin_visibility_vs_time_mle_numba(
        ts, counts, f_ramp=f_ramp, tau=float(tau), step=float(step),
        phase_offset=phase_offset, n_min=float(n_min), clip=False,make_plots = False, verbose=False
    )

    # Downsample for display
    if tcent.size > max_points:
        ds = int(np.ceil(tcent.size / max_points))
    else:
        ds = 1

    t_plot = tcent[::ds]
    V_plot = Vt[::ds]

    # QBER from visibility
    Vc = np.clip(Vt, 0.0, 1.0) if clip_for_qber else Vt
    Q = 0.5 * (1.0 - Vc)
    Q_plot = Q[::ds]

    # ---- Build one figure: [V time + V hist] [Q time + Q hist]
    fig = plt.figure(figsize=(14, 3))
    gs = GridSpec(
        1, 5,
        width_ratios=[4, 1, 0.6, 4, 1],  # <-- spacer column
        wspace=0.04
    )


    axV  = fig.add_subplot(gs[0, 0])
    axVh = fig.add_subplot(gs[0, 1], sharey=axV)
    
    # gs[0, 2] is intentionally empty (spacer)
    
    axQ  = fig.add_subplot(gs[0, 3])
    axQh = fig.add_subplot(gs[0, 4], sharey=axQ)


    # ---- Visibility time series
    axV.plot(t_plot, V_plot, lw=1)
    axV.set_xlabel("Time (s)")
    axV.set_ylabel(f"V(t) at τ≈{tau*1e3:.2f} ms")
    axV.set_title("Visibility vs time")
    axV.set_ylim(0.0, 1.01)
    axV.grid(True, alpha=0.25)

    # ---- Visibility histogram (normalized)
    finiteV = np.isfinite(V_plot)
    axVh.hist(
        V_plot[finiteV],
        bins=hist_bins,
        orientation="horizontal",
        density=True,     # normalized histogram
        alpha=0.7,
    )
    axVh.set_xlabel("PDF")
    axVh.grid(True, alpha=0.25)
    plt.setp(axVh.get_yticklabels(), visible=False)
    axVh.tick_params(axis="y", length=0)

    # ---- QBER time series
    axQ.plot(t_plot, Q_plot, lw=1)
    axQ.set_xlabel("Time (s)")
    axQ.set_ylabel(f"QBER(t) at τ≈{tau*1e3:.2f} ms")
    axQ.set_title("QBER vs time (from visibility)")
    axQ.set_ylim(0.0, 0.5)
    axQ.grid(True, alpha=0.25)

    # ---- QBER histogram (normalized)
    finiteQ = np.isfinite(Q_plot)
    axQh.hist(
        Q_plot[finiteQ],
        bins=hist_bins,
        orientation="horizontal",
        density=True,     # normalized histogram
        alpha=0.7,
    )
    axQh.set_xlabel("PDF")
    axQh.grid(True, alpha=0.25)
    plt.setp(axQh.get_yticklabels(), visible=False)
    axQh.tick_params(axis="y", length=0)

    fig.suptitle(f"Time series at τ={tau:.3e} s, f_ramp={f_ramp:g} Hz, overlap={overlap:.2f}, n_min={n_min:g}")
    plt.show()

    # ---- Print diagnostics (same as original)
    print("[Selected tau diagnostics]")
    print(diag)
    finite = np.isfinite(Vt)
    if np.any(finite):
        print("V median:", float(np.nanmedian(Vt)))
        print("V mean:  ", float(np.nanmean(Vt)))
        print("V 10–90%:", float(np.nanquantile(Vt, 0.10)), "…", float(np.nanquantile(Vt, 0.90)))

    vals1 = [tcent, Vt, diag]
    vals2 = [t_plot, V_plot]
    vals3 = [t_plot, Q_plot]
    return [vals1, vals2, vals3]

def AnalyseMeasurement_NewNumba(ms,i,n_taus,f_ramp,tau_min,tau_max,tau_op,cycles,overlap=0.5,hist_bins=100,binning=None,binfactor=None):
    if binning==None:
        ts=timestamps_to_elapsed_seconds(ms[i]['data']['timestamps'])
        counts=ms[i]['data']['counts']
    else:
        ts, counts =  bin_times_counts(timestamps_to_elapsed_seconds(ms[i]['data']['timestamps']), ms[i]['data']['counts'], binfactor)

    taus = hybrid_taus(
    tau_min,
    tau_max,
    n_taus,
    log_sub=True,
    log_cycles=True)
    # Run tau sweep
    res = lockin_visibility_summary_vs_tau_mle_numba(ts, counts, f_ramp=f_ramp, taus=taus,tau_min=tau_min,tau_max=tau_max,overlap=overlap, n_min=0, clip=False,n_phase= 128)
    # Plot results
    plot_visibility_and_qber_vs_tau(res)
    vals=plot_visibility_time_series_at_tau_MLEnumba(ts, counts, f_ramp=f_ramp, tau=tau_op, overlap=overlap, n_min=0, hist_bins=hist_bins)
    return([vals,res])
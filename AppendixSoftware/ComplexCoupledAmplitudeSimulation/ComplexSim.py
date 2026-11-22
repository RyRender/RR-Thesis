from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
import math
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, ScalarFormatter
import aotools
import random
import free_space_fibre_overlap_sim as sim
from time import time
import os, math
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

def _mp_init():
    """Run once in each worker: avoid BLAS/OpenMP oversubscription."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
# ---------- labels -----------
ZERN_INFO = [
    dict(n=0, m=0,  expr=r"1",                          name="Piston"),
    dict(n=1, m=-1, expr=r"r\sin(\theta)",              name="Tilt Y"),
    dict(n=1, m=1,  expr=r"r\cos(\theta)",              name="Tilt X"),
    dict(n=2, m=-2, expr=r"r^{2}\sin(2\theta)",         name="$45\degree$ Astigmatism"),
    dict(n=2, m=0,  expr=r"2r^{2}-1",                   name="Defocus"),
    dict(n=2, m=2,  expr=r"r^{2}\cos(2\theta)",         name="$0\degree$ Astigmatism"),
    dict(n=3, m=-3, expr=r"r^{3}\sin(3\theta)",         name="Trefoil Y"),
    dict(n=3, m=-1, expr=r"(3r^{3}-2r)\sin(\theta)",    name="Y Coma"),
    dict(n=3, m=1,  expr=r"(3r^{3}-2r)\cos(\theta)",    name="X Coma"),
    dict(n=3, m=3,  expr=r"r^{3}\cos(3\theta)",         name="Trefoil X"),
    dict(n=4, m=-4, expr=r"r^{4}\sin(4\theta)",         name="Tetrafoil Y"),
    dict(n=4, m=-2, expr=r"(4r^{4}-3r^{2})\sin(2\theta)", name="Secondary Astigmatism Y"),
    dict(n=4, m=0,  expr=r"6r^{4}-6r^{2}+1",            name="3rd Order Spherical"),
    dict(n=4, m=2,  expr=r"(4r^{4}-3r^{2})\cos(2\theta)", name="Secondary Astigmatism X"),
    dict(n=4, m=4,  expr=r"r^{4}\cos(4\theta)",         name="Tetrafoil X"),
]

def zern_label(j: int, style: str = "short") -> str:
    z = ZERN_INFO[j-1]
    if style == "short":
        return f"{z['name']}"
    return rf"$Z_{{{z['n']}}}^{{{z['m']}}}(r,\theta)={z['expr']}$ — {z['name']}"

# --- helpers ---
def _to_knn(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2: return arr[None, ...].astype(np.float64)
    if arr.ndim != 3: raise ValueError(f"Unsupported rank {arr.ndim}")
    s0,s1,s2 = arr.shape
    if s1==s2 and s0!=s1: out = arr
    elif s0==s1 and s0!=s2: out = np.moveaxis(arr,-1,0)
    elif s0==s2 and s0!=s1: out = np.moveaxis(arr,1,0)
    else:
        k_axis = int(np.argmin([s0,s1,s2]))
        out = arr if k_axis==0 else np.moveaxis(arr,k_axis,0)
        K,N1,N2 = out.shape
        if N1!=N2: raise ValueError(f"Cannot coerce {arr.shape} to (K,N,N)")
    return out.astype(np.float64)

def load_zernike_stack(path: Union[str, Path]) -> np.ndarray:
    return _to_knn(np.load(path))

def unit_disk_grid(n: int):
    xs = (np.arange(n) - (n - 1)/2.0) * (2.0/n)
    ys = (np.arange(n) - (n - 1)/2.0) * (2.0/n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    R = np.sqrt(X**2 + Y**2)
    mask = R <= 1.0 + 1e-12
    dA = (2.0/n)**2
    return R, mask, dA

def gaussian_on_pupil(R, mask, u0: float):
    M = np.zeros_like(R, dtype=np.float64)
    M[mask] = np.exp(-(R[mask]**2) / (u0**2))
    return M

# ---------- core overlap (now returns complex A as well) ----------
def phase_and_intensity_vs_coeff(mode2d, R, mask, dA, u0, c_vals):
    """
    Exact normalized overlap on a sampled pupil.
    Returns: phase th, intensity I, and complex A for further processing.
    """
    M = gaussian_on_pupil(R, mask, u0)
    m = mode2d[mask]; w = M[mask]
    norm_E = math.sqrt(m.size * dA)
    norm_M = math.sqrt(np.sum(w**2) * dA)
    th = np.empty_like(c_vals); I = np.empty_like(c_vals)
    A_out = np.empty_like(c_vals, dtype=np.complex128)

    # start at c0
    c0 = float(c_vals[0])
    E = np.exp(1j * (c0 * m)) if c0 != 0.0 else np.ones_like(m, dtype=np.complex128)
    A = (np.sum(E*w)*dA)/(norm_E*norm_M) if norm_M else 0j
    A_out[0] = A
    th[0] = np.angle(A); I[0] = float(np.abs(A)**2)

    # incremental update for uniform c-grid
    if len(c_vals) > 1:
        dc = float(c_vals[1]-c_vals[0]); q = np.exp(1j*(dc*m))
        for k in range(1, len(c_vals)):
            E *= q
            A = (np.sum(E*w)*dA)/(norm_E*norm_M) if norm_M else 0j
            A_out[k] = A
            th[k] = np.angle(A); I[k] = float(np.abs(A)**2)
    return th, I, A_out

# ---------- π tick helpers ----------
_PI_MAJOR = np.array([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])

def _pi_formatter():
    lab = {
        -np.pi:  r"$-\pi$",
        -np.pi/2: r"$-\frac{\pi}{2}$",
        0.0:     r"$0$",
        np.pi/2: r"$\frac{\pi}{2}$",
        np.pi:   r"$\pi$",
    }
    return FuncFormatter(lambda val, pos: lab.get(
        _PI_MAJOR[np.argmin(np.abs(_PI_MAJOR - val))] if np.min(np.abs(_PI_MAJOR - val)) < 1e-6 else val,
        rf"${val/np.pi:.2g}\pi$"
    ))

def apply_pi_ticks(ax):
    ax.set_ylim(-np.pi, np.pi)
    ax.yaxis.set_major_locator(FixedLocator(_PI_MAJOR))
    ax.yaxis.set_major_formatter(_pi_formatter())

def _fmt_pi_major(x, pos):
    k = int(np.round(x/np.pi))
    if abs(x/np.pi - k) > 1e-6: return ""
    if k == 0: return r"$0$"
    s = "-" if k < 0 else ""
    kabs = abs(k)
    return rf"${s}\pi$" if kabs == 1 else rf"${s}{kabs}\pi$"

def apply_pi_xticks(ax, max_mult=4, show_minors=True):
    ax.set_xlim(-max_mult*np.pi, max_mult*np.pi)
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_pi_major))
    if show_minors:
        ax.xaxis.set_minor_locator(MultipleLocator(base=np.pi/2))
def weighted_mean_m(mode2d, u0):
    """⟨m⟩_w with Gaussian weight; not used for the rule, but handy if you prefer a numeric criterion."""
    N = mode2d.shape[0]
    R, mask, dA = unit_disk_grid(N)
    M = gaussian_on_pupil(R, mask, u0)
    m = mode2d[mask]; w = M[mask]
    return float(np.sum(m*w)*dA / (np.sum(w)*dA))

# ---- extra helper for λ (waves) ticks on the x-axis ----
def apply_wave_xticks(ax, coeff_max_rad, show_minors=True):
    """
    Label x in 'waves' (multiples of λ) while data remain in radians.
    Major ticks at integer waves within [-coeff_max, +coeff_max].
    """
    ax.set_xlim(-coeff_max_rad, coeff_max_rad)

    def fmt_waves(x_rad, pos):
        w = x_rad / (2*np.pi)
        k = int(np.round(w))
        if abs(w - k) > 1e-6:
            return ""                        # only label integers
        if k == 0: return r"$0$"
        s = "-" if k < 0 else ""
        kabs = abs(k)
        return rf"${s}\lambda$" if kabs == 1 else rf"${s}{kabs}\,\lambda$"

    ax.xaxis.set_major_locator(MultipleLocator(base=2*np.pi))  # 1 wave
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_waves))
    if show_minors:
        ax.xaxis.set_minor_locator(MultipleLocator(base=np.pi))  # 1/2 wave


# ---- robust 0/π phase for (nearly) real overlaps ----
def phase_binary_from_real(A, I, Imin, eta=1e-3):
    """
    Produce a 0/π phase where overlap is effectively real (|Im(A)| ≤ eta*|A|),
    fall back to raw angle elsewhere; NaN where I ≤ Imin.
    """
    ph = np.full_like(I, np.nan, dtype=float)
    valid = I > Imin
    Aval = A[valid]
    mag  = np.abs(Aval)
    realish = np.abs(Aval.imag) <= eta * np.maximum(mag, 1e-15)
    out = np.empty_like(Aval.real)
    out[realish]  = np.where(Aval.real[realish] >= 0.0, 0.0, np.pi-0.1)
    out[~realish] = np.angle(Aval[~realish])
    ph[valid] = out
    return ph
    
def phase_to_opd(phi_rad: np.ndarray, wavelength_m: float) -> np.ndarray:
    """Wavefront OPD (meters) from phase (radians)."""
    return (wavelength_m / (2.0 * np.pi)) * phi_rad

def zernikes_to_opd(Z_stack: np.ndarray, coeff_rad: np.ndarray, wavelength_m: float) -> np.ndarray:
    """
    Z_stack: shape (K, N, N) Zernike modes (dimensionless).
    coeff_rad: shape (K,) coefficients in radians of phase.
    Returns OPD map (meters).
    """
    phi = np.tensordot(coeff_rad, Z_stack, axes=(0, 0))  # (N, N) phase in rad
    return phase_to_opd(phi, wavelength_m)
def phase_to_mirror_height(phi_rad: np.ndarray, wavelength_m: float) -> np.ndarray:
    """Surface height for a reflective mirror producing phase phi."""
    return (wavelength_m / (4.0 * np.pi)) * phi_rad
def phase_to_thickness(phi_rad: np.ndarray, wavelength_m: float, n: float, n0: float = 1.0) -> np.ndarray:
    """
    Material thickness t (meters) that produces phase phi for transmission through (n) in ambient n0.
    """
    dn = (n - n0)
    if abs(dn) < 1e-12:
        raise ValueError("n - n0 too small to compute transmissive thickness.")
    return (wavelength_m / (2.0 * np.pi * dn)) * phi_rad
    
# --- helper: draw vertical guides at multiples of λ (i.e., a = 2π n) ---
def draw_lambda_guides(ax, coeff_max,
                       include_zero=True, color="0.6", lw=1.0, ls=":", alpha=0.6):
    """
    Draw dotted vertical lines at a = 2π n (n ∈ ℤ) within [-coeff_max, +coeff_max].
    coeff_max is in radians of phase (your x-data domain).
    """
    step = 2*np.pi  # one wave of phase
    n_min = int(np.ceil(-coeff_max / step))
    n_max = int(np.floor( coeff_max / step))
    for n in range(n_min, n_max + 1):
        if n == 0 and not include_zero:
            continue
        x = n * step
        ax.axvline(x, color=color, lw=lw, ls=ls, alpha=alpha, zorder=0)  # zorder=0 keeps under the data
  
# --- top-level worker (picklable) ---
def _compute_one_mode(args):
    """
    Compute (phase_display, intensity) for one Zernike mode, with phase selection applied.
    args = (k, mode2d, R, mask, dA, u0, c_vals, Imin, phase_mode)
    """
    (k, mode2d, R, mask, dA, u0, c_vals, Imin, phase_mode) = args
    res = phase_and_intensity_vs_coeff(mode2d, R, mask, dA, u0, c_vals)
    if len(res) == 3:
        th, I, A = res
    else:
        th, I = res
        A = np.sqrt(np.clip(I, 0.0, None)) * np.exp(1j * th)

    if phase_mode == "auto":
        m = ZERN_INFO[k]["m"]
        ph = np.where(I > Imin, th, np.nan) if m == 0 else phase_binary_from_real(A, I, Imin, eta=1e-3)
    elif phase_mode == "unwrap":
        valid = I > Imin
        ph = np.full_like(th, np.nan); ph[valid] = np.unwrap(th[valid], discont=np.pi/2)
    elif phase_mode == "real_binary":
        ph = phase_binary_from_real(A, I, Imin, eta=1e-3)
    else:
        ph = np.where(I > Imin, th, np.nan)
    return k, ph, I

# ===================== PLOTTING FUNCTION (parallelizable) =====================
def plot_phase_and_intensity_neat(
    zernike_path="fixedzernike_stack.npy",
    Ksel=15,
    u0=0.30,
    coeff_max=10*np.pi,
    num=401,
    downsample=1,
    Imin=1e-14,
    titles=None,
    figsize=(12, 9),
    header_heights=(0.12,),
    phase_mode="auto",
    x_units="pi",
    row_gap=0.55,
    title_pad=2,
    tick_sizes=9,
    parallel=True,
    workers=None,
    backend="thread",     # 'thread' | 'process' | 'none'
    show_progress=True,   # <<< show a progress bar / progress prints
    progress_desc="Computing modes"
):
    # ---- data prep ----
    Z = load_zernike_stack(zernike_path)
    K, N, _ = Z.shape
    Ksel = min(Ksel, K)
    Z = Z[:Ksel, ::downsample, ::downsample]
    _, Nds, _ = Z.shape
    R, mask, dA = unit_disk_grid(Nds)

    c_vals = np.linspace(-coeff_max, coeff_max, num, dtype=np.float64)
    tasks = [(k, Z[k], R, mask, dA, u0, c_vals, Imin, phase_mode) for k in range(Ksel)]

    # ---- compute curves (parallel or sequential) ----
    # ---- compute curves (parallel or sequential) ----
    def _compute_sequential_with_progress():
        out = [None]*Ksel
        use_tqdm = show_progress and (_tqdm is not None)
        iterator = range(Ksel)
        if use_tqdm:
            iterator = _tqdm(iterator, total=Ksel, desc=progress_desc)
        for k in iterator:
            _, ph, I = _compute_one_mode((k, Z[k], R, mask, dA, u0, c_vals, Imin, phase_mode))
            out[k] = (ph, I)
        return out
    
    results = [None]*Ksel
    if (not parallel) or (backend == "none") or (Ksel <= 1):
        results = _compute_sequential_with_progress()
    else:
        # pick a sensible worker count
        phys = None
        try:
            import psutil
            phys = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        except Exception:
            phys = os.cpu_count() or 1
        n_workers = workers or max(1, min(Ksel, phys))
    
        # tqdm in parent is fine for both threads & processes
        use_tqdm = show_progress and (_tqdm is not None)
        try:
            if backend == "thread":
                from concurrent.futures import ThreadPoolExecutor
                Executor = ThreadPoolExecutor
                ex_kwargs = dict(max_workers=n_workers)
            else:
                from concurrent.futures import ProcessPoolExecutor
                Executor = ProcessPoolExecutor
                ex_kwargs = dict(max_workers=n_workers, initializer=_mp_init)
    
            tasks = [(k, Z[k], R, mask, dA, u0, c_vals, Imin, phase_mode) for k in range(Ksel)]
            chunksize = max(1, math.ceil(len(tasks) / (n_workers * 4)))
    
            with Executor(**ex_kwargs) as ex:
                it = ex.map(_compute_one_mode, tasks, chunksize=chunksize)
                if use_tqdm:
                    it = _tqdm(it, total=Ksel, desc=progress_desc)
                for (k, ph, I) in it:
                    results[k] = (ph, I)
    
            # paranoid fallback
            if any(r is None for r in results):
                if show_progress and not use_tqdm:
                    print("Parallel incomplete; falling back to sequential…")
                results = _compute_sequential_with_progress()
    
        except Exception as e:
            if show_progress:
                print(f"[parallel disabled: {e}] → running sequentially…")
            results = _compute_sequential_with_progress()

    # ---- figure + layout (unchanged besides using 'results') ----
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({'font.size': 10})
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=10)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "xtick.labelsize": tick_sizes,
        "ytick.labelsize": tick_sizes,
    })

    nrows, ncols = 5, 3
    mid_row = nrows // 2
    mid_col = ncols // 2

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows + 1, ncols, figure=fig,
                  height_ratios=[header_heights[0]] + [1]*nrows,
                  hspace=row_gap, wspace=0.20)
    title_ax = fig.add_subplot(gs[0, :]); title_ax.axis("off")
    axes = [fig.add_subplot(gs[r+1, c]) for r in range(nrows) for c in range(ncols)]

    if x_units == "waves":
        for ax in axes: apply_wave_xticks(ax, coeff_max, show_minors=True)
    else:
        max_mult = int(round(coeff_max/np.pi))
        for ax in axes: apply_pi_xticks(ax, max_mult=max_mult, show_minors=True)

    phase_handle = intensity_handle = None

    for k in range(Ksel):
        ax = axes[k]
        ph, I = results[k]

        h1, = ax.plot(c_vals, ph, color="tab:orange", label="arg(A)", lw=1.0)
        apply_pi_ticks(ax)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="both", labelsize=tick_sizes)

        ax2 = ax.twinx()
        h2, = ax2.plot(c_vals, I, color="tab:blue", label="|A|²")
        ax2.set_ylim(0, 1)
        fmt_math = ScalarFormatter(useMathText=True); fmt_math.set_useOffset(False)
        ax2.yaxis.set_major_formatter(fmt_math)
        ax2.tick_params(axis="both", which="both", labelsize=tick_sizes)

        row, col = divmod(k, ncols)
        if col != ncols - 1:
            ax2.spines["right"].set_visible(False)
            ax2.tick_params(right=False, labelright=False)

        if titles and k < len(titles):
            ax.set_title(titles[k], fontsize=9, pad=title_pad)
        else:
            ax.set_title(rf"$Z_{{{k+1}}}$: {zern_label(k+1, 'short')}", size=11, pad=title_pad)

        if x_units == "waves":
            draw_lambda_guides(ax, coeff_max, include_zero=True, color="black", lw=1.1, ls=":", alpha=1.0)

        if row < nrows - 1: ax.tick_params(labelbottom=False)
        if col > 0:         ax.tick_params(labelleft=False)

        if col == 0 and row == mid_row: ax.set_ylabel("$Φ$ [rad]", size=16)
        if col == ncols - 1 and row == mid_row: ax2.set_ylabel("$\\eta$", size=16)
        if row == nrows - 1 and col == mid_col:
            ax.set_xlabel("Zernike Coefficient $a_{i}$ [waves]" if x_units == "waves"
                          else "Zernike Coefficient $a_{i}$ [rad]", size=14)

        if phase_handle is None:
            phase_handle, intensity_handle = h1, h2

    for k in range(Ksel, nrows*ncols):
        axes[k].axis("off")

    title_ax.text(0.5, 0.995, "Complex Couple Amplitude Simulation plots",
                  ha="center", va="top", transform=title_ax.transAxes, size=14)

    fig.legend([intensity_handle, phase_handle], ["|A|²","arg(A)"],
               loc="upper center", bbox_to_anchor=(0.5, 0.975),
               ncol=2, frameon=False, handlelength=2.8, columnspacing=1.2)

    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.985)
    fig.savefig('ZernikePhaseWrapping.pdf', dpi=300)
    plt.show()
# --- optional progress (tqdm) ---
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

def _pick_chunk_size(npix: int, ncoef: int, target_mb: int = 256) -> int:
    """
    Heuristic: choose a chunk size so the temporary complex array (npix × chunk)
    is ~ target_mb (default 256 MB). 16 bytes per complex128.
    """
    bytes_per_complex = 16
    if npix <= 0:
        return ncoef
    chunk = max(1, min(ncoef, (target_mb * (1024**2)) // (npix * bytes_per_complex)))
    return int(chunk)

def phase_and_intensity_vs_coeff_vec(
    mode2d: np.ndarray,           # (N,N) Zernike (dimensionless)
    R: np.ndarray, mask: np.ndarray, dA: float,
    u0: float,
    c_vals: np.ndarray,           # (C,) coefficients [rad]
    *,
    chunk_size: int = None,       # None -> auto by memory heuristic
    show_progress: bool = False,
    progress_desc: str = "coefficients"
):
    """
    Vectorized evaluation of A(c) over all c_vals in chunks.
    Returns (theta, I, A) with shapes (C,).

    Memory: builds temporary (P × chunk) complex array, where P=#mask pixels.
    """
    # Weights on pupil
    M = gaussian_on_pupil(R, mask, u0)
    w = M[mask]                          # (P,)
    m = mode2d[mask]                     # (P,)
    P = m.size
    C = len(c_vals)

    # Normalizations
    norm_E = math.sqrt(P * dA)
    norm_M = math.sqrt(np.sum(w**2) * dA)
    if norm_M == 0:
        th = np.full(C, 0.0)
        I  = np.zeros(C)
        A  = np.zeros(C, dtype=np.complex128)
        return th, I, A

    # Chunking plan
    if chunk_size is None:
        chunk_size = _pick_chunk_size(P, C, target_mb=256)  # tune MB if desired
    # Safety: at least 1
    chunk_size = max(1, int(chunk_size))

    # Output buffers
    A_all = np.empty(C, dtype=np.complex128)

    # Progress iterator
    use_tqdm = show_progress and (_tqdm is not None)
    it = range(0, C, chunk_size)
    if use_tqdm:
        it = _tqdm(it, total=(C + chunk_size - 1)//chunk_size, desc=progress_desc)

    # Compute in blocks
    # For each block of coefficients c_block (shape B,), build E = exp( i m[:,None] * c_block[None,:] )
    # Then A_block = (w^T @ E) * dA / (norm_E * norm_M)
    scale = dA / (norm_E * norm_M)
    for start in it:
        stop = min(start + chunk_size, C)
        c_block = c_vals[start:stop]                       # (B,)
        # Broadcasting to (P,B):
        # IMPORTANT: build the phase with outer product to avoid huge temporary when unnecessary
        # We explicitly construct E to leverage BLAS in the dot below.
        E = np.exp(1j * (m[:, None] * c_block[None, :]))   # (P,B) complex128
        A_block = (w @ E) * scale                          # (B,)
        A_all[start:stop] = A_block

    th = np.angle(A_all)
    I  = np.abs(A_all)**2
    return th, I, A_all
def _compute_one_mode(args):
    (k, mode2d, R, mask, dA, u0, c_vals, Imin, phase_mode, vec_chunk_mb) = args
    # Vectorized over c_vals with chunking by memory budget
    chunk = _pick_chunk_size(mode2d[mask].size, len(c_vals), target_mb=vec_chunk_mb)
    th, I, A = phase_and_intensity_vs_coeff_vec(
        mode2d, R, mask, dA, u0, c_vals,
        chunk_size=chunk,
        show_progress=False  # avoid noisy per-worker bars
    )
    # Phase selection (same as before)
    if phase_mode == "auto":
        m = ZERN_INFO[k]["m"]
        ph = np.where(I > Imin, th, np.nan) if m == 0 else phase_binary_from_real(A, I, Imin, eta=1e-3)
    elif phase_mode == "unwrap":
        valid = I > Imin
        ph = np.full_like(th, np.nan); ph[valid] = np.unwrap(th[valid], discont=np.pi/2)
    elif phase_mode == "real_binary":
        ph = phase_binary_from_real(A, I, Imin, eta=1e-3)
    else:
        ph = np.where(I > Imin, th, np.nan)
    return k, ph, I
def phase_and_intensity_vs_coeff(
    mode2d, R, mask, dA, u0, c_vals,
    *, vectorized=True, chunk_size=None, show_progress=False
):
    """
    Legacy signature. If vectorized=True (default), use the fast chunked path.
    Otherwise, fall back to your original incremental update loop.
    """
    if vectorized:
        return phase_and_intensity_vs_coeff_vec(
            mode2d, R, mask, dA, u0, c_vals,
            chunk_size=chunk_size, show_progress=show_progress,
            progress_desc="coefficients"
        )
    # ---- original incremental method (kept for reference/fallback) ----
    M = gaussian_on_pupil(R, mask, u0)
    m = mode2d[mask]; w = M[mask]
    norm_E = math.sqrt(m.size * dA)
    norm_M = math.sqrt(np.sum(w**2) * dA)
    th = np.empty_like(c_vals); I = np.empty_like(c_vals)
    E = np.ones_like(m, dtype=np.complex128)
    c0 = float(c_vals[0])
    if c0 != 0.0: E = np.exp(1j * (c0 * m))
    A = (np.sum(E*w)*dA)/(norm_E*norm_M) if norm_M else 0j
    th[0] = np.angle(A); I[0] = float(np.abs(A)**2)
    if len(c_vals) > 1:
        dc = float(c_vals[1]-c_vals[0]); q = np.exp(1j*(dc*m))
        for k in range(1, len(c_vals)):
            E *= q
            A = (np.sum(E*w)*dA)/(norm_E*norm_M) if norm_M else 0j
            th[k] = np.angle(A); I[k] = float(np.abs(A)**2)
    return th, I

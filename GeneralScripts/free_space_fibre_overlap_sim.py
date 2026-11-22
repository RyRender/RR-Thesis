# free_space_fibre_overlap_sim.py (v1.1)
import numpy as np
import matplotlib.pyplot as plt

def _to_knn(arr):
    """
    Ensure array has shape (K, N, N), where N is the pupil size and K=#modes.
    Accepts (K,N,N), (N,N,K), and tries to fix weird permutations robustly.
    """
    if arr.ndim == 2:
        return arr[None, ...]  # (1,N,N)

    if arr.ndim != 3:
        raise ValueError(f"Unsupported ndarray rank {arr.ndim}; expected 2D or 3D.")

    s0, s1, s2 = arr.shape
    # Ideal cases:
    if s1 == s2 and s0 != s1:
        # (K,N,N)
        return arr
    if s0 == s1 and s0 != s2:
        # (N,N,K) -> move last axis to front
        return np.moveaxis(arr, -1, 0)
    if s0 == s2 and s0 != s1:
        # (N,K,N) -> move axis 1 to front
        return np.moveaxis(arr, 1, 0)

    # Heuristic fallback:
    # Treat the axis whose size is smallest as K (modes), others as N,N
    sizes = np.array([s0, s1, s2])
    k_axis = int(np.argmin(sizes))
    if k_axis == 0:
        candidate = arr
    elif k_axis == 1:
        candidate = np.moveaxis(arr, 1, 0)
    else:
        candidate = np.moveaxis(arr, 2, 0)

    K, N1, N2 = candidate.shape
    if N1 != N2:
        raise ValueError(
            "Could not interpret array as a stack of square modes.\n"
            f"Original shape: {arr.shape}, candidate K-N-N shape: {candidate.shape}."
        )
    return candidate

def load_zernike_stack(path):
    arr = np.load(path)
    stack = _to_knn(arr)
    return stack.astype(np.float64)

def unit_disk_grid(n):
    xs = (np.arange(n) - (n - 1) / 2.0) * (2.0 / n)
    ys = (np.arange(n) - (n - 1) / 2.0) * (2.0 / n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    R = np.sqrt(X**2 + Y**2)
    mask = R <= 1.0 + 1e-12
    dA = (2.0 / n) ** 2
    return X, Y, R, mask, dA

def gaussian_mode_on_pupil(R, mask, u0):
    M = np.zeros_like(R, dtype=np.float64)
    if u0 <= 0:
        raise ValueError("u0 must be > 0 (fraction of pupil radius).")
    M[mask] = np.exp(-(R[mask] ** 2) / (u0 ** 2))
    return M

def combine_phi(stack, coeffs, active_indices=None):
    K, N, _ = stack.shape
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if active_indices is None:
        active_indices = np.arange(min(len(coeffs), K))
    else:
        active_indices = np.asarray(active_indices, dtype=int)
        if active_indices.size != coeffs.size:
            raise ValueError("active_indices and coeffs must have the same length.")
        if np.any(active_indices < 0) or np.any(active_indices >= K):
            raise ValueError(
                f"active_indices out of bounds for provided stack (K={K})."
            )
    phi = np.zeros((N, N), dtype=np.float64)
    for c, idx in zip(coeffs, active_indices):
        phi += c * stack[idx]
    return phi

def overlap_A(phi, mask, M, dA):
    P = mask.astype(np.float64)
    E = P * np.exp(1j * phi)
    num = np.sum(E[mask] * M[mask]) * dA
    norm_E = np.sqrt(np.sum((np.abs(E[mask]) ** 2)) * dA)
    norm_M = np.sqrt(np.sum((M[mask] ** 2)) * dA)
    if norm_E == 0 or norm_M == 0:
        return 0.0 + 0.0j, 0.0, 0.0
    A = num / (norm_E * norm_M)
    return A, norm_E, norm_M

def simulate_from_file(zernike_path, coeffs, active_indices=None, u0=0.30,
                       show_figures=True, show_phase=True, show_gaussian=True,
                       plot_argand=True, plot_coupling_bar=True):
    stack = load_zernike_stack(zernike_path)
    K, N, _ = stack.shape
    X, Y, R, mask, dA = unit_disk_grid(N)
    M = gaussian_mode_on_pupil(R, mask, u0)
    phi = combine_phi(stack, coeffs, active_indices=active_indices)
    phi_masked = np.zeros_like(phi)
    phi_masked[mask] = phi[mask]
    A, nE, nM = overlap_A(phi_masked, mask, M, dA)
    results = {
        "A_complex": A,
        "A_real": float(np.real(A)),
        "A_imag": float(np.imag(A)),
        "A_arg": float(np.angle(A)),
        "A_coupling": float(np.abs(A) ** 2),
        "norm_E": float(nE),
        "norm_M": float(nM),
        "N": int(N),
        "K_modes": int(K),
        "u0": float(u0),
        "active_indices": [] if active_indices is None else list(map(int, active_indices)),
        "coeffs": list(map(float, coeffs)),
    }
    if show_figures:
        if show_phase:
            plt.figure(figsize=(6, 5))
            im = plt.imshow(phi_masked, origin="lower")
            plt.title("Phase map φ (radians) on pupil")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()
        if show_gaussian:
            plt.figure(figsize=(6, 5))
            im2 = plt.imshow(M, origin="lower")
            plt.title("Gaussian mode M(r) on pupil")
            plt.colorbar(im2, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()
        if plot_argand:
            plt.figure(figsize=(5, 5))
            t = np.linspace(0, 2 * np.pi, 512)
            plt.plot(np.cos(t), np.sin(t), linewidth=1.0)
            plt.scatter([results["A_real"]], [results["A_imag"]])
            plt.axhline(0, linewidth=0.8)
            plt.axvline(0, linewidth=0.8)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlabel("Re(A)")
            plt.ylabel("Im(A)")
            plt.title("A on the Argand plane")
            plt.tight_layout()
            plt.show()
        if plot_coupling_bar:
            plt.figure(figsize=(5, 4))
            plt.bar(["||A||^2"], [results["A_coupling"]])
            plt.ylim(0, 1)
            plt.title("Coupling ||A||^2")
            plt.tight_layout()
            plt.show()
    return results

def scan_single_mode(zernike_path, mode_index, coeff_min=-3.14, coeff_max=3.14, num=101, u0=0.30):
    stack = load_zernike_stack(zernike_path)
    K, N, _ = stack.shape
    if mode_index < 0 or mode_index >= K:
        raise ValueError(f"mode_index must be in [0, {K-1}] for this stack.")
    X, Y, R, mask, dA = unit_disk_grid(N)
    M = gaussian_mode_on_pupil(R, mask, u0)
    coeffs = np.linspace(coeff_min, coeff_max, num)
    re_vals = np.empty_like(coeffs)
    im_vals = np.empty_like(coeffs)
    arg_vals = np.empty_like(coeffs)
    coup_vals = np.empty_like(coeffs)
    for i, c in enumerate(coeffs):
        phi = c * stack[mode_index]
        phi_masked = np.zeros_like(phi)
        phi_masked[mask] = phi[mask]
        A, nE, nM = overlap_A(phi_masked, mask, M, dA)
        re_vals[i] = np.real(A)
        im_vals[i] = np.imag(A)
        arg_vals[i] = np.angle(A)
        coup_vals[i] = np.abs(A) ** 2
    plt.figure(figsize=(6, 4))
    plt.plot(coeffs, re_vals)
    plt.xlabel("Coefficient (radians)")
    plt.ylabel("Re(A)")
    plt.title(f"Re(A) vs coefficient (mode {mode_index})")
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(6, 4))
    plt.plot(coeffs, im_vals)
    plt.xlabel("Coefficient (radians)")
    plt.ylabel("Im(A)")
    plt.title(f"Im(A) vs coefficient (mode {mode_index})")
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(6, 4))
    plt.plot(coeffs, arg_vals)
    plt.xlabel("Coefficient (radians)")
    plt.ylabel("arg(A)")
    plt.title(f"arg(A) vs coefficient (mode {mode_index})")
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(6, 4))
    plt.plot(coeffs, coup_vals)
    plt.xlabel("Coefficient (radians)")
    plt.ylabel("||A||^2")
    plt.title(f"Coupling ||A||^2 vs coefficient (mode {mode_index})")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
    return {
        "coeffs": coeffs,
        "ReA": re_vals,
        "ImA": im_vals,
        "ArgA": arg_vals,
        "Coupling": coup_vals,
    }

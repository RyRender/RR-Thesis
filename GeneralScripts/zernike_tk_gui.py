# zernike_tk_gui.py  — ultra-fast Tkinter GUI (Python 3.8+)
# - 15 Zernike sliders with range [-10π, +10π]
# - Phase and Gaussian heatmaps (separate), Argand plot (point on unit circle)
# - No Re/Im bar charts
# - Incremental updates (O(Nmask) per slider change, not 15×)
# - Static clim & pre-created artists, no autoscale per frame

import math
import numpy as np
from pathlib import Path
from typing import Union, List
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---- Matplotlib (Tk backend) ----
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# -------------------- Math helpers --------------------
def _to_knn(arr: np.ndarray) -> np.ndarray:
    """Coerce arr to (K, N, N). Accepts (K,N,N), (N,N,K), (N,K,N), or (N,N)."""
    if arr.ndim == 2:   # (N,N)
        return arr[None, ...].astype(np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported ndarray rank {arr.ndim}; expected 2D or 3D.")
    s0, s1, s2 = arr.shape
    if s1 == s2 and s0 != s1:        # (K,N,N)
        out = arr
    elif s0 == s1 and s0 != s2:      # (N,N,K)
        out = np.moveaxis(arr, -1, 0)
    elif s0 == s2 and s0 != s1:      # (N,K,N)
        out = np.moveaxis(arr, 1, 0)
    else:
        k_axis = int(np.argmin([s0, s1, s2]))
        out = arr if k_axis == 0 else np.moveaxis(arr, k_axis, 0)
        K, N1, N2 = out.shape
        if N1 != N2:
            raise ValueError(f"Could not coerce to (K,N,N). Original: {arr.shape}, candidate: {out.shape}")
    return out.astype(np.float64)

def load_zernike_stack(path: Union[str, Path]) -> np.ndarray:
    return _to_knn(np.load(path))

def unit_disk_grid(n: int):
    xs = (np.arange(n) - (n - 1) / 2.0) * (2.0 / n)
    ys = (np.arange(n) - (n - 1) / 2.0) * (2.0 / n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    R = np.sqrt(X**2 + Y**2)
    mask = R <= 1.0 + 1e-12
    dA = (2.0 / n) ** 2
    return X, Y, R, mask, dA


# -------------------- Tkinter GUI --------------------
class ZernikeTkApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Zernike Overlap Explorer (Tkinter, fast) — first 15 terms")
        self.geometry("1280x820")

        # Data / caches
        self.stack_full = None     # (K,N,N) full-res
        self.stack = None          # (K,N,N) after downsample
        self.K = 0
        self.N = 0
        self.eff_Ksel = 0          # effective K used (<=15 & <=K)
        self.mask = None
        self.mask_idx = None       # 1-D indices where mask==True
        self.dA = None
        self.R = None

        # Gaussian & field caches
        self.M_flat = None         # Gaussian on mask (1-D)
        self.M_img  = None         # 2-D Gaussian for display
        self.norm_E = None         # constant for phase-only field inside pupil
        self.norm_M = None         # depends on u0

        # Zernike-projection caches
        self.modes_flat = None     # shape (eff_Ksel, Nmask)
        self.coeffs = None         # last coeff values (eff_Ksel,)
        self.phi_flat = None       # current phase on mask (1-D)
        self.phi_img = None        # 2-D phase image for display
        self.E_flat = None         # exp(1j*phi_flat) on mask (1-D complex)

        self.pi10 = 10.0 * math.pi

        # ------------- Layout: left plots, right controls -------------
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        plot_frame = ttk.Frame(self)
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        ctrl_frame = ttk.Frame(self)
        ctrl_frame.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        for r in range(10):
            ctrl_frame.rowconfigure(r, weight=0)
        ctrl_frame.rowconfigure(9, weight=1)
        ctrl_frame.columnconfigure(0, weight=1)

        # ------------- Matplotlib figure -------------
        # 2 rows x 2 cols: [Phase, Gaussian] / [Argand, (blank)]
        self.fig = plt.Figure(figsize=(8, 5))
        self.ax_phase = self.fig.add_subplot(2, 2, 1)
        self.ax_gaus  = self.fig.add_subplot(2, 2, 2)
        self.ax_arg   = self.fig.add_subplot(2, 2, 3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        # Pre-create artists (filled later once we have data)
        self.im_phase = None
        self.im_gaus  = None
        self.cb_phase = None
        self.cb_gaus  = None
        # Argand artists
        t = np.linspace(0, 2*np.pi, 512)
        self.argand_circle, = self.ax_arg.plot(np.cos(t), np.sin(t), linewidth=1.0)
        self.argand_point,  = self.ax_arg.plot([0], [0], "o")
        self.ax_arg.axhline(0, linewidth=0.8)
        self.ax_arg.axvline(0, linewidth=0.8)
        self.ax_arg.set_aspect("equal", adjustable="box")
        self.ax_arg.set_xlim(-1.05, 1.05)
        self.ax_arg.set_ylim(-1.05, 1.05)
        self.ax_arg.set_xlabel("Re(A)")
        self.ax_arg.set_ylabel("Im(A)")
        self.ax_arg.set_title("A on the Argand plane")

        # ------------- Controls -------------
        # File picker
        file_row = ttk.Frame(ctrl_frame)
        file_row.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.path_var = tk.StringVar()
        ttk.Entry(file_row, textvariable=self.path_var).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(file_row, text="Load…", command=self.on_browse).pack(side="left")

        # Info label
        self.info_var = tk.StringVar(value="No file loaded")
        ttk.Label(ctrl_frame, textvariable=self.info_var).grid(row=1, column=0, sticky="w", pady=(0, 6))

        # Downsample selector (default 4x for speed; change to 1 for full fidelity)
        ds_row = ttk.Frame(ctrl_frame)
        ds_row.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(ds_row, text="Downsample:").pack(side="left")
        self.ds_var = tk.StringVar(value="4")
        ttk.OptionMenu(ds_row, self.ds_var, "4", "1", "2", "4", command=lambda *_: self.apply_downsample()).pack(side="left", padx=(6, 0))

        # u0 slider
        self.u0_var = tk.DoubleVar(value=0.30)
        u0_row = ttk.Frame(ctrl_frame)
        u0_row.grid(row=3, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(u0_row, text="u0").pack(side="left")
        ttk.Scale(u0_row, from_=0.05, to=0.80, variable=self.u0_var, command=self._on_u0_change).pack(side="left", fill="x", expand=True, padx=8)
        self.u0_entry = ttk.Entry(u0_row, width=6)
        self.u0_entry.insert(0, f"{self.u0_var.get():.3f}")
        self.u0_entry.pack(side="left")
        self.u0_var.trace_add("write", lambda *args: self._sync_u0_entry())
        self.u0_entry.bind("<Return>", lambda e: self._u0_from_entry())

        # Coefficient sliders in a scrollable pane
        slider_container = ttk.Frame(ctrl_frame, borderwidth=1, relief="solid")
        slider_container.grid(row=4, column=0, sticky="nsew")
        slider_container.rowconfigure(0, weight=1)
        slider_container.columnconfigure(0, weight=1)

        canvas = tk.Canvas(slider_container, highlightthickness=0)
        vsb = ttk.Scrollbar(slider_container, orient="vertical", command=canvas.yview)
        self.sliders_frame = ttk.Frame(canvas)
        self.sliders_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.sliders_frame, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # 15 sliders j1..j15, range [-10π, +10π], each with an index-aware callback
        self.Ksel = 15
        self.coeff_vars: List[tk.DoubleVar] = []
        self.slider_widgets = []
        for i in range(self.Ksel):
            row = ttk.Frame(self.sliders_frame, padding=(4, 2))
            row.grid(row=i, column=0, sticky="ew")
            ttk.Label(row, text=f"j{i+1}", width=5).pack(side="left")
            var = tk.DoubleVar(value=0.0)
            scl = ttk.Scale(row, from_=-self.pi10, to=self.pi10, variable=var,
                            command=lambda _val, idx=i: self._on_coeff_change(idx))
            scl.pack(side="left", fill="x", expand=True, padx=6)
            ent = ttk.Entry(row, width=12)
            ent.insert(0, "0.00")
            ent.pack(side="left")
            # sync entry<->scale
            def make_entry_sync(v=var, entry=ent, idx=i):
                v.trace_add("write", lambda *args: entry.delete(0, "end") or entry.insert(0, f"{v.get():.2f}"))
                entry.bind("<Return>", lambda e: self._coeff_from_entry(v, entry, idx))
            make_entry_sync()
            self.coeff_vars.append(var)
            self.slider_widgets.append(scl)

        # Buttons
        btn_row = ttk.Frame(ctrl_frame)
        btn_row.grid(row=5, column=0, sticky="ew", pady=(8, 4))
        ttk.Button(btn_row, text="Zero all", command=self.on_zero_all).pack(side="left")
        ttk.Button(btn_row, text="Randomize", command=self.on_randomize).pack(side="left", padx=6)

        # Status line for A
        self.status_var = tk.StringVar(value="Re(A)=…  Im(A)=…  arg(A)=…  ||A||²=…")
        ttk.Label(ctrl_frame, textvariable=self.status_var, font=("Courier New", 10)).grid(row=6, column=0, sticky="ew", pady=(6, 0))

        # Debounce timer
        self._debounce_id = None

        # Try to auto-load if present
        for guess in ("newzernike_stack.npy", "newzernike.npy"):
            if Path(guess).exists():
                self.path_var.set(guess)
                self.load_file(Path(guess))
                break

    # ---------- UI helpers ----------
    def _sync_u0_entry(self):
        try:
            self.u0_entry.delete(0, "end")
            self.u0_entry.insert(0, f"{self.u0_var.get():.3f}")
        except Exception:
            pass

    def _u0_from_entry(self):
        try:
            val = float(self.u0_entry.get())
            val = min(max(val, 0.05), 0.80)
            self.u0_var.set(val)
        except ValueError:
            pass

    def on_browse(self):
        path = filedialog.askopenfilename(
            title="Open Zernike .npy",
            filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")]
        )
        if path:
            self.path_var.set(path)
            self.load_file(Path(path))

    def on_zero_all(self):
        if self.coeffs is None:
            return
        # Reset coeffs and rebuild phi/E in one pass (fast)
        for v in self.coeff_vars:
            v.set(0.0)
        self.coeffs[:] = 0.0
        self.phi_flat[:] = 0.0
        self.E_flat[:] = 1.0 + 0.0j
        self._write_phi_img()
        self._redraw_all()

    def on_randomize(self):
        if self.coeffs is None:
            return
        rng = np.random.default_rng()
        new_vals = rng.uniform(-1.0, 1.0, size=self.eff_Ksel)
        # Apply new coeffs by delta per mode (single pass)
        deltas = new_vals - self.coeffs
        self.coeffs[:] = new_vals
        for k in range(self.eff_Ksel):
            dk = deltas[k]
            if dk != 0.0:
                add = dk * self.modes_flat[k]
                self.phi_flat += add
                self.E_flat *= np.exp(1j * add)
        # Update sliders/entries visually
        for i, v in enumerate(self.coeff_vars[:self.eff_Ksel]):
            v.set(float(self.coeffs[i]))
        self._write_phi_img()
        self._redraw_all()

    # Per-slider incremental update (very fast)
    def _on_coeff_change(self, idx: int):
        if self.stack is None or self.coeffs is None or idx >= self.eff_Ksel:
            return
        # Debounce tightly to avoid excessive draws while dragging
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(10, lambda i=idx: self._apply_coeff_delta(i))

    def _apply_coeff_delta(self, idx: int):
        new = float(self.coeff_vars[idx].get())
        dk = new - self.coeffs[idx]
        if dk == 0.0:
            return
        self.coeffs[idx] = new

        add = dk * self.modes_flat[idx]        # (Nmask,)
        self.phi_flat += add                    # update phase
        self.E_flat *= np.exp(1j * add)        # update field (only 1 exp over Nmask)

        self._write_phi_img()
        self._redraw_all()

    def _on_u0_change(self, _=None):
        # Debounce u0 recompute (it rebuilds Gaussian & its norm)
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(20, self._rebuild_gaussian_and_redraw)

    def _coeff_from_entry(self, var: tk.DoubleVar, entry: ttk.Entry, idx: int):
        try:
            val = float(entry.get())
            val = min(max(val, -self.pi10), self.pi10)
            var.set(val)
            self._on_coeff_change(idx)
        except ValueError:
            pass

    # ---------- Data prep ----------
    def load_file(self, path: Path):
        try:
            arr = load_zernike_stack(path)
        except Exception as e:
            messagebox.showerror("Load error", f"Could not load file:\n{e}")
            return
        self.stack_full = arr
        self.apply_downsample()
        self.update_info()

    def apply_downsample(self):
        if self.stack_full is None:
            return
        try:
            f = int(self.ds_var.get())
        except Exception:
            f = 1
        if f not in (1, 2, 4):
            f = 1

        self.stack = self.stack_full[:, ::f, ::f]
        self.K, self.N, _ = self.stack.shape
        self.eff_Ksel = min(self.Ksel, self.K)

        # Build grid & mask
        _, _, R, mask, dA = unit_disk_grid(self.N)
        self.R, self.mask, self.dA = R, mask, dA
        self.mask_idx = np.flatnonzero(mask.ravel())
        Nmask = self.mask_idx.size

        # Precompute flat modes on the pupil mask (eff_Ksel x Nmask)
        if self.eff_Ksel > 0:
            self.modes_flat = np.empty((self.eff_Ksel, Nmask), dtype=np.float64)
            for k in range(self.eff_Ksel):
                self.modes_flat[k] = self.stack[k].ravel()[self.mask_idx]
        else:
            self.modes_flat = np.zeros((0, Nmask), dtype=np.float64)

        # Allocate images & vectors
        self.phi_img  = np.zeros((self.N, self.N), dtype=np.float64)
        self.M_img    = np.zeros((self.N, self.N), dtype=np.float64)
        self.phi_flat = np.zeros(Nmask, dtype=np.float64)
        self.E_flat   = np.ones(Nmask, dtype=np.complex128)
        self.coeffs   = np.zeros(self.eff_Ksel, dtype=np.float64)

        # Precompute norm_E (amplitude 1 inside the pupil)
        self.norm_E = math.sqrt(Nmask * self.dA)

        # (Re)build Gaussian & draw everything once
        self._rebuild_gaussian_and_redraw(first_time=True)
        self.update_info()

    def update_info(self):
        if self.stack is None:
            self.info_var.set("No file loaded")
        else:
            self.info_var.set(
                f"File: {self.path_var.get()}   |   Modes: {self.K}   |   Using first: {self.eff_Ksel}   |   N: {self.N}"
            )

    # ---------- Gaussian & drawing ----------
    def _rebuild_gaussian_and_redraw(self, first_time: bool=False):
        # Build Gaussian for current u0 on mask + image for display; update its norm
        u0 = float(self.u0_var.get())
        self.M_img.fill(0.0)
        inside = self.mask
        self.M_img[inside] = np.exp(-(self.R[inside] ** 2) / (u0 ** 2))
        self.M_flat = self.M_img.ravel()[self.mask_idx]
        self.norm_M = math.sqrt(np.sum(self.M_flat ** 2) * self.dA)
        self._write_phi_img()
        self._redraw_all(first_time=first_time)

    def _write_phi_img(self):
        # Write masked phi_flat back to 2-D image buffer
        flat = self.phi_img.ravel()
        flat[:] = 0.0
        flat[self.mask_idx] = self.phi_flat

    def _compute_A(self):
        if self.norm_E and self.norm_M:
            num = np.sum(self.E_flat * self.M_flat) * self.dA
            return num / (self.norm_E * self.norm_M)
        return 0.0 + 0.0j

    def _redraw_all(self, first_time: bool=False):
        A = self._compute_A()

        # Phase heatmap (fixed clim -> no autoscale cost)
        if self.im_phase is None:
            self.ax_phase.set_title("Phase map φ (radians) on pupil")
            self.im_phase = self.ax_phase.imshow(self.phi_img, origin="lower",
                                                 vmin=-self.pi10, vmax=self.pi10)
            self.cb_phase = self.fig.colorbar(self.im_phase, ax=self.ax_phase,
                                              fraction=0.046, pad=0.04)
        else:
            self.im_phase.set_data(self.phi_img)
            # no autoscale / clim changes per frame

        # Gaussian heatmap (fixed [0,1] range)
        if self.im_gaus is None or first_time:
            self.ax_gaus.set_title("Gaussian mode M(r) on pupil")
            if self.im_gaus is None:
                self.im_gaus = self.ax_gaus.imshow(self.M_img, origin="lower",
                                                   vmin=0.0, vmax=1.0)
                self.cb_gaus = self.fig.colorbar(self.im_gaus, ax=self.ax_gaus,
                                                 fraction=0.046, pad=0.04)
            else:
                self.im_gaus.set_data(self.M_img)
        else:
            self.im_gaus.set_data(self.M_img)

        # Argand point (just move the marker)
        self.argand_point.set_data([float(np.real(A))], [float(np.imag(A))])

        # Status line
        self.status_var.set(
            f"Re(A)={np.real(A): .6f}    Im(A)={np.imag(A): .6f}    arg(A)={np.angle(A): .6f}    ||A||²={np.abs(A)**2: .6f}"
        )

        self.canvas.draw_idle()


# -------------------- Run --------------------
if __name__ == "__main__":
    app = ZernikeTkApp()
    app.mainloop()

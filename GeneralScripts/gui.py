import logging
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
from koheron import connect
from aompc import AOMPhaseControl
import numpy as np
import io
import sys
import threading


class MainApplication(tk.ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.grid(column=0, row=0, sticky="n, e, s, w")

        self.ancestors = [parent]
        self.widgets = {}
        self.requires_connection = []
        self.threads = []
        self.terminate_event = threading.Event()

        self.driver = AOMPhaseControl
        self.driver_lock = threading.Lock()

        self.connected = False
        self.phase_stab_enabled = False
        self.fft_locking_enabled = False
        self.host = tk.StringVar()
        self.host.set("169.254.1.110")
        self.sp = tk.StringVar()
        self.kp = tk.StringVar()
        self.kp.set("0")

        self.get_bgcolour()

        self.setup_plots()
        self.create_control_frame()
        self.create_log_frame()

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # Prepare initial GUI state
        self.disable_widgets()
        self.widgets['phase_stab_enable_button'].focus()

    def get_bgcolour(self):
        # https://stackoverflow.com/questions/66489340/transparent-background-with-matplotlib-and-tkinter
        s = ttk.Style()
        bg = s.lookup("TFrame", "background")
        bg_16bit = self.ancestors[0].winfo_rgb(bg)
        self.bg_string = "#" + "".join([hex(bg_color >> 8)[2:]
                                        for bg_color in bg_16bit])

    def create_control_frame(self):
        # Set up frame for interface elements
        self.control_frame = ttk.Frame(master=self, padding="3 3 12 12")
        self.control_frame.grid(column=1, row=1)
        self.control_frame.ancestors = self.ancestors + [self]

        # Host config
        ttk.Label(self.control_frame, text="Host").grid(
            column=1, row=1, sticky="w")
        host_entry = ttk.Entry(
            self.control_frame, width=14, textvariable=self.host)
        host_entry.grid(column=2, row=1, sticky="w, e")
        ttk.Button(
            self.control_frame,
            text="Connect",
            command=self.connect_client
        ).grid(column=3, row=1, sticky="w")

        # Feedback config
        ttk.Label(self.control_frame, text="SP").grid(
            column=1, row=2, sticky="w")
        sp_entry = ttk.Entry(
            self.control_frame, width=14, textvariable=self.sp, state="disable")
        sp_entry.grid(column=2, row=2, sticky="w, e")
        sp_button = ttk.Button(
            self.control_frame, text="Update", command=self.update_sp)
        sp_button.grid(column=3, row=2, sticky="w")
        ttk.Label(self.control_frame, text="Kp").grid(
            column=1, row=3, sticky="w")
        kp_entry = ttk.Entry(
            self.control_frame, width=14, textvariable=self.kp, state="disable")
        kp_entry.grid(column=2, row=3, sticky="w, e")
        kp_button = ttk.Button(
            self.control_frame, text="Update", command=self.update_kp)
        kp_button.grid(column=3, row=3, sticky="w")

        ttk.Label(self.control_frame, text="Feedback").grid(
            column=1, row=4, sticky="w")
        enable_button = ttk.Button(
            self.control_frame, text="Enable", command=self.toggle_phase_stab)
        enable_button.grid(column=2, row=4, sticky="e")
        optimise_button = OptimiseKpButton(
            self.control_frame, text="Optimise")
        optimise_button.grid(column=3, row=4, sticky="w")
        guess_button = ttk.Button(
            self.control_frame, text="Guess", command=self.guess
        )
        guess_button.grid(column=3, row=5, sticky="w")

        # FFT laser locking
        ttk.Label(self.control_frame, text="FFT laser locking").grid(
            column=1, row=7, sticky="w")
        fft_button = ttk.Button(
            self.control_frame, text="Enable", command=self.toggle_fft_locking)
        fft_button.grid(column=3, row=7, sticky="e")

        self.requires_connection += [
            sp_entry,
            sp_button,
            kp_entry,
            kp_button,
            enable_button,
            optimise_button,
            fft_button,
        ]

        self.widgets.update({
            'sp_entry': sp_entry,
            'sp_button': sp_button,
            'kp_entry': kp_entry,
            'kp_button': kp_button,
            'phase_stab_enable_button': enable_button,
            'optimise_button': optimise_button,
            'fft_locking_enable_button': fft_button
        })

    def guess(self):
        self.driver.update_setpoint()
        self.driver.update_kp()
        self.sp.set(str(self.driver.get_pi_sp()))
        self.kp.set(str(self.driver.get_pi_kp()))
        # self.driver.set_min_phase_incr(int(1357209666 * 79.9999 / 79))
        # self.driver.set_max_phase_incr(int(1391569404 * 80.0001 / 81))

    def create_log_frame(self):
        self.log_frame = ttk.Frame(self, padding="3 3 12 12")
        self.log_frame.grid(column=1, row=2, sticky="s")
        ttk.Label(self.log_frame, text="Log").grid(
            column=1, row=1, columnspan=3)
        self.log = tk.Text(self.log_frame, height=20, width=60,
                           font=tk.font.Font(size=10))
        self.log.grid(column=1, row=2, columnspan=3, pady=(5, 30))
        self.log.bind("<Key>", lambda e: "break")
        self.log.bind("<Button-1>", lambda e: "break")
        save_log = ttk.Button(self.log_frame, text="Save log", state="disable")
        save_log.grid(column=1, row=6)
        copy_log = ttk.Button(self.log_frame, text="Copy log", state="disable")
        copy_log.grid(column=2, row=6)
        clear_log = ttk.Button(self.log_frame, text="Clear log",
                               command=lambda: self.log.delete(1.0, tk.END))
        clear_log.grid(column=3, row=6)

    def setup_plots(self):
        self.plots_frame = PlotsFrame(self)
        self.plots_frame.grid(column=2, row=1, rowspan=2)
        signal_widget = self.create_signal_plot_widget(self.plots_frame)
        psd_widget = self.create_psd_plot_widget(self.plots_frame)
        self.threads += [
            threading.Thread(
                target=self.update_signal_plot_widget,
                args=(signal_widget,),
                daemon=True
            ),
            threading.Thread(
                target=self.update_psd_plot_widget,
                args=(psd_widget,),
                daemon=True
            )
        ]

    def enable_widgets(self):
        '''
        Enable all widgets that require a connection to the device.
        '''
        for widget in self.requires_connection:
            widget.config(state='enable')

    def disable_widgets(self):
        '''
        Disable all widgets that require a connection to the device.
        '''
        for widget in self.requires_connection:
            widget.config(state='disable')

    def connect_client(self):
        '''
        Attempt to connect to the device
        '''
        with self.driver_lock:
            try:
                client = connect(
                    self.host.get(), name='aompc', restart=True)
                self.driver = AOMPhaseControl(client)
                logging.info("Connected to device")
                self.enable_widgets()
                self.connected = True
                for thread in self.threads:
                    thread.start()
            except Exception:
                self.connected = False
                logging.warning("Failed to connect to device")

    def update_sp(self):
        self.driver.set_pi_sp(int(self.sp.get()))

    def update_kp(self):
        self.driver.set_pi_kp(int(self.kp.get()))

    def toggle_phase_stab(self):
        self.phase_stab_enabled = not self.phase_stab_enabled

        with self.driver_lock:
            if self.phase_stab_enabled:
                self.widgets['phase_stab_enable_button'].config(text="Disable")
                self.driver.enable_feedback()
            else:
                self.widgets['phase_stab_enable_button'].config(text="Enable")
                self.driver.disable_feedback()

    def toggle_fft_locking(self):
        # TODO: make this get enable/disable state from driver object
        self.fft_locking_enabled = not self.fft_locking_enabled

        with self.driver_lock:
            if self.fft_locking_enabled:
                self.widgets['fft_locking_enable_button'].config(
                    text="Disable")
                self.driver.enable_fft_locking()
            else:
                self.widgets['fft_locking_enable_button'].config(text="Enable")
                self.driver.disable_fft_locking()

    def create_signal_plot_widget(self, plots_frame):
        plot_widget = plots_frame.create_plot_widget('signal')
        plot_widget.t_us = np.arange(16384) / 250
        plot_widget.widget.grid(column=1, row=1)
        plot_widget.widget.config(bg=self.bg_string)

        ax = plot_widget.figure.add_subplot(111)
        ax.set_xlabel('time (us)')
        ax.set_ylabel('signal (V)')
        ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
        ax.set_xlim(plot_widget.t_us[0], plot_widget.t_us[-1])
        ax.set_ylim((-0.5, 0.5))

        plot_widget.lines = [
            Line2D([], [], color='blue', label='ADC 1'),
            Line2D([], [], color='green', label='ADC 2')
        ]

        for line in plot_widget.lines:
            ax.add_line(line)
        return plot_widget

    def update_signal_plot_widget(self, widget):

        adc0 = [widget.t_us, None]
        adc1 = [widget.t_us, None]

        while not self.terminate_event.is_set():
            with self.driver_lock:
                self.driver.trigger_acquisition()
                self.driver.get_adc()
                adc0[1] = self.driver.adc[0, :] / 2**16
                adc1[1] = self.driver.adc[1, :] / 2**16

            widget.update_lines(adc0, adc1)
            widget.draw()

    def create_psd_plot_widget(self, plots_frame):
        plot_widget = plots_frame.create_plot_widget('psd')
        plot_widget.f = np.linspace(0, 125, 4096, endpoint=False)
        plot_widget.widget.grid(column=1, row=2)
        plot_widget.widget.config(bg=self.bg_string)

        ax = plot_widget.figure.add_subplot(111)
        ax.set_xlabel("Frequency / MHz")
        ax.set_ylabel("Raw PSD")
        ax.set_xlim(0, 125)
        ax.set_ylim(0, 1e1)

        plot_widget.lines = [
            Line2D([], [], color='blue'),
        ]

        for line in plot_widget.lines:
            ax.add_line(line)

        return plot_widget

    def update_psd_plot_widget(self, widget):
        psd = [widget.f, None]
        while not self.terminate_event.is_set():
            with self.driver_lock:
                psd[1] = self.driver.get_psd()
            widget.update_lines(psd)
            widget.draw()

    def terminate_threads(self):
        self.terminate_event.set()
        self.ancestors[0].destroy()


class PlotsFrame(tk.ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.ancestors = parent.ancestors + [parent]
        self.widgets = {}

    def create_plot_widget(self, name,  *args, **kwargs):
        self.widgets[name] = PlotWidget(self, *args, **kwargs)
        return self.widgets[name]


class PlotWidget():
    def __init__(self, parent, *args, **kwargs):
        self.ancestors = parent.ancestors + [parent]
        self.figure = plt.Figure()
        self.figure.set_facecolor('none')
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.widget = self.canvas.get_tk_widget()

    def update_lines(self, *args):
        for line, data in zip(self.lines, args):
            if data is None:
                data = (None, None)
            line.set_data(*data)
        self.figure.canvas.draw()

    def draw(self):
        self.figure.canvas.draw()


class OptimiseKpButton(tk.ttk.Button):
    def __init__(self, parent, *args, **kwargs):
        tk.ttk.Button.__init__(self, parent, *args,
                               command=self.optimise, **kwargs)
        self.ancestors = parent.ancestors + [parent]
        self.widgets = self.ancestors[1].widgets

    def _run_kp_optimisation(self):
        kp = 0
        with self.ancestors[1].driver_lock:
            self.ancestors[1].driver.optimise_kp()
            kp = self.ancestors[1].driver.kp
        logging.info(f"Optimisation complete. Obtained value: {kp}")

        # Update GUI
        self.widgets['optimise_button'].config(state="enable", text="Optimise")
        self.widgets['kp_entry'].config(state="enable")
        self.ancestors[1].kp.set(str(kp))
        self.widgets['kp_button'].config(state="enable")

    def optimise(self):
        if not self.ancestors[1].phase_stab_enabled:
            self.ancestors[1].toggle_phase_stab()

        # Disable buttons
        self.config(state="disable", text="Optimising...")
        self.widgets['kp_entry'].config(state="disable")
        self.widgets['kp_button'].config(state="disable")

        # Perform optimisation
        logging.info("Running optimisation")
        optimise_thread = threading.Thread(target=self._run_kp_optimisation)
        optimise_thread.start()


class LogStdOut(io.TextIOBase):
    def __init__(self, log_dest):
        self.log_dest = log_dest

    def write(self, s):
        self.log_dest.insert(tk.END, s)
        self.log_dest.see(tk.END)


def main():
    # Configure app root
    root = tk.Tk()
    root.title("AOM Phase Control")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    app = MainApplication(root, padding="60 3 12 12")

    # Set up logging
    sys.stdout = LogStdOut(app.log)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='[%(levelname)s]: %(message)s')

    root.protocol("WM_DESTROY_WINDOW", app.terminate_threads)
    root.mainloop()


if __name__ == '__main__':
    main()

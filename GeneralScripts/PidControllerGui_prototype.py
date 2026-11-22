import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import pyvisa
from pylablib.devices import Thorlabs
from datetime import datetime
import time

# Initialize global variables for x and y data, alpha values, and plotting settings
x_data = []
y_data = []
x_voltage=[]
y_voltage=[]
time_data = []
power_data = []
alpha_values = []
start_time = None
time_plot_data = []
current_index = 0
plot_active = False
settings_file = "settings.json"
pm160_device = None
kpa_device = None

def get_unique_filename(base_name):
    """
    Generates a unique filename by appending a number if the file already exists.
    """
    if not os.path.exists(base_name):
        return base_name
    
    # Extract the base name and extension
    name, ext = os.path.splitext(base_name)
    counter = 1
    
    # Keep generating new filenames until one is unique
    while os.path.exists(f"{name}_{counter}{ext}"):
        counter += 1
    
    return f"{name}_{counter}{ext}"

def generate_filename(device_type):
    """
    Generates a filename based on the current date, PID values, and polling rate.
    Args:
        device_type (str): Either 'KPA' or 'PM'.
    Returns:
        str: The generated filename.
    """
    date_str = datetime.now().strftime("%d%m%y")
    
    # Retrieve PID values and replace decimal points with an empty string
    proportional = proportional_entry.get().strip().replace('.', '')
    integral = integral_entry.get().strip().replace('.', '')
    differential = gain_entry.get().strip().replace('.', '')  # Assuming 'gain' is used as the differential term

    # Format the polling rate as an integer to avoid decimals
    polling_rate = polling_rate_var.get()
    polling_rate_formatted = f"{int(polling_rate):02d}"

    # Construct the filename
    return f"{device_type}_{date_str}_{proportional}_{integral}_{differential}_{polling_rate_formatted}"

def log_message(message):
    """
    Inserts a message at the bottom of the log_textbox and automatically scrolls to the bottom.
    """
    log_textbox.insert(tk.END, message)
    log_textbox.see(tk.END)  # Automatically scroll to the bottom to show the latest message

# Function to start the plot
def start_plot():
    global plot_active
    plot_active = True
    update_plot()

# Function to stop the plot
def stop_plot():
    global plot_active
    plot_active = False

def plot_kpa():
    """
    Records xdiff and ydiff from the KPA device, along with a timestamp and power data from the PM device
    at the rate defined by the polling rate.
    """
    global kpa_device, pm160_device, x_data, y_data, x_voltage, y_voltage, time_data, power_data, time_plot_data, plot_active, start_time

    if not plot_active:
        return  # Exit if not actively plotting

    # Initialize the start time when plotting begins
    if start_time is None:
        start_time = time.time()  # Set the initial start time

    # Check if KPA device is connected
    if kpa_device is None:
        log_message("KPA is not connected. Cannot continue plotting.\n")
        return

    try:
        # Get the readings from the KPA device
        xdiff, ydiff, sum_val, xpos, ypos = kpa_device.get_readings()

        # Record the current time
        timestamp = time.time()

        # Calculate the elapsed time since start and store it in time_plot_data
        time_elapsed = timestamp - start_time  # Time since the plot started
        time_plot_data.append(time_elapsed)

        # Append KPA readings and timestamp to the data arrays
        x_data.append(xdiff)
        y_data.append(ydiff)
        x_voltage.append(xpos)
        y_voltage.append(ypos)
        time_data.append(timestamp)

        # Check if the PM device is connected
        if pm160_device is not None:
            try:
                # Get the power reading from the PM device
                power = pm160_device.get_reading()  # Assuming `get_reading()` returns a power value
                power_data.append(power)  # Append the power reading to power_data array
            except Exception as e:
                log_message(f"Failed to get PM reading: {e}\n")
        else:
            log_message("PM is not connected. Skipping power recording.\n")

        # Get the polling rate in milliseconds
        polling_rate_ms = int(polling_rate_var.get() * 1000)

        # Schedule the next call to this function based on the polling rate if active
        if plot_active:
            root.after(polling_rate_ms, plot_kpa)

    except Exception as e:
        log_message(f"Failed to get KPA readings: {e}\n")

def start_plotting_kpa():
    """
    Starts the plotting process by setting plot_active to True and scheduling the first plot_kpa call.
    """
    global plot_active
    if not plot_active:  # Only start if not already active
        plot_active = True
        log_message("Started plotting KPA readings.\n")
        plot_kpa()  # Start plotting

def stop_plotting_kpa():
    """
    Stops the plotting process by setting plot_active to False.
    """
    global plot_active
    if plot_active:
        plot_active = False
        log_message("Stopped plotting KPA readings.\n")

def update_plot():
    global current_index, alpha_values, plot_active, line2  # Ensure line2 is accessible

    if plot_active and len(x_data) > 0 and len(y_data) > 0:
        # Get the time interval and number of points from the entry boxes
        try:
            time_interval = int(interval_entry.get())
        except ValueError:
            time_interval = 1000  # Default to 1000 ms if invalid input
        
        try:
            max_points = int(points_entry.get())
        except ValueError:
            max_points = len(x_data)  # Default to all points if invalid input

        # Calculate the alpha decrement per update based on max_points
        alpha_decrement = 1.0 / (max_points * 10)

        # Ensure alpha_values has entries for all points
        while len(alpha_values) < len(x_data):
            alpha_values.append(1.0)

        # Update alpha values over time for currently visible points
        for i in range(max(0, current_index - max_points + 1), current_index + 1):
            alpha_values[i] = max(0, alpha_values[i] - alpha_decrement)

        # Extract the last `max_points` data points for plotting on ax1
        start_index = max(0, current_index - max_points + 1)
        xs = [x_data[i] for i in range(start_index, current_index + 1) if alpha_values[i] > 0]
        ys = [y_data[i] for i in range(start_index, current_index + 1) if alpha_values[i] > 0]
        alphas = [alpha_values[i] for i in range(start_index, current_index + 1) if alpha_values[i] > 0]

        # Check if the colormap should be inverted
        if invert_colormap_var.get():
            alphas = alphas[::-1]

        # Get the selected colormap
        selected_colormap = colormap_combo.get()

        # Capture current x and y limits on ax1 to preserve zoom level
        xlim, ylim = ax1.get_xlim(), ax1.get_ylim()

        # Remove the previous scatter plot collection without clearing the axis entirely
        for collection in ax1.collections:
            collection.remove()

        # Update scatter plot with selected colormap on ax1
        if xs and ys:
            ax1.scatter(xs, ys, c=alphas, alpha=1, marker='x', cmap=selected_colormap, s=100)

        # Reapply the captured x and y limits to maintain zoom on ax1
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

        # Update Power vs Time plot on ax2 with the last max_points entries
        if line2 is not None:  # Ensure line2 is defined
            line2.set_data(time_plot_data[-max_points:], power_data[-max_points:])

        # Adjust ax2's x and y limits to fit the data
        ax2.relim()      # Recalculate limits based on the new data
        ax2.autoscale()  # Automatically rescale the axis

        # Update labels and grid on ax2
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Power")
        ax2.grid(True, linestyle='--', linewidth=0.5)
        
        # Keep other plot elements for ax1
        ax1.set_xlabel("X Voltage")
        ax1.set_ylabel("Y Voltage")
        ax1.set_aspect('equal', adjustable='box')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.set_xticks(np.arange(-5, 6, 1))
        ax1.set_yticks(np.arange(-5, 6, 1))

        # Redraw the canvas with both updated plots
        canvas.draw()

        # Move to the next point if we haven't reached the end of the data
        if current_index < len(x_data) - 1:
            current_index += 1

        # Continue updating the plot after the specified interval
        root.after(time_interval, update_plot)
    else:
        plot_active = False  # Stop updating if the plot is not active


# Function to reset the plot without starting
def reset_plot():
    global current_index, alpha_values, plot_active, line2
    plot_active = False  # Stop any ongoing plotting
    current_index = 0  # Reset the index to start from the beginning
    alpha_values = [1.0] * len(x_data)  # Reset alpha values for all data points

    # Clear the X-Y plot (ax1) and prepare it for new data without starting the plotting
    ax1.clear()
    ax1.set_xlabel("X Voltage")
    ax1.set_ylabel("Y Voltage")
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_xticks(np.arange(-5, 6, 1))  # Set ticks from -5 to 5 with step 1 on x-axis
    ax1.set_yticks(np.arange(-5, 6, 1))  # Set ticks from -5 to 5 with step 1 on y-axis

    # Clear the data in Power vs. Time plot (ax2) by setting line2 data to empty lists
    line2.set_data([], [])  # Reset the line plot to an empty state
    ax2.relim()             # Reset axis limits
    ax2.autoscale()         # Autoscale with the empty data

    # Redraw the canvas to reflect the reset state
    canvas.draw()

    # Log the reset action
    log_textbox.insert(tk.END, "Plot reset. Ready to start.\n")

# Function to load a CSV and extract x and y values without starting the plot
def load_csv():
    global x_data, y_data, alpha_values, current_index
    # Check if debug mode is enabled before loading CSV
    if not debug_toggle_var.get():
        log_textbox.insert(tk.END, "Debug mode is off. Cannot load CSV.\n")
        return

    file_path = debug_entry.get()
    if file_path:
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # Extract x and y data
            x_data = df.iloc[:, 0].to_numpy()
            y_data = df.iloc[:, 1].to_numpy()
            alpha_values = [1.0] * len(x_data)  # Start all points with full opacity
            current_index = 0  # Reset index when new data is loaded
            log_textbox.insert(tk.END, f"Loaded CSV: {file_path}\n")
            log_textbox.insert(tk.END, f"Loaded {len(x_data)} points from CSV. Ready to start plotting.\n")
            
            # Prepare the plot without starting the plot automatically
            reset_plot()
        except Exception as e:
            log_textbox.insert(tk.END, f"Failed to load CSV: {e}\n")
    else:
        log_textbox.insert(tk.END, "No CSV file selected.\n")

# Function to load a CSV file and insert its path into the debug text box
def browse_csv():
    file_path = filedialog.askopenfilename(title="Open CSV", filetypes=[("CSV Files", "*.csv")])
    if file_path:
        debug_entry.delete(0, tk.END)
        debug_entry.insert(0, file_path)

# Function to save settings to a JSON file
def save_settings():
    settings = {
        "time_interval": interval_entry.get(),
        "max_points": points_entry.get(),
        "proportional": proportional_entry.get(),
        "integral": integral_entry.get(),
        "gain": gain_entry.get(),
        "pm": pm_entry.get(),
        "kpa": kpa_entry.get(),
        "csv_path": debug_entry.get(),
        "debug_toggle": debug_toggle_var.get(),
        "polling_rate": polling_rate_var.get(),  # Save the polling rate
        "colormap": colormap_combo.get(),  # Save the selected colormap
        "invert_colormap": invert_colormap_var.get()  # Save the invert colormap state
    }
    with open(settings_file, "w") as file:
        json.dump(settings, file)
    log_textbox.insert(tk.END, "Settings saved.\n")
def load_settings():
    if os.path.exists(settings_file):
        with open(settings_file, "r") as file:
            settings = json.load(file)
            interval_entry.delete(0, tk.END)
            interval_entry.insert(0, settings.get("time_interval", "1000"))
            
            points_entry.delete(0, tk.END)
            points_entry.insert(0, settings.get("max_points", "1000"))
            
            proportional_entry.delete(0, tk.END)
            proportional_entry.insert(0, settings.get("proportional", ""))
            
            integral_entry.delete(0, tk.END)
            integral_entry.insert(0, settings.get("integral", ""))
            
            gain_entry.delete(0, tk.END)
            gain_entry.insert(0, settings.get("gain", ""))

            pm_entry.delete(0, tk.END)
            pm_entry.insert(0, settings.get("pm", ""))
            
            kpa_entry.delete(0, tk.END)
            kpa_entry.insert(0, settings.get("kpa", ""))
            
            debug_entry.delete(0, tk.END)
            csv_path = settings.get("csv_path", "")
            debug_entry.insert(0, csv_path)

            # Load the debug toggle state before attempting to load the CSV
            debug_toggle_var.set(settings.get("debug_toggle", True))

            # Load the polling rate
            polling_rate_var.set(settings.get("polling_rate", 1.0))
            polling_entry.delete(0, tk.END)
            polling_entry.insert(0, str(polling_rate_var.get()))

            # Auto-fill filenames for KPA and PM recordings
            record_kpa_entry.delete(0, tk.END)
            record_kpa_entry.insert(0, generate_filename("KPA"))

            record_pm_entry.delete(0, tk.END)
            record_pm_entry.insert(0, generate_filename("PM"))
            
            # Load the colormap value
            colormap = settings.get("colormap", "viridis")  # Default to 'viridis'
            colormap_combo.set(colormap)

            # Load the invert colormap state
            invert_colormap_var.set(settings.get("invert_colormap", False))

            # Attempt to load the CSV if debug mode is enabled and the path is valid
            if debug_toggle_var.get() and csv_path and os.path.isfile(csv_path) and csv_path.lower().endswith('.csv'):
                load_csv()

        log_textbox.insert(tk.END, "Settings loaded.\n")
    else:
        log_textbox.insert(tk.END, "No settings file found. Using default values.\n")

# Dummy functions for connect buttons and other GUI actions
def connect_pm():
    global pm160_device

    # Get the VISA resource from the PM entry text box
    visa_resource = pm_entry.get().strip()
    rm = pyvisa.ResourceManager()
    available_resources = rm.list_resources()

    # Try using the existing VISA resource if provided
    if visa_resource:
        if try_connect_pm160(visa_resource):
            log_textbox.insert(tk.END, f"Connected to PM160 with resource: {visa_resource}\n")
            return
        else:
            log_textbox.insert(tk.END, f"Failed to connect using provided resource: {visa_resource}\n")

    # Try connecting to each available VISA resource if no valid resource is found
    for resource in available_resources:
        if try_connect_pm160(resource):
            pm_entry.delete(0, tk.END)
            pm_entry.insert(0, resource)
            log_textbox.insert(tk.END, f"Connected to PM160 with resource: {resource}\n")
            save_settings()  # Save the successful resource
            return

    log_textbox.insert(tk.END, "Failed to find a valid PM160 connection.\n")

def try_connect_pm160(resource):
    """
    Attempt to connect to a Thorlabs PM160 power meter using the given VISA resource.
    Returns True if successful, False otherwise.
    """
    global pm160_device
    try:
        # If there is an existing connection, close it first
        if pm160_device:
            pm160_device.close()

        # Try to establish a new connection
        pm160_device = Thorlabs.PM160(resource)
        return True
    except Exception as e:
        log_textbox.insert(tk.END, f"Error connecting to {resource}: {e}\n")
        pm160_device = None
        return False

def close_pm160_connection():
    """
    Closes the PM160 connection when the program is closed.
    """
    global pm160_device
    if pm160_device:
        try:
            pm160_device.close()
            log_textbox.insert(tk.END, "Closed PM160 connection.\n")
        except Exception as e:
            log_textbox.insert(tk.END, f"Error closing PM160 connection: {e}\n")
        pm160_device = None

def connect_kpa():
    global kpa_device

    # Get the serial number from the KPA entry text box
    serial_number = kpa_entry.get().strip()
    kinesis_devices = Thorlabs.list_kinesis_devices()

    # Filter for 'Position Aligner' devices
    position_aligners = [dev for dev in kinesis_devices if dev[1] == "Position Aligner"]

    # Try using the existing serial number if provided
    if serial_number:
        if try_connect_kpa(serial_number):
            log_textbox.insert(tk.END, f"Connected to Kinesis Quad Detector with serial number: {serial_number}\n")
            return
        else:
            log_textbox.insert(tk.END, f"Failed to connect using provided serial number: {serial_number}\n")

    # Try connecting to each available Position Aligner if no valid connection is found
    for dev_serial, dev_type in position_aligners:
        if try_connect_kpa(dev_serial):
            kpa_entry.delete(0, tk.END)
            kpa_entry.insert(0, dev_serial)
            log_textbox.insert(tk.END, f"Connected to Kinesis Quad Detector with serial number: {dev_serial}\n")
            save_settings()  # Save the successful serial number
            return

    log_textbox.insert(tk.END, "Failed to find a valid Kinesis Quad Detector connection.\n")

def try_connect_kpa(serial_number):
    """
    Attempt to connect to a Thorlabs Kinesis Quad Detector using the given serial number.
    Returns True if successful, False otherwise.
    """
    global kpa_device
    try:
        # If there is an existing connection, close it first
        if kpa_device:
            kpa_device.close()

        # Try to establish a new connection
        kpa_device = Thorlabs.kinesis.KinesisQuadDetector(serial_number)
        return True
    except Exception as e:
        log_textbox.insert(tk.END, f"Error connecting to KPA with serial {serial_number}: {e}\n")
        kpa_device = None
        return False

def close_kpa_connection():
    """
    Closes the Kinesis Quad Detector connection when the program is closed.
    """
    global kpa_device
    if kpa_device:
        try:
            kpa_device.close()
            log_textbox.insert(tk.END, "Closed Kinesis Quad Detector connection.\n")
        except Exception as e:
            log_textbox.insert(tk.END, f"Error closing Kinesis Quad Detector connection: {e}\n")
        kpa_device = None

def set_pid_parameters():
    """
    Updates the file names in the record KPA and PM entries to match the current PID values.
    Sets the PID values on the KPA device if it is connected.
    """
    global kpa_device
    if kpa_device is None:
        log_textbox.insert(tk.END, "KPA is not connected. Cannot update file names or set PID parameters.\n")
        return

    # Retrieve the PID values from the text boxes
    try:
        proportional = float(proportional_entry.get().strip())
        integral = float(integral_entry.get().strip())
        differential = float(gain_entry.get().strip())  # Assuming gain is used for differential
    except ValueError:
        log_textbox.insert(tk.END, "Invalid PID values. Please ensure the values are numbers.\n")
        return

    # Set the PID parameters on the KPA device
    try:
        kpa_device.set_pid_parameters(proportional, integral, differential)
        log_textbox.insert(
            tk.END,
            f"Set KPA PID parameters: Proportional={proportional}, Integral={integral}, Differential={differential}\n"
        )
    except Exception as e:
        log_textbox.insert(tk.END, f"Failed to set PID parameters on KPA: {e}\n")
        return

    # Regenerate the filenames with the new PID values
    record_kpa_entry.delete(0, tk.END)
    record_kpa_entry.insert(0, generate_filename("KPA"))

    record_pm_entry.delete(0, tk.END)
    record_pm_entry.insert(0, generate_filename("PM"))
    
    log_textbox.insert(tk.END, "File names updated to reflect new PID values.\n")

def check_pid_parameters():
    """
    Retrieves the PID parameters from the KPA device and logs them.
    Only works if the KPA is connected.
    """
    global kpa_device
    if kpa_device is None:
        log_textbox.insert(tk.END, "KPA is not connected. Cannot retrieve PID values.\n")
        return

    try:
        # Retrieve the PID parameters from the KPA device as a tuple
        pid_parameters = kpa_device.get_pid_parameters()
        proportional, integral, differential = pid_parameters[0], pid_parameters[1], pid_parameters[2]

        # Log the PID values
        log_textbox.insert(
            tk.END, 
            f"Retrieved PID values from KPA: Proportional={proportional}, Integral={integral}, Differential={differential}\n"
        )
    except Exception as e:
        log_textbox.insert(tk.END, f"Failed to retrieve PID parameters from KPA: {e}\n")

def on_debug_toggle():
    global x_data, y_data, x_voltage, y_voltage, alpha_values, time_data, power_data
    # Check if the debug mode has been turned off
    if not debug_toggle_var.get():
        # Clear the data arrays
        x_data = []
        y_data = []
        x_voltage = []
        y_voltage = []
        alpha_values = []
        time_data = []
        power_data = []
        current_index = 0

        # Clear the plot to reflect that no data is loaded
        ax1.clear()
        ax1.set_xlabel("X Voltage")
        ax1.set_ylabel("Y Voltage")
        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-5, 5)
        ax1.set_aspect('equal', adjustable='box')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.set_xticks(np.arange(-5, 6, 1))  # Set ticks from -10 to 10 with step 1 on x-axis
        ax1.set_yticks(np.arange(-5, 6, 1))  # Set ticks from -10 to 10 with step 1 on y-axis
        # Clear the data in Power vs. Time plot (ax2) by setting line2 data to empty lists
        line2.set_data([], [])  # Reset the line plot to an empty state
        ax2.relim()             # Reset axis limits
        ax2.autoscale()         # Autoscale with the empty data
        canvas.draw()

        log_textbox.insert(tk.END, "Debug mode is off. Data has been cleared.\n")

def set_open_loop():
    """
    Sets the KPA device to open-loop mode.
    """
    global kpa_device
    if kpa_device is None:
        log_textbox.insert(tk.END, "KPA is not connected. Cannot set to open-loop mode.\n")
        return

    try:
        kpa_device.set_operation_mode("open_loop")
        log_textbox.insert(tk.END, "KPA set to open-loop mode.\n")
    except Exception as e:
        log_textbox.insert(tk.END, f"Failed to set KPA to open-loop mode: {e}\n")

def set_closed_loop():
    """
    Sets the KPA device to closed-loop mode.
    """
    global kpa_device
    if kpa_device is None:
        log_textbox.insert(tk.END, "KPA is not connected. Cannot set to closed-loop mode.\n")
        return

    try:
        kpa_device.set_operation_mode("closed_loop")
        log_textbox.insert(tk.END, "KPA set to closed-loop mode.\n")
    except Exception as e:
        log_textbox.insert(tk.END, f"Failed to set KPA to closed-loop mode: {e}\n")

def set_polling_rate():
    """
    Set the polling rate based on the user's input.
    Ensures that the polling rate is a positive number.
    """
    polling_rate = polling_entry.get().strip()
    try:
        rate = float(polling_rate)
        if rate > 0:
            polling_rate_var.set(rate)
            log_textbox.insert(tk.END, f"Polling rate set to {rate} Hz.\n")
            save_settings()  # Save the polling rate to settings
        else:
            log_textbox.insert(tk.END, "Polling rate must be greater than 0.\n")
    except ValueError:
        log_textbox.insert(tk.END, "Invalid polling rate. Please enter a positive number.\n")

def record_kpa():
    """
    Save the current x_data, y_data, and time_data to a CSV file with the name in the Record KPA text box.
    If a file with the same name exists, a unique name with an incremental suffix is generated.
    """
    global kpa_device, x_data, y_data, x_voltage, y_voltage, time_data

    if kpa_device is None:
        log_message("KPA is not connected. Please connect the KPA device first.\n")
        return

    # Get the file name from the Record KPA text box
    file_name = record_kpa_entry.get().strip()
    if not file_name:
        log_message("No file name provided for KPA recording.\n")
        return

    # Ensure file name ends with .csv
    if not file_name.lower().endswith('.csv'):
        file_name += ".csv"

    # Get a unique filename to prevent overwriting existing files
    unique_file_name = get_unique_filename(file_name)

    try:
        # Create a DataFrame from x_data, y_data, and time_data
        df = pd.DataFrame({
            'time': time_data,
            'x_diff': x_data,
            'y_diff': y_data,
            'x_vol': x_voltage,
            'y_vol': y_voltage
        })

        # Save the DataFrame to a CSV file
        df.to_csv(unique_file_name, index=False)
        log_message(f"KPA data saved to {unique_file_name}\n")
    except Exception as e:
        log_message(f"Failed to save KPA data: {e}\n")

def record_pm():
    """
    Save the current time_data and power_data to a CSV file with the name in the Record PM text box.
    If a file with the same name exists, a unique name with an incremental suffix is generated.
    """
    global pm160_device, time_data, power_data

    if pm160_device is None:
        log_message("PM is not connected. Please connect the PM device first.\n")
        return

    # Get the file name from the Record PM text box
    file_name = record_pm_entry.get().strip()  # Assuming you have an entry field named record_pm_entry
    if not file_name:
        log_message("No file name provided for PM recording.\n")
        return

    # Ensure file name ends with .csv
    if not file_name.lower().endswith('.csv'):
        file_name += ".csv"

    # Get a unique filename to prevent overwriting existing files
    unique_file_name = get_unique_filename(file_name)

    try:
        # Create a DataFrame from time_data and power_data
        df = pd.DataFrame({
            'time': time_data,
            'power': power_data
        })

        # Save the DataFrame to a CSV file
        df.to_csv(unique_file_name, index=False)
        log_message(f"PM data saved to {unique_file_name}\n")
    except Exception as e:
        log_message(f"Failed to save PM data: {e}\n")


def record_both():
    """
    Placeholder function for recording both KPA and PM data.
    Replace with the logic for starting the recording of both.
    """
    log_textbox.insert(tk.END, "Started recording both KPA and PM data.\n")

def get_kpa_reading():
    """
    Gets a reading from the KPA device and prints it to the log.
    """
    global kpa_device
    if kpa_device is None:
        log_textbox.insert(tk.END, "KPA is not connected. Cannot get reading.\n")
        return

    try:
        reading = kpa_device.get_readings()
        log_textbox.insert(tk.END, f"KPA Reading: {reading}\n")
    except Exception as e:
        log_textbox.insert(tk.END, f"Failed to get KPA reading: {e}\n")

def get_pm_reading():
    """
    Gets a reading from the PM device and prints it to the log.
    """
    global pm160_device
    if pm160_device is None:
        log_textbox.insert(tk.END, "PM is not connected. Cannot get reading.\n")
        return

    try:
        reading = pm160_device.get_reading()
        log_textbox.insert(tk.END, f"PM Reading: {reading}\n")
    except Exception as e:
        log_textbox.insert(tk.END, f"Failed to get PM reading: {e}\n")

# Create the main window
root = tk.Tk()
root.title("Control GUI")

# Load previous settings if they exist

polling_rate_var = tk.DoubleVar(value=1.0)  # Default to 1.0 Hz

# Function to save settings when closing the window
def on_closing():
    save_settings()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", lambda: [save_settings(), close_pm160_connection(), close_kpa_connection(), root.destroy()])

# Get the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the initial window size based on a fraction of the screen size
window_width = int(screen_width * 0.8)
window_height = int(screen_height * 0.8)
root.geometry(f"{window_width}x{window_height}")

# Set a scaling factor based on the screen size for fonts and padding
scaling_factor = min(screen_width / 1920, screen_height / 1080)
font_size = int(12 * scaling_factor)
label_font_size = int(14 * scaling_factor)
entry_font_size = int(14 * scaling_factor)
entry_ipady = int(10 * scaling_factor)
entry_width = int(30 * scaling_factor)
pad_x = int(10 * scaling_factor)
pad_y = int(5 * scaling_factor)

# Create a style for consistent font sizing
style = ttk.Style()
style.configure("TButton", font=("Helvetica", font_size))
style.configure("TLabel", font=("Helvetica", font_size))
style.configure("TEntry", font=("Helvetica", entry_font_size))
style.configure("TLabelframe.Label", font=("Helvetica", label_font_size))

# Define the list of colormaps suitable for fading out points
fading_colormaps = ['viridis', 'plasma', 'cividis', 'inferno', 'magma', 'cubehelix', 'Greys']

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np

# Create the top left frame for the matplotlib canvas and controls
top_left_frame = ttk.Frame(root)
top_left_frame.grid(row=0, column=0, padx=pad_x, pady=pad_y, sticky="nsew")

# Create the figure with two subplots, with different width ratios for each axis
fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(8, 5))

# Configure the first axis (main plot)
ax1.set_xlabel("X Voltage")
ax1.set_ylabel("Y Voltage")
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax1.set_aspect('equal', adjustable='box')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.set_xticks(np.arange(-5, 6, 1))
ax1.set_yticks(np.arange(-5, 6, 1))

# Configure the second axis (tall and thin plot for Power vs Time)
ax2.set_xlabel("Time")
ax2.set_ylabel("Power")
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.grid(True, linestyle='--', linewidth=0.5)
line2, = ax2.plot([], [], 'r-', label="Power")  # Red line plot for power vs time

# Create the canvas and add it to the Tkinter frame
canvas = FigureCanvasTkAgg(fig, master=top_left_frame)
canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
canvas.draw()

# Create a separate frame for the toolbar
toolbar_frame = ttk.Frame(top_left_frame)
toolbar_frame.grid(row=1, column=0, sticky="ew")

# Add the Matplotlib toolbar to the toolbar frame using pack
toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
toolbar.update()
toolbar.pack(side="top", fill="x")

# Create a frame specifically for plot control buttons and inputs
plot_controls_frame = ttk.Frame(top_left_frame)
plot_controls_frame.grid(row=2, column=0, sticky="ew", padx=pad_x, pady=pad_y)

# Configure the weights to ensure resizing behavior
top_left_frame.grid_rowconfigure(0, weight=1)  # The plot area should expand
top_left_frame.grid_rowconfigure(1, weight=0)  # Toolbar should not expand
top_left_frame.grid_rowconfigure(2, weight=0)  # Controls should not expand
top_left_frame.grid_columnconfigure(0, weight=1)

# Arrange the widgets in the plot controls frame
start_button = ttk.Button(plot_controls_frame, text="Start Plot", command=start_plot)
start_button.grid(row=0, column=0, padx=pad_x, pady=pad_y, sticky="ew")

stop_button = ttk.Button(plot_controls_frame, text="Stop Plot", command=stop_plot)
stop_button.grid(row=0, column=1, padx=pad_x, pady=pad_y, sticky="ew")

interval_label = ttk.Label(plot_controls_frame, text="Time Interval (ms):")
interval_label.grid(row=0, column=2, padx=pad_x, pady=pad_y, sticky="w")

interval_entry = ttk.Entry(plot_controls_frame, width=10, font=("Helvetica", entry_font_size))
interval_entry.insert(0, "1000")
interval_entry.grid(row=0, column=3, padx=pad_x, pady=pad_y, sticky="ew")

# Additional controls as per the previous setup
points_label = ttk.Label(plot_controls_frame, text="Max Points:")
points_label.grid(row=1, column=0, padx=pad_x, pady=pad_y, sticky="w")

points_entry = ttk.Entry(plot_controls_frame, width=10, font=("Helvetica", entry_font_size))
points_entry.insert(0, "1000")
points_entry.grid(row=1, column=1, padx=pad_x, pady=pad_y, sticky="ew")

colormap_label = ttk.Label(plot_controls_frame, text="Colormap:")
colormap_label.grid(row=1, column=2, padx=pad_x, pady=pad_y, sticky="w")

colormap_combo = ttk.Combobox(plot_controls_frame, values=fading_colormaps, width=15)
colormap_combo.set('viridis')
colormap_combo.grid(row=1, column=3, padx=pad_x, pady=pad_y, sticky="ew")


# Add a BooleanVar to track the state of the checkbox
invert_colormap_var = tk.BooleanVar(value=False)

# Add the checkbox for inverting the colormap
invert_colormap_checkbox = ttk.Checkbutton(
    plot_controls_frame, 
    text="Invert Colormap", 
    variable=invert_colormap_var
)
invert_colormap_checkbox.grid(row=1, column=4, padx=pad_x, pady=pad_y, sticky="w")

# Make sure the columns expand evenly if the window is resized
for i in range(4):
    plot_controls_frame.grid_columnconfigure(i, weight=1)


# Create the top right frame for the connect section and debug section
top_right_frame = ttk.Frame(root)
top_right_frame.grid(row=0, column=1, padx=pad_x, pady=pad_y, sticky="nsew")

# Connect section
connect_frame = ttk.LabelFrame(top_right_frame, text="Connect")
connect_frame.pack(fill="x", pady=pad_y)

# PM Connect
pm_entry = ttk.Entry(connect_frame, width=entry_width, font=("Helvetica", entry_font_size))
pm_entry.grid(row=0, column=0, padx=pad_x, pady=pad_y, sticky="ew", ipady=entry_ipady)
pm_button = ttk.Button(connect_frame, text="Connect PM", command=connect_pm)
pm_button.grid(row=0, column=1, padx=pad_x, pady=pad_y)

# KPA Connect
kpa_entry = ttk.Entry(connect_frame, width=entry_width, font=("Helvetica", entry_font_size))
kpa_entry.grid(row=1, column=0, padx=pad_x, pady=pad_y, sticky="ew", ipady=entry_ipady)
kpa_button = ttk.Button(connect_frame, text="Connect KPA", command=connect_kpa)
kpa_button.grid(row=1, column=1, padx=pad_x, pady=pad_y)

# Record Data section
record_frame = ttk.LabelFrame(top_right_frame, text="Record Data")
record_frame.pack(fill="x", pady=pad_y)

# Polling rate entry and button
polling_entry = ttk.Entry(record_frame, width=entry_width, font=("Helvetica", entry_font_size))
polling_entry.grid(row=0, column=0, padx=pad_x, pady=pad_y, sticky="ew", ipady=entry_ipady)
polling_button = ttk.Button(record_frame, text="Set Polling Rate", command=set_polling_rate)
polling_button.grid(row=0, column=1, padx=pad_x, pady=pad_y)

# Record KPA entry and button
record_kpa_entry = ttk.Entry(record_frame, width=entry_width, font=("Helvetica", entry_font_size))
record_kpa_entry.grid(row=1, column=0, padx=pad_x, pady=pad_y, sticky="ew", ipady=entry_ipady)
record_kpa_button = ttk.Button(record_frame, text="Record KPA", command=record_kpa)
record_kpa_button.grid(row=1, column=1, padx=pad_x, pady=pad_y)

# Record PM entry and button
record_pm_entry = ttk.Entry(record_frame, width=entry_width, font=("Helvetica", entry_font_size))
record_pm_entry.grid(row=2, column=0, padx=pad_x, pady=pad_y, sticky="ew", ipady=entry_ipady)
record_pm_button = ttk.Button(record_frame, text="Record PM", command=record_pm)
record_pm_button.grid(row=2, column=1, padx=pad_x, pady=pad_y)

# Record both button
record_both_button = ttk.Button(record_frame, text="Record Both", command=record_both)
record_both_button.grid(row=3, column=0, columnspan=2, padx=pad_x, pady=pad_y, sticky="ew")

# Get Reading KPA button
get_kpa_reading_button = ttk.Button(record_frame, text="Get Reading KPA", command=get_kpa_reading)
get_kpa_reading_button.grid(row=1, column=2, padx=pad_x, pady=pad_y)

# Get Reading PM button
get_pm_reading_button = ttk.Button(record_frame, text="Get Reading PM", command=get_pm_reading)
get_pm_reading_button.grid(row=2, column=2, padx=pad_x, pady=pad_y)

# Start Plotting KPA button
plot_kpa_button = ttk.Button(record_frame, text="Plot KPA", command=lambda: [start_plotting_kpa(), start_plot()])
plot_kpa_button.grid(row=1, column=3, padx=pad_x, pady=pad_y)

# Stop Plotting KPA button
stop_plotting_button = ttk.Button(record_frame, text="Stop Plot KPA", command=stop_plotting_kpa)
stop_plotting_button.grid(row=1, column=4, padx=pad_x, pady=pad_y)

# Debug section
debug_frame = ttk.LabelFrame(top_right_frame, text="Debug")
debug_frame.pack(fill="x", pady=pad_y)

debug_entry = ttk.Entry(debug_frame, width=entry_width, font=("Helvetica", entry_font_size))
debug_entry.grid(row=0, column=0, padx=pad_x, pady=pad_y, sticky="ew", ipady=entry_ipady)
browse_button = ttk.Button(debug_frame, text="Browse", command=browse_csv)
browse_button.grid(row=0, column=1, padx=pad_x, pady=pad_y)

debug_toggle_var = tk.BooleanVar(value=True)

# Add the checkbox for debug mode with the command to call on_debug_toggle when changed
debug_toggle_checkbox = ttk.Checkbutton(
    debug_frame,
    text="Debug Toggle",
    variable=debug_toggle_var,
    command=on_debug_toggle  # Bind the function here
)
debug_toggle_checkbox.grid(row=2, column=0, columnspan=2, padx=pad_x, pady=pad_y, sticky="w")
# LOAD CSV button under the text box
load_csv_button = ttk.Button(debug_frame, text="LOAD CSV", command=load_csv)
load_csv_button.grid(row=1, column=0, columnspan=2, pady=pad_y)

# Add "Reset" button in the debug frame
reset_button = ttk.Button(debug_frame, text="Reset", command=reset_plot)
reset_button.grid(row=2, column=0, columnspan=2, pady=pad_y)

# Create the bottom left frame for the proportional, integral, gain and loop controls
bottom_left_frame = ttk.Frame(root)
bottom_left_frame.grid(row=1, column=0, padx=pad_x, pady=pad_y, sticky="nsew")

# Proportional, Integral, Gain controls
pid_frame = ttk.LabelFrame(bottom_left_frame, text="PID Controls")
pid_frame.grid(row=0, column=0, padx=pad_x, pady=pad_y, sticky="nsew")

proportional_label = ttk.Label(pid_frame, text="Proportional:")
proportional_label.grid(row=0, column=0, sticky="w", padx=pad_x)
proportional_entry = ttk.Entry(pid_frame, font=("Helvetica", entry_font_size))
proportional_entry.grid(row=0, column=1, padx=pad_x, sticky="ew", ipady=entry_ipady)

integral_label = ttk.Label(pid_frame, text="Integral:")
integral_label.grid(row=1, column=0, sticky="w", padx=pad_x)
integral_entry = ttk.Entry(pid_frame, font=("Helvetica", entry_font_size))
integral_entry.grid(row=1, column=1, padx=pad_x, sticky="ew", ipady=entry_ipady)

gain_label = ttk.Label(pid_frame, text="Gain:")
gain_label.grid(row=2, column=0, sticky="w", padx=pad_x)
gain_entry = ttk.Entry(pid_frame, font=("Helvetica", entry_font_size))
gain_entry.grid(row=2, column=1, padx=pad_x, sticky="ew", ipady=entry_ipady)

check_pid_button = ttk.Button(pid_frame, text="Check", command=check_pid_parameters)
check_pid_button.grid(row=3, column=0, padx=pad_x, pady=pad_y)

set_pid_button = ttk.Button(pid_frame, text="Set", command=set_pid_parameters)
set_pid_button.grid(row=3, column=1, padx=pad_x, pady=pad_y)

# Open/Closed loop controls
loop_frame = ttk.LabelFrame(bottom_left_frame, text="Loop Control")
loop_frame.grid(row=0, column=1, padx=pad_x, pady=pad_y, sticky="nsew")

open_loop_button = ttk.Button(loop_frame, text="Open Loop", command=set_open_loop)
open_loop_button.grid(row=0, column=0, padx=pad_x, pady=pad_y)

closed_loop_button = ttk.Button(loop_frame, text="Closed Loop", command=set_closed_loop)
closed_loop_button.grid(row=0, column=1, padx=pad_x, pady=pad_y)

# Create the bottom right frame for the log section
bottom_right_frame = ttk.Frame(root)
bottom_right_frame.grid(row=1, column=1, padx=pad_x, pady=pad_y, sticky="nsew")

log_textbox = scrolledtext.ScrolledText(bottom_right_frame, height=10, font=("Helvetica", font_size))
log_textbox.pack(fill="both", expand=True)

load_settings()

# Configure the grid to allow resizing
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
top_left_frame.grid_rowconfigure(0, weight=1)
top_left_frame.grid_columnconfigure(0, weight=1)
top_right_frame.grid_rowconfigure(0, weight=1)
top_right_frame.grid_columnconfigure(0, weight=1)
bottom_left_frame.grid_rowconfigure(0, weight=1)
bottom_left_frame.grid_columnconfigure(0, weight=1)
bottom_left_frame.grid_columnconfigure(1, weight=1)
bottom_right_frame.grid_rowconfigure(0, weight=1)
bottom_right_frame.grid_columnconfigure(0, weight=1)

# Variable to keep track of plotting state
plot_active = False

# Start the Tkinter main loop
root.mainloop()
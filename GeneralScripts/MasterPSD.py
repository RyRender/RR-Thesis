import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
import os
from os import listdir
from os.path import isfile, join
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from datetime import datetime

def PSDanalysisOLD(list2, list3, t, nperseg=int(1e6)):
    # Convert to numpy arrays
    list2 = np.asarray(list2)
    list3 = np.asarray(list3)

    # Assume 1D, get first column
    list2_vals = list2[:, 0] if list2.ndim > 1 else list2
    list3_vals = list3[:, 0] if list3.ndim > 1 else list3

    S = np.max(list3_vals) + np.min(list3_vals)
    D = np.max(list3_vals) - np.min(list3_vals)

    x = ((2 * list2_vals) - S) / D

    # Clip and count
    x_clipped = np.clip(x, -1, 1)
    count = np.sum((x > 1) | (x < -1))

    fs = len(x_clipped) / t
    f, Pxx_den = signal.welch(x_clipped, fs, window='hann', nperseg=nperseg)

    print(count)
    return f, Pxx_den

def PSDanalysis(list2, list3, t, nperseg=int(1e6)):
    # Convert to numpy arrays
    list2 = np.asarray(list2)
    list3 = np.asarray(list3)

    # Assume 1D, get first column
    list2_vals = list2[:, 0] if list2.ndim > 1 else list2
    list3_vals = list3[:, 0] if list3.ndim > 1 else list3

    S = np.max(list3_vals) + np.min(list3_vals)
    D = np.max(list3_vals) - np.min(list3_vals)

    x = ((2 * list2_vals) - S) / D
    # Clip and count
    x_clipped = np.clip(x, -1, 1)
    count = np.sum((x > 1) | (x < -1))
    phi=np.arccos(x_clipped)
    fs = len(x_clipped) / t
    f, Pxx_den = signal.welch(phi, fs, window='hann', nperseg=nperseg)

    print(count)
    return f, Pxx_den

def AnalyseFull(maindir, fandir, time):
    # Load main data
    main = np.loadtxt(maindir, skiprows=5, delimiter=',', usecols=1)

    # Load fan data
    if fandir == 'self' or fandir == maindir:
        fan = main
    else:
        fan = np.loadtxt(fandir, skiprows=5, delimiter=',', usecols=1)

    # Extract date from path (YYYY-MM-DD)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', maindir)
    if date_match:
        date_str_raw = date_match.group(1)
        date_str = datetime.strptime(date_str_raw, "%Y-%m-%d").strftime("%d_%m_%Y")
    else:
        raise ValueError(f"Could not extract date from path: {maindir}")

    # Extract folder name (e.g., 'Keithley')
    parent_folder = os.path.basename(os.path.dirname(maindir))

    # Extract config (e.g., '250ks') from filename
    filename = os.path.basename(maindir)
    config_match = re.search(r'-([a-zA-Z0-9]+)--', filename)
    if config_match:
        config = config_match.group(1)
    else:
        config = 'unknown'
        print(f"Warning: Could not extract config from filename: {filename}")

    # Extract index (e.g., '--00000.csv')
    index_match = re.search(r'--(\d+)\.csv$', filename)
    if index_match:
        index = int(index_match.group(1))
    else:
        index = 0
        print(f"Warning: Could not extract index from filename: {filename}")

    # Construct final shortname
    shortname = f"{date_str}_{parent_folder}_{config}_{index}"

    # Run PSD analysis
    main_f, main_Pxx = PSDanalysis(main, fan, time)

    # Save outputs
    np.save(f"{shortname}_f.npy", main_f)
    np.save(f"{shortname}_Pxx.npy", main_Pxx)

    return main_f, main_Pxx

def process_file(args):
    path, fandir, param_map = args
    filename = os.path.basename(path)

    # Try underscore-style pattern first
    underscore_pattern = re.compile(r'_([a-zA-Z0-9]+)--')
    match = underscore_pattern.search(filename)

    # If no match, try hyphen-style pattern (last hyphen-separated word before --)
    if not match:
        hyphen_pattern = re.compile(r'-([a-zA-Z0-9]+)--')
        match = hyphen_pattern.search(filename)

    if match:
        key = match.group(1)  # e.g., "5ms"
        if key in param_map:
            try:
                AnalyseFull(path, fandir, param_map[key]["time"])
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Unknown key '{key}' in file {filename}, skipping.")
    else:
        print(f"Could not extract parameters from {filename}")

def AnalyseDirs(file_paths, fandir, param_map):
    args = [(path, fandir, param_map) for path in file_paths]
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_file, args), total=len(file_paths)))

def fildirs(mypath):
    onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if 
    os.path.isfile(os.path.join(mypath, f))]
    return(onlyfiles)

def SplicePSD4(PSDs, splices):
    xs=[]
    ys=[]
    for i in range(len(PSDs[0][0])):
        if PSDs[0][0][i]<splices[0]:
            xs.append(PSDs[0][0][i])
            ys.append(PSDs[0][1][i])
    for i in range(len(PSDs[1][0])):
        if PSDs[1][0][i]>splices[0]:
            if PSDs[1][0][i]<splices[1]:
                xs.append(PSDs[1][0][i])
                ys.append(PSDs[1][1][i])
    for i in range(len(PSDs[2][0])):
        if PSDs[2][0][i]>splices[1]:
            if PSDs[2][0][i]<splices[2]:
                xs.append(PSDs[2][0][i])
                ys.append(PSDs[2][1][i])
    for i in range(len(PSDs[3][0])):
        if PSDs[3][0][i]>splices[2]:
            xs.append(PSDs[3][0][i])
            ys.append(PSDs[3][1][i])
    return(xs,ys)

def SplicePSD5(PSDs, splices):
    xs=[]
    ys=[]
    for i in range(len(PSDs[0][0])):
        if PSDs[0][0][i]<splices[0]:
            xs.append(PSDs[0][0][i])
            ys.append(PSDs[0][1][i])
    for i in range(len(PSDs[1][0])):
        if PSDs[1][0][i]>splices[0]:
            if PSDs[1][0][i]<splices[1]:
                xs.append(PSDs[1][0][i])
                ys.append(PSDs[1][1][i])
    for i in range(len(PSDs[2][0])):
        if PSDs[2][0][i]>splices[1]:
            if PSDs[2][0][i]<splices[2]:
                xs.append(PSDs[2][0][i])
                ys.append(PSDs[2][1][i])
    for i in range(len(PSDs[3][0])):
        if PSDs[3][0][i]>splices[2]:
            if PSDs[3][0][i]<splices[3]:
                xs.append(PSDs[3][0][i])
                ys.append(PSDs[3][1][i])
    for i in range(len(PSDs[4][0])):
        if PSDs[4][0][i]>splices[3]:
            xs.append(PSDs[4][0][i])
            ys.append(PSDs[4][1][i])
    return(xs,ys)
def sort_key(item):
    label = item[0]  # e.g., '250ks_0'
    match = re.match(r'([a-zA-Z0-9]+)_([0-9]+)', label)
    if match:
        config_str = match.group(1)  # e.g. '250ks'
        index = int(match.group(2))

        # Use custom-defined order
        desired_order = ['250ks', '5ms', '50ms', '500ms', '5gs']
        try:
            priority = desired_order.index(config_str)
        except ValueError:
            priority = float('inf')  # unknown configs go last

        return (priority, index)

    return (float('inf'), 9999)  # fallback
def load_analysis_data(date_prefix, category):
    """
    Load and pair _f.npy and _Pxx.npy files based on naming convention:
    '{date}_{category}_{config}_{index}_f.npy'
    """
    folder = os.getcwd()
    pattern_f = os.path.join(folder, f"{date_prefix}_{category}_*_f.npy")
    files_f = glob.glob(pattern_f)

    data_pairs = []

    # Pattern to extract config + index
    regex = re.compile(rf"{date_prefix}_{category}_(.*?)_(\d+)_f\.npy")

    for f_path in files_f:
        base_f = os.path.basename(f_path)
        match = regex.match(base_f)

        if match:
            config = match.group(1)
            index = match.group(2)

            # Construct matching _Pxx.npy filename
            pxx_filename = f"{date_prefix}_{category}_{config}_{index}_Pxx.npy"
            pxx_path = os.path.join(folder, pxx_filename)

            if os.path.exists(pxx_path):
                f_data = np.load(f_path)
                Pxx_data = np.load(pxx_path)
                label = f"{config}_{index}"
                data_pairs.append((label, f_data, Pxx_data))
            else:
                print(f"Warning: Missing Pxx file for {base_f}")
        else:
            print(f"Warning: Could not parse config/index from {base_f}")

    data_pairs.sort(key=sort_key)

    return data_pairs
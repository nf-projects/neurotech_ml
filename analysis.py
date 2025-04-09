import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_multitaper  # Used for multitaper PSD estimation
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import os
import sys
from datetime import datetime
from io import StringIO
import contextlib

# === Configuration ===
SAMPLING_RATE = 250           # Hz
EEG_CHANNELS = ['P4', 'Pz', 'P3', 'C3', 'Cz', 'C4', 'F3', 'F4']

# Updated Event mapping: keys as event names (strings) and values as event IDs (ints)
EVENT_ID_MAP = {
    'DelayedBeep': 33285,    # stress-inducing delayed tone
    'NonDelayedBeep': 33286  # control, normal tone
}

# Filtering parameters
LOW_CUT = 1.0    # Hz
HIGH_CUT = 30.0  # Hz

# Epoching parameters: capture anticipation (pre-stimulus) and reaction (post-stimulus)
EPOCH_TMIN = -0.5  # seconds (before event)
EPOCH_TMAX = 1.5   # seconds (after event)

# Directory paths
CSV_DIR = 'csvs'
RESULTS_DIR = 'results'

# Ensure directories exist
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


@contextlib.contextmanager
def capture_output():
    """Context manager to capture stdout."""
    stdout_buffer = StringIO()
    stdout_backup, sys.stdout = sys.stdout, stdout_buffer
    try:
        yield stdout_buffer
    finally:
        sys.stdout = stdout_backup


def analyze_file(file_path):
    """Process a single CSV file and return the figures and output."""
    file_name = os.path.basename(file_path)
    print(f"\n=== Processing {file_name} ===")
    
    # === 1. Load Data ===
    print("\n=== Loading Data ===")
    df = pd.read_csv(file_path)
    # Rename the first column to "Time" if necessary
    df.rename(columns={df.columns[0]: 'Time'}, inplace=True)
    print(f"Data shape: {df.shape}")

    # === 2. Extract and Clean Event Markers ===
    print("\n=== Extracting Event Markers ===")
    # Retain rows with non-null Event Id
    events_raw = df.dropna(subset=['Event Id']).copy()

    def clean_event_id(event_id):
        """Parse an event id into an integer, handling compound values."""
        try:
            event_str = str(event_id)
            if ':' in event_str:
                event_str = event_str.split(':')[0].strip()
            return int(float(event_str))
        except Exception:
            return np.nan

    events_raw['Cleaned_ID'] = events_raw['Event Id'].apply(clean_event_id)
    events_clean = events_raw.dropna(subset=['Cleaned_ID'])
    events_clean['Cleaned_ID'] = events_clean['Cleaned_ID'].astype(int)
    print(f"Found {len(events_clean)} event markers.")

    # Create an MNE-compatible event array [sample, 0, event_id]
    event_list = []
    for _, row in events_clean.iterrows():
        sample = int(round(row['Time'] * SAMPLING_RATE))
        sample = max(0, min(sample, len(df) - 1))  # Ensure within valid index range
        event_list.append([sample, 0, row['Cleaned_ID']])
    mne_events = np.array(event_list, dtype=int)
    print(f"MNE event array shape: {mne_events.shape}")

    # === 3. Create an MNE Raw Object ===
    print("\n=== Creating MNE Raw Object ===")
    # Check if all required EEG channels exist in the dataframe
    missing_channels = [ch for ch in EEG_CHANNELS if ch not in df.columns]
    if missing_channels:
        print(f"Warning: Missing EEG channels: {missing_channels}")
        print(f"Available columns: {list(df.columns)}")
        print("Cannot proceed with analysis. Skipping this file.")
        return [], f"Missing EEG channels: {missing_channels}"
        
    # Extract EEG channels; expected shape: (n_channels, n_samples)
    eeg_data = df[EEG_CHANNELS].values.T
    # Convert from microvolts to Volts (if appropriate)
    eeg_data_volts = eeg_data * 1e-6
    ch_types = ['eeg'] * len(EEG_CHANNELS)
    info = mne.create_info(ch_names=EEG_CHANNELS, sfreq=SAMPLING_RATE, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data_volts, info)
    # Set montage (10-20 electrode locations)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn')
    print(raw)

    # === 4. Filter and Clean the Data (ICA) ===
    print("\n=== Filtering Raw Data ===")
    raw_filtered = raw.copy().filter(l_freq=LOW_CUT, h_freq=HIGH_CUT, fir_design='firwin')
    print("Filtering complete.")

    print("\n=== Running ICA for Artifact Removal (Optional) ===")
    ica = mne.preprocessing.ICA(n_components=len(EEG_CHANNELS), random_state=97, max_iter=800)
    ica.fit(raw_filtered)
    # In practice, inspect ICA components before rejecting; here we apply all components as-is
    raw_clean = ica.apply(raw_filtered.copy())
    print("ICA complete. Using cleaned raw data for epoching.")

    # === 5. Epoching Around Conditions of Interest ===
    print("\n=== Creating Epochs for DelayedBeep and NonDelayedBeep ===")
    # Filter events to only those matching our conditions (using the event ID values from our map)
    selected_events = mne_events[np.isin(mne_events[:, 2], list(EVENT_ID_MAP.values()))]
    if selected_events.size == 0:
        print("No events for DelayedBeep/NonDelayedBeep found!")
        return [], "No events found"
    else:
        # Create a dynamic event_id_map containing only events that are actually present in the data
        available_event_ids = set(selected_events[:, 2])
        available_event_map = {name: id for name, id in EVENT_ID_MAP.items() if id in available_event_ids}
        
        if not available_event_map:
            print("No matching events found in available events IDs!")
            return [], "No matching events found"
            
        print(f"Available event types: {list(available_event_map.keys())}")
        
        # Create epochs with only the available event types
        epochs = mne.Epochs(raw_clean, events=selected_events, event_id=available_event_map,
                           tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=None, preload=True)
        print(epochs)
        for cond in available_event_map:  # now iterating over keys which are event names
            print(f"  {cond}: {len(epochs[cond])} epochs")

    # === 6. Compute Band Power (Delta and Beta) ===
    print("\n=== Calculating Band Power (Delta and Beta) ===")
    # Get epoch data: shape (n_epochs, n_channels, n_times)
    epochs_data = epochs.get_data()
    # Compute PSD using multitaper on the epochs array
    psds, freqs = psd_array_multitaper(epochs_data, sfreq=SAMPLING_RATE, fmin=1, fmax=30, verbose=False)
    # psds shape: (n_epochs, n_channels, n_freqs)

    # Define frequency band masks
    delta_mask = (freqs >= 1) & (freqs <= 4)
    beta_mask = (freqs >= 13) & (freqs <= 30)
    delta_power = psds[:, :, delta_mask].mean(axis=2)  # average delta power per epoch per channel
    beta_power  = psds[:, :, beta_mask].mean(axis=2)    # average beta power per epoch per channel

    # Compute a simple delta-beta ratio per epoch (averaged across channels)
    delta_beta_ratio = beta_power.mean(axis=1) / delta_power.mean(axis=1)
    print("Delta-Beta ratio (first 10 epochs):")
    print(delta_beta_ratio[:10])

    # === 7. Optional: Machine Learning Pipeline with CSP & LDA ===
    print("\n=== Optional: ML Classification (CSP + LDA) ===")
    # Extract epochs and labels for ML – restrict to our available conditions
    X = epochs.get_data()    # shape: (n_epochs, n_channels, n_times)
    y = epochs.events[:, 2]  # event IDs as labels
    valid_idx = np.isin(y, list(available_event_map.values()))
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Check if we have at least 2 different classes
    unique_classes = np.unique(y)
    if len(unique_classes) >= 2:
        csp = CSP(n_components=4, reg=None, log=True)
        clf_pipeline = Pipeline([('CSP', csp), ('LDA', LinearDiscriminantAnalysis())])
        scores = cross_val_score(clf_pipeline, X, y, cv=5)
        print("Classification accuracy (CSP + LDA): {:.2f}%".format(np.mean(scores) * 100))
    else:
        print("Insufficient classes for ML classification (need at least 2 different classes).")
    
    # === 8. Visualization ===
    print("\n=== Visualizing Results ===")
    figures = []
    
    # Create a single comprehensive figure with multiple subplots
    evoked_dict = {cond: epochs[cond].average() for cond in available_event_map}
    
    if evoked_dict:
        # Determine the number of plots needed (1 for compare, plus 1 for each condition)
        n_plots = 1 + len(evoked_dict) if len(evoked_dict) >= 2 else len(evoked_dict)
        
        # Create figure with appropriate number of subplots
        fig = plt.figure(figsize=(12, 4 * n_plots))
        
        # First plot: comparison (if we have 2+ conditions)
        plot_idx = 1
        if len(evoked_dict) >= 2:
            ax_compare = fig.add_subplot(n_plots, 1, plot_idx)
            mne.viz.plot_compare_evokeds(evoked_dict, picks=EEG_CHANNELS, combine='mean', axes=ax_compare, show=False)
            ax_compare.set_title(f"Compare Evokeds - {file_name}")
            plot_idx += 1
        
        # Individual condition plots
        for cond in evoked_dict:
            ax_cond = fig.add_subplot(n_plots, 1, plot_idx)
            evoked_dict[cond].plot(axes=ax_cond, show=False)
            ax_cond.set_title(f"ERP for {cond} - {file_name}")
            plot_idx += 1
        
        plt.tight_layout(h_pad=4.0)
        figures.append(("eeg_analysis", fig))
    else:
        print("No event types available for visualization")
    
    print("\n=== Analysis Complete ===")
    
    return figures


def process_all_files():
    """Process all CSV files in the CSV_DIR directory."""
    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {CSV_DIR} directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    for csv_file in csv_files:
        file_path = os.path.join(CSV_DIR, csv_file)
        file_name = os.path.splitext(csv_file)[0]
        result_dir = os.path.join(RESULTS_DIR, file_name)
        os.makedirs(result_dir, exist_ok=True)
        
        # Capture console output while processing the file
        with capture_output() as output:
            figures = analyze_file(file_path)
        
        # Save console output to text file
        log_file = os.path.join(result_dir, f"{file_name}_log.txt")
        with open(log_file, 'w') as f:
            f.write(output.getvalue())
        
        # Save figures to PNG files if there are any
        if figures and isinstance(figures, list) and figures:
            for fig_name, fig in figures:
                if fig is not None:
                    try:
                        fig_path = os.path.join(result_dir, f"{file_name}_{fig_name}.png")
                        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                    except Exception as e:
                        print(f"Error saving figure {fig_name}: {e}")
        
        print(f"Results saved to {result_dir}")


if __name__ == "__main__":
    process_all_files()

import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_multitaper  # Used for multitaper PSD estimation
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

# === Configuration ===
FILE_PATH = 'Andy.csv'
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

# === 1. Load Data ===
print("\n=== Loading Data ===")
df = pd.read_csv(FILE_PATH)
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
    exit()
else:
    epochs = mne.Epochs(raw_clean, events=selected_events, event_id=EVENT_ID_MAP,
                        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=None, preload=True)
    print(epochs)
    for cond in EVENT_ID_MAP:  # now iterating over keys which are event names
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
# Extract epochs and labels for ML – restrict to our two conditions
X = epochs.get_data()    # shape: (n_epochs, n_channels, n_times)
y = epochs.events[:, 2]  # event IDs as labels
valid_idx = np.isin(y, list(EVENT_ID_MAP.values()))
X = X[valid_idx]
y = y[valid_idx]
if len(y) > 0:
    csp = CSP(n_components=4, reg=None, log=True)
    clf_pipeline = Pipeline([('CSP', csp), ('LDA', LinearDiscriminantAnalysis())])
    scores = cross_val_score(clf_pipeline, X, y, cv=5)
    print("Classification accuracy (CSP + LDA): {:.2f}%".format(np.mean(scores) * 100))
else:
    print("Insufficient epochs for ML classification.")

# === 8. Visualization ===
print("\n=== Visualizing Results ===")
# Plot average evoked responses for each condition
evoked_dict = {cond: epochs[cond].average() for cond in EVENT_ID_MAP}
mne.viz.plot_compare_evokeds(evoked_dict, picks=EEG_CHANNELS, combine='mean', show=True)

# Optionally, plot the average ERP for one condition (e.g., NonDelayedBeep)
if 'NonDelayedBeep' in evoked_dict:
    evoked = evoked_dict['NonDelayedBeep']
    evoked.plot(title='Average ERP for NonDelayedBeep', show=True)

print("\n=== Script Complete ===")

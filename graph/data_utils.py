from lib import *

def convert_to_bipolar(eeg_window, ch_names, bipolar_pairs):
    bipolar_data = []
    bipolar_names = []
    
    name_to_idx = {name: i for i, name in enumerate(ch_names)}

    for (A, B) in bipolar_pairs:
        idxA, idxB = name_to_idx[A], name_to_idx[B]
        bipolar_data.append(eeg_window[idxA] - eeg_window[idxB])
        bipolar_names.append(f"{A}-{B}")

    return np.array(bipolar_data), bipolar_names
# ------------------------------get data_paths, labels, sub_id -----------------------------------
# def get_data_paths_and_labels(data_dir, tsv_path, num_class = 'all3'):
def aheap_get_paths(data_dir, tsv_path, num_class = 'all3'):
    df = pd.read_table(tsv_path)

    if num_class == 'all2':
        label_mapping = {"A": 1, "F":1, "C": 0}
    elif num_class == 'all3':
        label_mapping = {"A": 1, "F":2, "C": 0}
    elif num_class == 'adhc':
        label_mapping = {"A": 1, "C": 0}
        df = df[~(df['Group'] == 'F')]
    elif num_class == 'ftdhc':
        label_mapping = {"F":1, "C": 0}
        df = df[~(df['Group'] == 'A')]
    elif num_class == 'adftd':
        label_mapping = {"A": 1, "F":0}
        df = df[~(df['Group'] == 'C')]
    else:
        raise ValueError(f"Invalid num_class '{num_class}'.")

    participant_labels = {row['participant_id']: label_mapping[row['Group']] for _, row in df.iterrows()}
    data_paths = []
    labels = []
    sub_id_list = []

    for sub_id in df['participant_id'].tolist():
        if sub_id == "sub-086":
            continue
        sub_path = os.path.join(data_dir, sub_id, 'eeg', f"{sub_id}_task-eyesclosed_eeg.set")
        if os.path.exists(sub_path):
            label = participant_labels.get(sub_id, -1)
            if label != -1:
                data_paths.append(sub_path)
                labels.append(label)
                sub_id_list.append(sub_id)
            else:
                print(f"Warning: Label for {sub_id} not found. Skipping.")
    # print(f"Total number of file paths: {len(data_paths)}")
    return data_paths, labels, sub_id_list

# def process_preprocessed_data(csv_path, data_folder):
def dryad_get_paths(csv_path, data_folder):
    df = pd.read_csv(csv_path)
    data_paths = []
    labels = []
    sub_id_list = []

    for _, row in df.iterrows():
        file_name = row["file"]
        label_str = str(row["label"]).strip().upper()  # e.g., "HC" or "AD"
        subject_id = os.path.splitext(file_name)[0]    # remove ".set"

        # Construct full file path
        file_path = os.path.join(data_folder, file_name)

        # Determine numeric label
        if label_str == "HC":
            label = 0
        elif label_str == "AD":
            label = 1
        elif label_str == "MCI":
            label = 2
        else:
            continue  # Skip unknown label

        # Add to lists
        data_paths.append(file_path)
        labels.append(label)
        sub_id_list.append(subject_id)

    # print(f"✅ Loaded {len(data_paths)} preprocessed EEG files from {data_folder}")
    return data_paths, labels, sub_id_list


# def process_json_data(json_path, data_folder, serial_length=5):
def caueeg_get_paths(json_path, data_folder, serial_length=5):
    with open(json_path, 'r') as f:
        data_json = json.load(f)

    data_paths = []
    labels = []
    sub_id_list = []

    for entry in data_json["data"]:
        serial = entry["serial"]
        symptoms = [s.lower() for s in entry.get("symptom", [])]

        # Pad serial number (e.g., "20" → "00020")
        serial_padded = serial.zfill(serial_length)
        file_path = os.path.join(data_folder, f"{serial_padded}.edf")

        # Assign label
        if any(s in symptoms for s in ["normal", "smi"]):
            label = 0  # Normal
        elif any(s in symptoms for s in ["dementia", "ad", "load"]):
            label = 1  # Dementia
        elif any(s in symptoms for s in ["mci", "mci_amnestic", "mci_vascular"]):
            label = 2  # MCI
        else:
            continue  # Skip if label not recognized

        data_paths.append(file_path)
        labels.append(label)
        sub_id_list.append(serial_padded)

    # === Print label summary ===
    label_counts = Counter(labels)
    print("\n=== Label Summary ===")
    print(f"Normal (0):   {label_counts.get(0, 0)} subjects")
    print(f"Dementia (1): {label_counts.get(1, 0)} subjects")
    print(f"MCI (2):      {label_counts.get(2, 0)} subjects")

    print("======================\n")

    return data_paths, labels, sub_id_list




# ------------------------------process eeg file-----------------------------------
def preprocess_eeg(file_path, dataset):
    logging.getLogger('mne').setLevel(logging.ERROR)
    if dataset.lower() == "caueeg":
        raw = mne.io.read_raw_edf(file_path, preload=True)
    else:
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
    eeg_data = raw.get_data()  # (channels, timepoints)
    # channel_names = raw.ch_names
    # print(channel_names)
    # ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    scaler = StandardScaler()
    eeg_data_cleaned = scaler.fit_transform(eeg_data.T).T
    return eeg_data_cleaned
def preprocess_eeg_update(file_path, dataset, sid, save_folder):
    logging.getLogger('mne').setLevel(logging.ERROR)
    if dataset.lower() == "caueeg":
        raw = mne.io.read_raw_edf(file_path, preload=True)
    else:
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        sfreq = 500
    eeg_data = raw.get_data()  # (channels, timepoints)
    # channel_names = raw.ch_names
    # print(channel_names)
    # ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

    # --- DIAGNOSTIC STEP: Check Frequency Content ---
    # We calculate PSD on the data BEFORE scaling to see the true spectrum
    freqs, psd = welch(eeg_data, sfreq, nperseg=int(sfreq))
    avg_psd = np.mean(psd, axis=0)
    
    # Identify the frequency with the maximum power
    peak_freq = freqs[np.argmax(avg_psd)]
    
    # Calculate power ratio below 1Hz to see if filter worked
    low_freq_mask = freqs < 1.0
    ultra_low_power = np.sum(avg_psd[low_freq_mask]) / np.sum(avg_psd)
    
    # print(f"--- Frequency Check for {file_path} ---")
    # print(f"Sampling Rate: {sfreq} Hz")
    # print(f"Peak Frequency: {peak_freq:.2f} Hz")
    # print(f"Power below 1Hz: {ultra_low_power*100:.2f}% of total signal")
    
    # Optional: Plot the spectrum to visually confirm the 0.5-45Hz range
    # plt.semilogy(freqs, avg_psd)
    # plt.title("Global PSD (Verify 0.5-45Hz range)")
    # plt.xlim(0, 60)
    # plt.savefig(f"{save_folder}/{sid}_raw_PSD.png")
    # plt.close()
    # plt.show()
    # -----------------------------------------------

    scaler = StandardScaler()
    eeg_data_cleaned = scaler.fit_transform(eeg_data.T).T
    return eeg_data_cleaned

# ------------------------------data preparation for baseline-----------------------------------

def data_preparation(dataset, data_paths, labels, sub_id_list, fs, T=4, overlap=2):
    eeg_data_list = []
    processed_labels = []
    window_size = T * fs
    step_size = (T - overlap) * fs
    target_length = window_size
    num_subsegments = int(T * 2)  # 4s / 0.5s = 8
    segment_length = fs // 2      # 250 samples

    for i in range(len(data_paths)):
        eeg_data_cleaned = preprocess_eeg(data_paths[i], dataset)  # Shape: (C, timepoints)
        # if sfreq != fs:
        #     print("Wrong Sampling Rate")
        #     break
        n_channels, n_timepoints = eeg_data_cleaned.shape
        num_windows = (n_timepoints - window_size) // step_size + 1
        for k in range(num_windows):
            start = k * step_size
            end = start + window_size
            eeg_window = eeg_data_cleaned[:, start:end]  # Shape: (C, window_size)
            eeg_data_list.append(eeg_window)
            processed_labels.append(labels[i])

    eeg_data_array = np.array(eeg_data_list)  
    labels_array = np.array(processed_labels)
    eeg_data_array = np.expand_dims(eeg_data_array, axis=1)
    return eeg_data_array, labels_array


def data_bipolar_preparation(dataset, data_paths, labels, sub_id_list, fs, T=4, overlap=2):
    eeg_data_list = []
    processed_labels = []
    window_size = T * fs
    step_size = (T - overlap) * fs
    target_length = window_size
    num_subsegments = int(T * 2)  # 4s / 0.5s = 8
    segment_length = fs // 2      # 250 samples
    channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

    bipolar_pairs = [
        # ("F7", "Fp1"),
        # ("F3", "Fp1"),
        # ("Fp1", "Fz"), 
        # ("F8", "Fp2"), 
        # ("F4", "Fp2"),
        # ("Fz", "Fp2"),
        # ("Fp1", "Fp2"), 
        ("F7", "F3"),
        ("F3", "Fz"),
        ("F4", "Fz"), 
        ("F8", "F4"), 
        ("F3", "C3"),
        ("Fz", "Cz"),
        ("F4", "C4"), 
        ("C3", "T3"),
        ("C3", "Cz"),
        ("C4", "Cz"),
        ("C4", "T4"), 
        ("T3", "T5"),
        ("C3", "P3"),
        ("Cz", "Pz"),
        ("C4", "P4"), 
        ("T4", "T6"), 
        ("P3", "Pz"),
        ("P4", "Pz"),
        ("P4", "O2"), 
        ("P3", "O1"),
        ("T6", "O2"), 
        ("T5", "O1"),
        ("O1", "O2")
    ]
    for i in range(len(data_paths)):
        eeg_data_cleaned = preprocess_eeg_update(data_paths[i], dataset)  # Shape: (C, timepoints)
        # eeg_data_cleaned = preprocess_eeg(data_paths[i], dataset)  # Shape: (C, timepoints)

        bipolar_data, bipolar_names = convert_to_bipolar(eeg_data_cleaned, channel_names, bipolar_pairs)    
        n_channels, n_timepoints = bipolar_data.shape
        # print("bipolar_data.shape", bipolar_data.shape)

        num_windows = (n_timepoints - window_size) // step_size + 1
        for k in range(num_windows):
            start = k * step_size
            end = start + window_size
            eeg_window = bipolar_data[:, start:end]  # Shape: (C, window_size)
            # print("eeg_window", eeg_window.shape)
            eeg_data_list.append(eeg_window)
            processed_labels.append(labels[i])

    eeg_data_array = np.array(eeg_data_list)  
    labels_array = np.array(processed_labels)
    eeg_data_array = np.expand_dims(eeg_data_array, axis=1)
    return eeg_data_array, labels_array


class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels):
        self.eeg_data = torch.tensor(eeg_data, dtype=torch.float32)  # (batch, 1, 19, 2000)
        self.labels = torch.tensor(labels, dtype=torch.long)          # (batch,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx]
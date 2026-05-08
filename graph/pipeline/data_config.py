FEATURE_DIM_DICT = {
                    "rbp": 5,       # Relative Band Power
                    "hjorth": 3,    # Hjorth parameters
                    "stats": 4,     # Statistical features
                    "energies": 6,  # Wavelet energies
                    "svd": 1,       # SVD entropy
                    "zero": 1,      # Zero-crossing rate
                    "hfd": 1        # Higuchi fractal dimension
                    }
BANDS = [
    (1, 4),   # delta
    (4, 8),     # theta
    (8, 13),    # alpha
    (13, 30),   # beta
    (30, 45)    # gamma
]

NUM_BANDS = len(BANDS)

BANDS_DICT = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
    }


AHEAP_DIR = '/mnt/data/anphan/derivatives'
AHEAP_TSV_PATH = '/home/anphan/Documents/EEG_Project/participants.tsv'

CAUEEG_DIR = "/mnt/data/anphan/CAUEEG/caueeg-dataset"

MONO_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"
]

MONOFIXEDGES = [
        ("F7", "Fp1"),
        ("F3", "Fp1"),
        ("Fp1", "Fz"), 
        ("F8", "Fp2"), 
        ("F4", "Fp2"),
        ("Fz", "Fp2"),
        ("Fp1", "Fp2"), 
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



region_to_channels = {
    "frontal":  ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central":  ["C3", "C4", "Cz"],
    "parietal": ["P3", "P4", "Pz"],
    "occipital":["O1", "O2"],
    "temporal": ["T3", "T4", "T5", "T6"],
}


region_hyperedges = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central": ["C3", "C4", "Cz"],
    "parietal": ["P3", "P4", "Pz"],
    "temporal": ["T3", "T4", "T5", "T6"],
    "occipital": ["O1", "O2"],

    # "left_frontal": ["Fp1", "F3", "F7"],
    # "right_frontal": ["Fp2", "F4", "F8"],
    # "left_central": ["C3"],
    # "right_central": ["C4"],
    # "left_parietal": ["P3"],
    # "right_parietal": ["P4"],
    # "left_temporal": ["T3", "T5"],
    # "right_temporal": ["T4", "T6"],
    # "left_occipital": ["O1"],
    # "right_occipital": ["O2"],

    "midline": ["Fz", "Cz", "Pz"],

    "fronto_central": ["F3", "F4", "Fz", "C3", "C4", "Cz"],
    "centro_parietal": ["C3", "C4", "Cz", "P3", "P4", "Pz"],
    "parieto_occipital": ["P3", "P4", "Pz", "O1", "O2"],
    "fronto_temporal_left": ["Fp1", "F3", "F7", "T3"],
    "fronto_temporal_right": ["Fp2", "F4", "F8", "T4"],
    "temporo_parietal_left": ["T3", "T5", "P3"],
    "temporo_parietal_right": ["T4", "T6", "P4"],

    # "left_hemisphere": ["Fp1", "F3", "F7", "C3", "P3", "O1", "T3", "T5"],
    # "right_hemisphere": ["Fp2", "F4", "F8", "C4", "P4", "O2", "T4", "T6"],
}


bi23_channel_names = [
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

from itertools import combinations
from os.path import join


SEED = 42
RATE = 4000
MAX_DURATION = 30

PATIENT_DF_FILE = "patient_df.csv"
SAMPLES_DF_FILE = "samples_df.csv"
AUDIO_DATA_FILE = "audio_data.npy"

pre_config = {
    "sr": RATE,
    "freq_highcut": 150,  # 80 100 150  -> Latest: 100
    "order_highcut": 10,
    "freq_lowcut": 800,  # 900 1000 1250  -> Latest: 900
    "order_lowcut": 4
}

feat_config = {
    "transform": "logmel",
    "sr": RATE,
    "n_fft": 256,
    "hop_length": 64,  # 128
    "center": True,
    "n_mels": 32,  # 32 64  -> Latest: 32
    "fmin": 250,  # 200 250
    "fmax": 750,  # 750 900 1000  -> Latest: 900
    "n_mfcc": 40,
    "roll_percent": 0.85
}

split_config = {
    "sr": RATE,
    "max_duration": MAX_DURATION,
    "pad_audio": False,
    "split_duration": 5,
    "overlap": 0.5,
    "center": False
}

filenames_config = {
    "model_file": "{}_D{}_V{}_T{}.pt",
    "feature_file": "{}_features_D{}_V{}_T{}.npy",
    "output_file": "{}_outputs_D{}_V{}_T{}.npy",
    "temp_file": "{}_temp_D{}_V{}_T{}.npy",
    "aggregate_file": "{}_agg_D{}_V{}_T{}.csv",
}

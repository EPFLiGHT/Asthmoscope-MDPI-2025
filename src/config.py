"""Configuration for Asthmoscope experiment."""
from itertools import combinations
from os.path import join
from pathlib import Path

# Random seed for reproducibility
SEED = 42

# Audio configuration
RATE = 4000
MAX_DURATION = 30

# Data file names
PATIENT_DF_FILE = "patient_df.csv"
SAMPLES_DF_FILE = "samples_df.csv"
AUDIO_DATA_FILE = "audio_data.npy"

# Preprocessing configuration
pre_config = {
    "sr": RATE,
    "freq_highcut": 150,
    "order_highcut": 10,
    "freq_lowcut": 800,
    "order_lowcut": 4
}

# Feature extraction configuration
feat_config = {
    "transform": "logmel",
    "sr": RATE,
    "n_fft": 256,
    "hop_length": 64,
    "center": True,
    "n_mels": 32,
    "fmin": 250,
    "fmax": 750,
    "n_mfcc": 40,
    "roll_percent": 0.85
}

# Audio splitting configuration
split_config = {
    "sr": RATE,
    "max_duration": MAX_DURATION,
    "pad_audio": False,
    "split_duration": 5,
    "overlap": 0.5,
    "center": False
}

# File naming patterns
filenames_config = {
    "model_file": "{}_D{}_V{}_T{}.pt",
    "feature_file": "{}_features_D{}_V{}_T{}.npy",
    "output_file": "{}_outputs_D{}_V{}_T{}.npy",
    "temp_file": "{}_temp_D{}_V{}_T{}.npy",
    "aggregate_file": "{}_agg_D{}_V{}.csv",
}

# Network architecture
# Note: Only "dense" feature model (Cnn10Att) is used in this experiment
network = {
    "feature_model": "dense",  # Corresponds to Cnn10Att architecture
    "n_feats": feat_config["n_mels"],
    "conv_channels": 32,
    "fc_features": 128,
    "classes_num": 1,
    "conv_dropout": 0.2,
    "fc_dropout": 0.5,
    "feat_config": feat_config,
    "time_drop_width": 20,
    "freq_drop_width": 4
}

# Optimizer configuration
optimizer = {
    "name": "adamw",
    "parameters": {
        "lr": 1e-4,
        "weight_decay": 0.005
    }
}

# Main pipeline configuration
pipeline_config = {
    "target": ['asthma'],
    "with_splits": True,
    "stetho": ['L', 'E'],
    "preprocessing": ["highpass", "lowpass"],
    "augment": False,
    "features": ["logmel"],
    "pre_config": pre_config,
    "feat_config": feat_config,
    "split_config": split_config,
    "train_loc": ['GVA'],
    "cv_folds": list(combinations(range(5), 2)),
    "epochs": 90,
    "validation_start": 1,
    "batch_size": 64,
    "balanced_sampling": True,
    "sampling_alpha": 0.6,
    "balanced_sampling_additional_columns": ["stethoscope"],
    "mixup": False,
    "mixup_alpha": 0.3,
    "network": network,
    "exclude": [],
    "optimizer": optimizer,
    "loss": "bce",
    "save_outputs": False
}


def get_experiment_config(experiment_id="asthmoscope_mdpi_2025"):
    """Get configuration for a specific experiment."""
    # Set up experiment directories
    experiment_dirpath = Path("experiments") / experiment_id
    experiment_out_dirpath = experiment_dirpath / "out"
    out_data_path = experiment_out_dirpath / "data_files"
    out_data_path.mkdir(parents=True, exist_ok=True)
    
    # Update filenames config with experiment paths
    config = pipeline_config.copy()
    config.update({
        "experiment_dirpath": str(experiment_dirpath),
        "out_folder": str(experiment_out_dirpath),
        "patient_df_path": str(out_data_path / PATIENT_DF_FILE),
        "samples_df_path": str(out_data_path / SAMPLES_DF_FILE),
        "samples_path": str(out_data_path / AUDIO_DATA_FILE),
    })
    config.update(filenames_config)
    
    return config

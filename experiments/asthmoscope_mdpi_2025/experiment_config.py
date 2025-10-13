from itertools import combinations

from src import config

pre_config = config.pre_config
feat_config = config.feat_config
split_config = config.split_config

network = {
    "feature_model": "dense",  # cnn6 cnn10 dense
    "n_feats": feat_config["n_mels"],
    "conv_channels": 32,  # 32
    "fc_features": 128,
    "classes_num": 1,  # 2
    "conv_dropout": 0.2,
    "fc_dropout": 0.5,
    "feat_config": feat_config,
    "time_drop_width": 20,  # changed
    "freq_drop_width": 4
}

optimizer = {
    "name": "adamw",  # adamw
    "parameters": {
        "lr": 1e-4,  # 1e-4
        "weight_decay": 0.005  # 0.005
    }
}

pipeline_config = dict(
    target=['asthma'],  # 0: healthy, 1: bacterial pn., 2: viral pn., 3: bronchite obstructive, 4: asthme, 5: bronchiolite
    # target=['pathological'],
    with_splits=True,
    stetho=['L', 'E'],
    preprocessing=["highpass", "lowpass"],
    augment=False,
    features=["logmel"],
    pre_config=pre_config,
    feat_config=feat_config,
    split_config=split_config,
    train_loc=['GVA'],  # "GVA", "POA", "YAO", "DKR", "MAR", "RBA"
    cv_folds=list(combinations(range(5), 2)),  # combinations(range(5), 2) [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    epochs=90,  # 90 80 150 TODO: CHANGEME
    validation_start=1,  # 70 60 100 TODO: CHANGEME
    batch_size=64,  # 32 64 128
    balanced_sampling=True,  # it does not make sense for Asthmoscope polyclinique
    sampling_alpha=0.6,  # changeme
    balanced_sampling_additional_columns=["stethoscope"],  # changeme
    mixup=False,
    mixup_alpha=0.3,  # PANN: 1.0 - Mixup: 0.4
    network=network,
    exclude=[],
    optimizer=optimizer,
    loss="bce",
    save_outputs=False
)

if __name__ == '__main__':
    print('hello')
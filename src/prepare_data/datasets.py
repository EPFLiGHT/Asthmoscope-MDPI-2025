import numpy as np
import torch
from audiomentations import AddGaussianSNR
from torch.utils.data import Dataset

from src.audio_frontend import features as feats
from src.config import pre_config, split_config


class AudioDataset(Dataset):

    def __init__(self, samples_df, data, target,
                 preprocessing=["highpass"], augment=False,
                 pre_config=pre_config, split_config=split_config, train=True):

        self.samples_df = samples_df.reset_index().rename(columns={"index": "old_index"})

        self.data = data
        self._preprocess_data(preprocessing, pre_config)

        # labels = self.samples_df.asthma.astype(int)
        labels = self.samples_df[target[0]].astype(int)
        self.samples_df["label"] = labels

        # self.target_codes = target
        self.targets = np.zeros((labels.size, 2))
        self.targets[np.arange(labels.size), labels] = 1

        locations = sorted(self.samples_df.location.unique())
        self.location_code = {loc: i for i, loc in enumerate(locations)}

        self.split_config = split_config
        self.train = train

        if augment:
            self.gsnr = AddGaussianSNR(max_SNR=0.1, p=0.5)

    @property
    def samples_df(self):
        return self._samples_df

    @samples_df.setter
    def samples_df(self, df):
        self._samples_df = df

    def _preprocess_data(self, preprocessing, pre_config):
        filters = feats.AudioFeatures(features=[], preprocessing=preprocessing, pre_config=pre_config)
        for i, n_samples in enumerate(self.samples_df.end.values):
            audio = self.data[i][:n_samples]
            filtered_audio = filters.transform(audio)
            self.data[i][:n_samples] = filtered_audio

    def get_class_counts(self):
        class_sample_count = np.unique(self.samples_df["label"], return_counts=True)[1]
        return torch.from_numpy(class_sample_count.astype(np.float32))

    def __len__(self):
        return len(self.samples_df)

    def __getitem__(self, i):
        y = np.array([self.samples_df["label"].values[i]]).astype(np.float32)

        sample_length = int(self.split_config["sr"] * self.split_config["split_duration"])
        n_samples = self.samples_df.iloc[i].end
        audio = self.data[i][:n_samples]
        if self.train:
            # If duration of recording (n_samples) is smaller than split duration
            # (sample_length), make an audio of split duration containing the whole
            # original audio starting at a random time, and zero-padded
            if n_samples < sample_length:
                new_audio = np.zeros(sample_length, dtype=audio.dtype)
                random_start = np.random.choice(sample_length - n_samples)
                new_audio[random_start:(random_start + n_samples)] = audio
                audio = new_audio
            # If duration of recording (n_samples) is larger than split duration
            # (sample_length), make an audio of split duration containing an excerpt of
            # the original audio starting at a random time
            else:
                end_idx = n_samples - sample_length
                random_start = np.random.choice(end_idx)
                audio = audio[random_start:(random_start + sample_length)]

            # Added Augmentation
            if hasattr(self, "gsnr"):
                audio = self.gsnr(audio, self.split_config["sr"])

        x = audio.astype(np.float32)

        batch_dict = {
            "sample_idx": torch.from_numpy(np.array(i)),
            "data": torch.from_numpy(x),
            "target": torch.from_numpy(y)
        }

        return batch_dict


class AudioDatasetWithSplits(Dataset):

    def __init__(self, samples_df, data, target,
                 preprocessing=["highpass"], augment=False,
                 pre_config=pre_config, split_config=split_config, train=True,
                 chosen_split_duration=5):

        self.orig_samples_df = samples_df.reset_index().rename(columns={"index": "old_index"}).copy()

        self.data = data
        self._preprocess_data(preprocessing, pre_config)

        # labels = self.orig_samples_df.asthma.astype(int)
        labels = self.orig_samples_df[target[0]].astype(int)
        self.orig_samples_df["label"] = labels

        # self.target_codes = target
        self.targets = np.zeros((labels.size, 2))
        self.targets[np.arange(labels.size), labels] = 1

        locations = sorted(self.orig_samples_df.location.unique())
        self.location_code = {loc: i for i, loc in enumerate(locations)}

        self.split_config = split_config
        self.train = train

        # Make actual samples_df containing every 5 sec split as record
        self.chosen_split_duration = chosen_split_duration
        self.orig_samples_df['n_splits'] = self.orig_samples_df['duration'] // chosen_split_duration
        self.orig_samples_df['n_splits_list'] = self.orig_samples_df['n_splits'].apply(
            lambda n_splits: list(range(int(n_splits))))
        self.samples_df = self.orig_samples_df.explode('n_splits_list').reset_index().rename(
            columns={'index': 'index_before_split', 'n_splits_list': 'split_id'})

        sample_length = int(split_config["sr"] * split_config["split_duration"])
        self.samples_df['start_sample'] = self.samples_df['split_id'] * sample_length
        self.samples_df['end_sample'] = self.samples_df['start_sample'] + sample_length

        if augment:
            self.gsnr = AddGaussianSNR(max_SNR=0.1, p=0.5)

    @property
    def orig_samples_df(self):
        return self._orig_samples_df

    @orig_samples_df.setter
    def orig_samples_df(self, df):
        self._orig_samples_df = df

    def _preprocess_data(self, preprocessing, pre_config):
        filters = feats.AudioFeatures(features=[], preprocessing=preprocessing, pre_config=pre_config)
        for i, n_samples in enumerate(self.orig_samples_df.end.values):
            audio = self.data[i][:n_samples]
            filtered_audio = filters.transform(audio)
            self.data[i][:n_samples] = filtered_audio

    def get_class_counts(self):
        class_sample_count = np.unique(self.samples_df["label"], return_counts=True)[1]
        return torch.from_numpy(class_sample_count.astype(np.float32))

    def __len__(self):
        return len(self.samples_df)

    def __getitem__(self, i):
        r = self.samples_df[['label', 'index_before_split', 'split_id',
                             'start_sample', 'end_sample']].iloc[i]

        y = np.array([r["label"]]).astype(np.float32)

        n_samples = self.samples_df.iloc[i].end
        audio = self.data[r['index_before_split']][r['start_sample']:r['end_sample']]

        # Added Augmentation
        if hasattr(self, "gsnr"):
            audio = self.gsnr(audio, split_config["sr"])

        x = audio.astype(np.float32)

        batch_dict = {
            "sample_idx": torch.from_numpy(np.array(i)),
            "data": torch.from_numpy(x),
            "target": torch.from_numpy(y)
        }

        return batch_dict

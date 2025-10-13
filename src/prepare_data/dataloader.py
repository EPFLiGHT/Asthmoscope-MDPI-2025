import numpy as np
from torch.utils.data import BatchSampler, WeightedRandomSampler


class GroupedSampler(BatchSampler):
    def __init__(self, df, columns):
        self.grouped_samples = df.reset_index()\
                                 .groupby(columns)["index"]\
                                 .apply(list)

    def __iter__(self):
        return iter(self.grouped_samples.values)

    def __len__(self):
        return len(self.grouped_samples)


class WeightedRandomBatchSampler(BatchSampler):  # Write test to check if it works
    def __init__(self, labels, batch_size, n_batches=None):
        class_sample_count = np.unique(labels, return_counts=True)[1]
        class_weights = 1. / class_sample_count
        sample_weights = class_weights[labels]
        self.sampler = WeightedRandomSampler(sample_weights, batch_size, replacement=True)
        self.n_batches = n_batches if n_batches is not None else len(sample_weights) // batch_size

    def __iter__(self):
        for batch_idx in range(self.n_batches):
            selected = list(self.sampler)
            yield selected

    def __len__(self):
        return self.n_batches

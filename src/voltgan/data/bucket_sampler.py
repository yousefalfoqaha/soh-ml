import random

from torch.utils.data import Sampler


class BucketSampler(Sampler):
    def __init__(
        self,
        dataset,
        max_batch_size: int,
        max_padding_threshold: int = 50,
        noise_scale: int = 5,
        min_length: int = 1000,
    ):
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.max_padding_threshold = max_padding_threshold
        self.noise_scale = noise_scale

        self.valid_instances = []
        for i, instance in enumerate(self.dataset.instances):
            seq_len = len(instance.data)
            if seq_len >= min_length:
                self.valid_instances.append((i, seq_len))

        print(
            f"Sampler initialized: Kept {len(self.valid_instances)} instances "
            f"(Discarded {len(self.dataset.instances) - len(self.valid_instances)} below {min_length} steps)."
        )

    def __iter__(self):
        noisy_indices = [
            (idx, length + random.randint(-self.noise_scale, self.noise_scale))
            for idx, length in self.valid_instances
        ]

        sorted_elements = sorted(noisy_indices, key=lambda x: x[1])

        batches = []
        current_batch = []
        batch_min_len = None

        for idx, current_len in sorted_elements:
            if not current_batch:
                current_batch.append(idx)
                batch_min_len = current_len
                continue

            length_diff_ok = (current_len - batch_min_len) <= self.max_padding_threshold
            size_ok = len(current_batch) < self.max_batch_size

            if length_diff_ok and size_ok:
                current_batch.append(idx)
            else:
                batches.append(current_batch)
                current_batch = [idx]
                batch_min_len = current_len

        if current_batch:
            batches.append(current_batch)

        random.shuffle(batches)

        self._len = len(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return getattr(
            self, "_len", max(1, len(self.valid_instances) // self.max_batch_size)
        )

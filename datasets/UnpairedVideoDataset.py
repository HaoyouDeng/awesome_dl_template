import random
from typing import Mapping

from torch.utils.data import Dataset

from datasets import VideoDataset


class UnpairedVideoDataset(Dataset):
    def __init__(
        self, folders_a, folders_b, clip_len, transform, recursive=True, return_image_path=False
    ):
        if isinstance(transform, Mapping):
            transform_a = transform["A"]
            transform_b = transform["B"]
        else:
            transform_a = transform
            transform_b = transform

        self.dataset_a = VideoDataset(
            folders_a, clip_len, transform_a, recursive, return_image_path
        )
        self.dataset_b = VideoDataset(
            folders_b, clip_len, transform_b, recursive, return_image_path
        )

    def __len__(self):
        return max(len(self.dataset_b), len(self.dataset_a))

    def __getitem__(self, idx):
        j = random.randint(0, len(self.dataset_b) - 1)
        result_a = self.dataset_a[idx % len(self.dataset_a)]
        result_b = self.dataset_b[j]
        return dict(a=result_a, b=result_b)

    def __repr__(self):
        attrs = ["dataset_a", "dataset_b"]
        attr_str = "".join([f"\t{a}={getattr(self, a)}\n" for a in attrs])
        return f"{self.__class__.__name__}(\n{attr_str})"

from datasets.UnpairedDataset import UnpairedDataset

class PairedDataset(UnpairedDataset):
    def __init__(
        self, folders_a, folders_b, transform, recursive=False, return_image_path=False
    ):
        super(PairedDataset, self).__init__(
            folders_a, folders_b, transform, recursive, return_image_path
        )

        self.dataset_a.files = sorted(self.dataset_a.files, key=lambda x: str(x))
        self.dataset_b.files = sorted(self.dataset_b.files, key=lambda x: str(x))
        assert len(self.dataset_b) == len(self.dataset_a)

    def __getitem__(self, idx):
        result_a = self.dataset_a[idx]
        result_b = self.dataset_b[idx]
        return dict(a=result_a, b=result_b)
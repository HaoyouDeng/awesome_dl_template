from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets.folder import is_image_file, default_loader

try:
    from datasets.basic_transform import pipeline
except:
    from .basic_transform import pipeline


class ImageDataset(Dataset):
    def __init__(self, folders, transform, recursive=False, return_image_path=False):
        if isinstance(folders, (str, Path)):
            folders = [
                folders,
            ]
        folders = [Path(f) for f in folders]
        for f in folders:
            assert f.exists(), f"{f} not exist, can not build ImageDataset"

        self.folders = folders
        self.recursive = recursive
        self.return_image_path = return_image_path
        self.files = self.list_image_files(self.folders, recursive=recursive)
        self.transform = transform if callable(transform) else pipeline(transform)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        attrs = ["folders", "return_image_path", "recursive"]
        attr_str = "".join([f"\t{a}={getattr(self, a)}\n" for a in attrs])
        return f"{self.__class__.__name__}(\n{attr_str})"

    @staticmethod
    def list_image_files(folders, recursive=False):
        pattern = "**/*" if recursive else "*"
        image_files = []
        for f in folders:
            if not f.exists():
                continue
            files = [file for file in f.glob(pattern) if is_image_file(file.name)]
            image_files.extend(files)
        return image_files

    def __getitem__(self, idx):
        file_path = self.files[idx]
        out = dict(image=self.transform(default_loader(file_path)))
        if self.return_image_path:
            out["path"] = str(file_path)
        return out

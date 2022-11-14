import pickle
import random
from io import BytesIO
from pathlib import Path
from typing import Iterable, Callable, Mapping
from zipfile import ZipFile

import lmdb
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import is_image_file, default_loader


def image_loader(path):
    if Path(path).suffix == ".npy":
        return np.load(path)
    return default_loader(path)


def is_valid_image_file(path):
    if Path(path).suffix == ".npy":
        return True
    return is_image_file(path)


def image_from_byte(data, filename=None):
    if str(filename).endswith(".npy"):
        return np.load(BytesIO(data))
    return Image.open(data)


def instantiate(module, description):
    class_name, args = class_name_and_args(description)
    return getattr(module, class_name)(**args)


def class_name_and_args(description):
    if isinstance(description, str):
        return description, dict()
    if isinstance(description, MutableMapping):
        if "_type" in description:
            args = deepcopy(description)
            return args.pop("_type"), args
        elif len(description) == 1:
            class_name, arguments = tuple(description.items())[0]
            arguments = dict(arguments.items())
            return class_name, arguments
        else:
            raise ValueError(
                f"Invalid `description`, Mapping `description` must contain "
                f"the type information, but got {description}"
            )
    else:
        raise TypeError(
            f"`description` must be `MutableMapping` or a str,"
            f" but got {type(description)}"
        )
        

def pipeline(pipeline_description: Iterable) -> Callable:
    transforms_list = []
    for pd in pipeline_description:
        transforms_list.append(utils.instantiate(torchvision.transforms, pd))

    return torchvision.transforms.Compose(transforms_list)


class ImageDataset(Dataset):
    def __init__(
        self,
        folders,
        transform,
        recursive=False,
        return_image_path=False,
        archive_type=None,
    ):
        if archive_type is None:
            archive_type = "files"
            if isinstance(folders, (str, Path)):
                root = Path(folders)
                if root.suffix == ".zip":
                    archive_type = "zip"
                elif (root / "data.mdb").exists() and (root / "lock.mdb").exists():
                    archive_type = "lmdb"
        assert archive_type in [
            "files",
            "zip",
            "lmdb",
        ], f"got invalid type: {archive_type}"

        if archive_type == "files":
            if isinstance(folders, (str, Path)):
                folders = [folders]
            folders = [Path(f) for f in folders]
            for f in folders:
                assert f.exists(), f"{f} not exist, can not build ImageDataset"
        else:
            folders = Path(folders)
            assert folders.exists(), f"{folders} not exist, can not build ImageDataset"

        self.folders = folders
        self.archive_type = archive_type
        self.recursive = recursive
        self.return_image_path = return_image_path
        self.files = self.list_files()
        self.transform = transform if callable(transform) else pipeline(transform)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        attrs = ["archive_type", "folders", "return_image_path", "recursive"]
        attr_str = "".join([f"\t{a}={getattr(self, a)}\n" for a in attrs])
        return f"{self.__class__.__name__}(\n{attr_str})"

    @staticmethod
    def _open_lmdb(lmdb_path):
        env = lmdb.open(
            str(lmdb_path),
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not env:
            raise IOError("Cannot open lmdb dataset", lmdb_path)
        return env

    def list_files(self):
        if self.archive_type == "files":
            return self.list_image_files(self.folders, recursive=self.recursive)
        elif self.archive_type == "lmdb":
            env = self._open_lmdb(self.folders)
            with env.begin(write=False) as txn:
                files = pickle.load(BytesIO(txn.get("filenames".encode("utf-8"))))
            env.close()
            return files
        elif self.archive_type == "zip":
            zf = ZipFile(self.folders)
            files = [f for f in zf.namelist() if is_image_file(f)]
            zf.close()
            return files
        else:
            raise ValueError(f"invalid archive_type: {self.archive_type}")

    @staticmethod
    def list_image_files(folders, recursive=False):
        pattern = "**/*" if recursive else "*"
        image_files = []
        for f in folders:
            if not f.exists():
                continue
            files = [file for file in f.glob(pattern) if is_valid_image_file(file.name)]
            image_files.extend(files)
        return image_files

    def load_router(self, p):
        if self.archive_type == "files":
            return image_loader(p)
        elif self.archive_type == "zip":
            # create the zipfile object at the first data iteration.
            # to prevent un-pickle-able error when using ddp
            if not hasattr(self, "_zipfile"):
                self._zipfile = ZipFile(self.folders)
            return image_from_byte(self._zipfile.open(p, "r"), p)
        elif self.archive_type == "lmdb":
            # create the environment object at the first data iteration.
            # to prevent un-pickle-able error when using ddp
            if not hasattr(self, "_txn"):
                env = self._open_lmdb(self.folders)
                self._txn = env.begin(write=False)
            return image_from_byte(self._txn.get(p.encode("utf-8")), p)
        else:
            raise ValueError(f"invalid archive_type: {self.archive_type}")

    def __getitem__(self, idx):
        file_path = self.files[idx]
        out = dict(image=self.transform(self.load_router(file_path)))
        if self.return_image_path:
            out["path"] = str(file_path)
        return out
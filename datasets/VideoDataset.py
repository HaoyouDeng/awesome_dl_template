from pathlib import Path
import re

from torch.utils.data import Dataset
from torchvision.datasets.folder import is_image_file, default_loader
import numpy as np

try:
    from datasets.basic_transform import pipeline
except:
    from .basic_transform import pipeline


class VideoDataset(Dataset):
    def __init__(self, folders, clip_len, transform, recursive=True, return_image_path=False):
        if isinstance(folders, (str, Path)):
            folders = [
                folders,
            ]
        folders = [Path(f) for f in folders]
        for f in folders:
            assert f.exists(), f"{f} not exist, can not build ImageDataset"

        self.folders = folders
        self.clip_len = clip_len
        self.recursive = recursive
        self.return_image_path = return_image_path
        self.files = self.list_image_files(self.folders, recursive=recursive)
        self.transform = transform if callable(transform) else pipeline(transform)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        attrs = ["folders", "clip_len", "return_image_path", "recursive"]
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
        frames, frames_path = self.load_frames(file_path, self.clip_len)
        out = dict(frames=frames)
        if self.return_image_path:
            out["path"] = str(file_path)
            out["frames_path"] = str(frames_path)
        return out

    def load_frames(self, file_path, clip_len):
        frames = []
        video_folder = Path(file_path).parents[0]
        video_frames = self.list_image_files(video_folder, recursive=False)
        video_frames = sorted(video_frames, key=lambda x: int(re.match(r'(\d+)', x.name).group()))
        frame_index = video_frames.index(file_path)

        t = int(clip_len/2)
        buffer = []
        a = -t
        b = t + 1 if clip_len % 2 == 1 else t
        for i in range(a, b):
            index = frame_index + i
            index = np.clip(index, 0, (len(video_frames)-1))
            buffer.append(video_frames[index])

        for name in buffer:
            frames.append(self.transform(default_loader(name)))

        return frames, buffer

from typing import Iterable, Callable, MutableMapping
from copy import deepcopy

import torch
import torchvision
import numpy as np


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
        transforms_list.append(instantiate(torchvision.transforms, pd))

    return torchvision.transforms.Compose(transforms_list)


class RandomHorizontalFlip:
    """Applies the :class:`~torchvision.transforms.RandomHorizontalFlip` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        p (float): probability of an image being flipped.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, p=0.5, inplace=False):
        self.p = p
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
        Returns:
            Tensor: Randomly flipped Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        flipped = torch.rand(tensor.size(0)) < self.p
        tensor[flipped] = torch.flip(tensor[flipped], [3])
        return tensor


class RandomVerticalFlip:
    """Applies the :class:`~torchvision.transforms.RandomVerticalFlip` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        p (float): probability of an image being flipped.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, p=0.5, inplace=False):
        self.p = p
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
        Returns:
            Tensor: Randomly flipped Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        flipped = torch.rand(tensor.size(0)) < self.p
        tensor[flipped] = torch.flip(tensor[flipped], [2])
        return tensor


class RandomCrop:
    """Applies the :class:`~torchvision.transforms.RandomCrop` transform to a batch of images.
    Args:
        size (int): Desired output size of the crop.
        padding (int, optional): Optional padding on each border of the image.
            Default is None, i.e no padding.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, size, padding=None, device="cpu"):
        self.size = size
        self.padding = padding
        self.device = device

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
        Returns:
            Tensor: Randomly cropped Tensor.
        """
        if self.padding is not None:
            padded = torch.zeros(
                (
                    tensor.size(0),
                    tensor.size(1),
                    tensor.size(2) + self.padding * 2,
                    tensor.size(3) + self.padding * 2,
                ),
                dtype=tensor.dtype,
                device=self.device,
            )
            padded[
            :, :, self.padding: -self.padding, self.padding: -self.padding
            ] = tensor
        else:
            padded = tensor

        h, w = padded.size(2), padded.size(3)
        th, tw = self.size, self.size
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = torch.randint(0, h - th + 1, (tensor.size(0),), device=self.device)
            j = torch.randint(0, w - tw + 1, (tensor.size(0),), device=self.device)

        rows = torch.arange(th, dtype=torch.long, device=self.device) + i[:, None]
        columns = torch.arange(tw, dtype=torch.long, device=self.device) + j[:, None]
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[
                 :,
                 torch.arange(tensor.size(0))[:, None, None],
                 rows[:, torch.arange(th)[:, None]],
                 columns[:, None],
                 ]
        return padded.permute(1, 0, 2, 3)


class RandomHorizontalFlip:
    """Applies the :class:`~torchvision.transforms.RandomHorizontalFlip` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        p (float): probability of an image being flipped.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, p=0.5, inplace=False):
        self.p = p
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
        Returns:
            Tensor: Randomly flipped Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        flipped = torch.rand(tensor.size(0)) < self.p
        tensor[flipped] = torch.flip(tensor[flipped], [3])
        return tensor


class RandomVerticalFlip:
    """Applies the :class:`~torchvision.transforms.RandomVerticalFlip` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        p (float): probability of an image being flipped.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, p=0.5, inplace=False):
        self.p = p
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
        Returns:
            Tensor: Randomly flipped Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        flipped = torch.rand(tensor.size(0)) < self.p
        tensor[flipped] = torch.flip(tensor[flipped], [2])
        return tensor


class RandomCrop:
    """Applies the :class:`~torchvision.transforms.RandomCrop` transform to a batch of images.
    Args:
        size (int): Desired output size of the crop.
        padding (int, optional): Optional padding on each border of the image.
            Default is None, i.e no padding.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, size, padding=None, device="cpu"):
        self.size = size
        self.padding = padding
        self.device = device

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
        Returns:
            Tensor: Randomly cropped Tensor.
        """
        if self.padding is not None:
            padded = torch.zeros(
                (
                    tensor.size(0),
                    tensor.size(1),
                    tensor.size(2) + self.padding * 2,
                    tensor.size(3) + self.padding * 2,
                ),
                dtype=tensor.dtype,
                device=self.device,
            )
            padded[
            :, :, self.padding: -self.padding, self.padding: -self.padding
            ] = tensor
        else:
            padded = tensor

        h, w = padded.size(2), padded.size(3)
        th, tw = self.size, self.size
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = torch.randint(0, h - th + 1, (tensor.size(0),), device=self.device)
            j = torch.randint(0, w - tw + 1, (tensor.size(0),), device=self.device)

        rows = torch.arange(th, dtype=torch.long, device=self.device) + i[:, None]
        columns = torch.arange(tw, dtype=torch.long, device=self.device) + j[:, None]
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[
                 :,
                 torch.arange(tensor.size(0))[:, None, None],
                 rows[:, torch.arange(th)[:, None]],
                 columns[:, None],
                 ]
        return padded.permute(1, 0, 2, 3)


def random_resize(image: torch.Tensor,
                  min_resolution, low, high):
    h, w = image.shape[-2:]
    factor = min(h / min_resolution, w / min_resolution)
    h, w = h / factor, w / factor
    size_scale = np.random.uniform(low, high, 1).item()
    image = torch.nn.functional.interpolate(
        image,
        (int(h * size_scale), int(w * size_scale)),
        mode="bilinear",
        align_corners=False,
    )
    if image.shape[-1] <= min_resolution or image.shape[-2] <= min_resolution:
        px = max(min_resolution - image.shape[-1] + 1, 0)
        py = max(min_resolution - image.shape[-2] + 1, 0)
        ptop = torch.randint(py, [1]).item() if py > 0 else 0
        pleft = torch.randint(px, [1]).item() if px > 0 else 0

        image = torch.nn.functional.pad(
            image,
            pad=[pleft, px - pleft, ptop, py - ptop],
            mode="replicate",
        )
    return image

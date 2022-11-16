from typing import Iterable, Union, Optional, MutableMapping
from copy import deepcopy

import torch
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_tensor

from PIL import Image
from datetime import datetime
from omegaconf import OmegaConf
from loguru import logger


def backup_config(config, output_dir):
    if config.local_rank <= 0:
        cur = datetime.now()
        config_name = f'config@{cur.strftime("%Y-%m-%d_%H-%M-%S")}.yml'

        (output_dir / config_name).write_text(OmegaConf.to_yaml(config))
        logger.debug(f"backup config at {output_dir / config_name}")
    else:
        pass


def init_weights(net, init_type="xavier_uniform", init_gain=1):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method:
            normal | xavier_normal | xavier_uniform | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier_normal":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "xavier_uniform":
                init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(
                    m.weight.data, a=0, mode="fan_in", nonlinearity="relu"
                )
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    logger.info("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


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


def grid_transpose(
    tensors: Union[torch.Tensor, Iterable], original_nrow: Optional[int] = None
) -> torch.Tensor:
    """
    batch tensors transpose.
    :param tensors: Tensor[(ROW*COL)*D1*...], or Iterable of same size tensors.
    :param original_nrow: original ROW
    :return: Tensor[(COL*ROW)*D1*...]
    """
    assert torch.is_tensor(tensors) or isinstance(tensors, Iterable)
    if not torch.is_tensor(tensors) and isinstance(tensors, Iterable):
        seen_size = None
        grid = []
        for tensor in tensors:
            if seen_size is None:
                seen_size = tensor.size()
                original_nrow = original_nrow or len(tensor)
            elif tensor.size() != seen_size:
                raise ValueError("expect all tensor in images have the same size.")
            grid.append(tensor)
        tensors = torch.cat(grid)

    assert original_nrow is not None

    cell_size = tensors.size()[1:]

    tensors = tensors.reshape(-1, original_nrow, *cell_size)
    tensors = torch.transpose(tensors, 0, 1)
    return torch.reshape(tensors, (-1, *cell_size))


def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert("RGB")
    if size is not None:
        size = (size, size) if isinstance(size, int) else size
        img = img.resize(size, Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize(
            (int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS
        )
    return to_tensor(img).unsqueeze_(0)


class mylog():
    def __init__(self, local_rank):
        self.local_rank = local_rank

    def add(self, file_path):
        if self.local_rank <= 0:
            logger.add(file_path)
    
    def log(
        self,
        module,
        message,
        **args,
    ):
        if self.local_rank <= 0:
            getattr(logger, module)(message, **args)

    def info(
        self,
        message,
    ):
        if self.local_rank <= 0:
            logger.info(message)
    
    def success(
        self,
        message,
    ):
        if self.local_rank <= 0:
            logger.success(message)


def bulit_tbwriter(tb_path, local_rank):
    if local_rank <= 0:
        if not tb_path.exists():
            tb_path.mkdir(parents=True)
        tb_writer = SummaryWriter(tb_path.as_posix())
        logger.info(f"bulit tb_writer: {tb_path}")
    else:
        tb_writer = None
    return tb_writer
        
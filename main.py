from operator import xor
import os
import time
from xml.etree.ElementPath import xpath_tokenizer_re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import tqdm
import fire
from typing import Iterable
from omegaconf import OmegaConf
from pathlib import Path
from loguru import logger

import utils
import networks
import datasets


def build_criterion(config, device):
    loss_weights = config.loss.weight

    criterion = dict()

    def _empty_l(*args, **kwargs):
        return 0

    def valid_l(name):
        return (loss_weights.get(name, 0.0) > 0)

    criterion["l1"] = torch.nn.L1Loss().to(device) if valid_l("l1") else _empty_l

    return criterion


def train(config):
    # define gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define output dir (checkpoints, logs)
    output_dir = Path(config.work_dir) / config.name / f"checkpoints"

    # backup train config
    utils.backup_config(config, output_dir)
    mylog = utils.mylog(local_rank=config.local_rank)
    mylog.add(file_path=output_dir/"file_{time}.log")

    # define model
    model = utils.instantiate(networks, config.model)
    utils.init_weights(model)
    mylog.info( f"build generator over: {model.__class__.__name__}")

    # bulid optimizer
    optimizer = getattr(torch.optim, config.optimizer._type)(
        [p for p in model.parameters() if p.requires_grad], lr=config.optimizer.lr
    )
    mylog.info(f"build optimizer over: {optimizer}")

    # define dataset & dataloader
    train_dataset = datasets.ImageDataset(**config.train.dataset)
    # TODO: need to fix to multi gpus
    if config.local_rank == -1:
        train_dataloader = DataLoader(train_dataset, **config.train.dataloader)
    else:
        train_dataloader = DataLoader(train_dataset, **config.train.dataloader)
    mylog.info(
        f"build train_dataset over: {train_dataset}. with config {config.train.dataset}"
    )
    mylog.info(f"{len(train_dataset)=}")

    evaluate_dataset = datasets.ImageDataset(**config.evaluate.dataset)
    evaluate_dataloader = DataLoader(evaluate_dataset, **config.evaluate.dataloader)
    mylog.info(f"build train_dataloader with config: {config.train.dataloader}")
    
    start_epoch = 1
    # resume model & optimizer from checkpoint
    if config.get("resume_from", None) is not None:
        checkpoint_path = Path(config.resume_from)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint '{checkpoint_path}' is not found")
        ckp = torch.load(checkpoint_path.as_posix(), map_location=torch.device("cpu"))
        start_epoch = ckp["epoch"] + 1
        model.load_state_dict(ckp["model"])
        optimizer.load_state_dict(ckp["optimizer"])
        mylog.success(f"load state_dict from {checkpoint_path}")

    # model to cuda
    if config.local_rank == -1:
        model = model.to(device)
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[config.local_rank],
            output_device=config.local_rank,
        )

    # define criterion
    criterion = build_criterion(config, device)

    # bulid tensorboard writer
    tb_path = Path(config.work_dir) / f"tb_logs" / config.name
    tb_writer = utils.bulit_tbwriter(tb_path, config.local_rank)

    for epoch in range(start_epoch, config.train.num_epoch + 1):
        epoch_start_time = time.time()
        logger.info(f"EPOCH[{epoch}/{config.train.num_epoch}] START")
        model.train()

        for iteration, batch in tqdm(
            enumerate(train_dataloader, 1),
            total=len(train_dataloader),
            leave=False,
            ncols=120,
        ):
            x, gt = batch
            y = model(x)

            loss = dict(
                l1=criterion["l1"](y, gt)
            )

            for k in loss:
                loss[k] = config.loss.weight.get(k, 0.0) * loss[k]

            optimizer.zero_grad()
            total_loss = sum(loss.values())
            total_loss.backward()
            optimizer.step()

            # write log
    

def main(config, *omega_options, gpus="all"):
    config = Path(config)
    assert config.exists(), f"config file {config} do not exists."

    omega_options = [str(o) for o in omega_options]
    cli_config = OmegaConf.from_cli(omega_options)
    if len(cli_config) > 0:
        logger.info(f"set options from cli:\n{OmegaConf.to_yaml(cli_config)}")

    config = OmegaConf.merge(OmegaConf.load(config), cli_config)


    if gpus == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info(f"using cpu")
    elif gpus == "all":
        torch.distributed.init_process_group(backend="nccl")
        config.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(config.local_rank)
        logger.info(f"using gpu: {config.local_rank}")
    elif gpus != "all":
        gpus = gpus if isinstance(gpus, Iterable) else [gpus]
        gpus = ",".join([str(g) for g in gpus])
        torch.cuda.set_device(int(gpus))
        logger.info(f"set CUDA_VISIBLE_DEVICES={gpus}")

    train(config)


if __name__ == "__main__":
    fire.Fire(main)
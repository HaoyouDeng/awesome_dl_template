import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision

from tqdm import tqdm
import fire
from typing import Iterable
from collections import defaultdict
from omegaconf import OmegaConf
from pathlib import Path
from loguru import logger

import utils
import networks
import datasets
from test import evaluate_fn


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

    if (config.local_rank<=0) and (not output_dir.exists()):
        output_dir.mkdir(parents=True)

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
    if config.local_rank == -1:
        train_dataloader = DataLoader(train_dataset, **config.train.dataloader)
    else:
        sampler = DistributedSampler(train_dataset)
        config.train.dataloader.shuffle = False
        train_dataloader = DataLoader(train_dataset, sampler=sampler, **config.train.dataloader)
    mylog.info(
        f"build train_dataset over: {train_dataset}. with config {config.train.dataset}"
    )
    mylog.info(f"len train_dataset: {len(train_dataset)}")

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
    if config.tb_dir == "None":
        tb_path = Path(config.work_dir) / config.name / f"tb_logs"
    else:
        tb_path = Path(config.tb_dir)
    tb_writer = utils.bulit_tbwriter(tb_path, config.local_rank)
    running_scalars = defaultdict(float)

    for epoch in range(start_epoch, config.train.num_epoch + 1):
        epoch_start_time = time.time()
        logger.info(f"EPOCH[{epoch}/{config.train.num_epoch}] START")
        model.train()
        if config.local_rank != -1:
            train_dataloader.sampler.set_epoch(epoch)

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
            if config.local_rank <= 0:
                for k, v in loss.items():
                    running_scalars[k] = running_scalars[k] + v.detach().mean().item()

                global_step = (epoch - 1) * len(train_dataloader) + iteration

                if global_step % config.log.tensorboard.scalar_interval == 0:
                    tb_writer.add_scalar(
                        "metric/total_loss", total_loss.detach().cpu().item(), global_step
                    )
                    for k in running_scalars:
                        v = running_scalars[k] / config.log.tensorboard.scalar_interval
                        running_scalars[k] = 0.0
                        tb_writer.add_scalar(f"loss/{k}", v, global_step)

                if global_step % config.log.tensorboard.image_interval == 0:
                    images = utils.grid_transpose(
                        [x, gt, y]
                    )
                    images = torchvision.utils.make_grid(
                        images, nrow=3, value_range=(0, 1), normalize=True
                    )
                    tb_writer.add_image(
                        f"train/x|gt|y",
                        images,
                        global_step,
                    )
        
        # log for each epoch        
        if config.local_rank <= 0:
            mylog.info(
                    f"EPOCH[{epoch}/{config.train.num_epoch}] END "
                    f"Taken {(time.time() - epoch_start_time) / 60.0:.4f} min"
                )

            # save checkpoint
            if epoch % config.log.checkpoint.interval_epoch == 0:
                to_save = dict(
                    model=model.state_dict(), optim=optimizer.state_dict(), epoch=epoch
                )
                torch.save(to_save, output_dir / f"epoch_{epoch:03d}.pt")
                logger.info(f"save checkpoint at {output_dir / f'epoch_{epoch:03d}.pt'}")
            # evaluate
            if epoch % config.log.evaluate.interval_epoch == 0:
                model.eval()
                with torch.no_grad():
                    metrics = evaluate_fn(
                        config, model, evaluate_dataloader, device=device
                    )
                model.train()
                logger.info(
                    f"EPOCH[{epoch}/{config.train.num_epoch}] metrics "
                    + "\t".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                )
                for m, v in metrics.items():
                    tb_writer.add_scalar(f"evaluate/{m}", v, epoch * len(train_dataloader))

    logger.success(f"train over.")
    

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
import torch
from torch.utils.data import DataLoader

from pathlib import Path
import fire
from loguru import logger
from omegaconf import OmegaConf
import kornia as K

import networks
import utils
import datasets


def evaluate_fn(
    config,
    evaluate_dataloader,
    model,
    device,
):
    
    return 0


def init_model(
    config,
    resume_from,
    device
):
    model = utils.instantiate(networks, config.model).to(device)

    checkpoint_path = Path(resume_from)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint '{checkpoint_path}' is not found")
    ckp = torch.load(checkpoint_path.as_posix(), map_location=torch.device("cpu"))
    model.load_state_dict(ckp["model"])
    logger.success(f"load model weights from {checkpoint_path} over")

    model.eval()

    return model


def evaluate(
    config,
    resume_from,
    gpu: int,
):
    torch.cuda.set_device(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = OmegaConf.load(config)
    model = init_model(config, resume_from, device)

    evaluate_dataset = datasets.ImageDataset(**config.evaluate.dataset)
    logger.info(f"build evaluate_dataset over: {evaluate_dataset}")
    logger.info(f"{len(evaluate_dataset)=}")

    evaluate_dataloader = DataLoader(evaluate_dataset, **config.evaluate.dataloader)
    logger.info(f"build evaluate_dataloader with config: {config.evaluate.dataloader}")

    metrics = evaluate_fn(
        config, evaluate_dataloader, model, device
    )
    logger.success(
        "evaluated metrics:\n"
        + "\n".join([f"\t{k}={v:.4f}" for k, v in metrics.items()])
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    fire.Fire(dict(
        evaluate=evaluate,
    ))
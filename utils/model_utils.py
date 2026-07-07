import random
import numpy as np
import torch
import logging
from datetime import datetime

def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, best_metric, best_metric_epoch):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_metric": best_metric,
        "best_metric_epoch": best_metric_epoch,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }, path)

def load_checkpoint(path, model, optimizer, scheduler, scaler, device,logger):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    
    if ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
        # cuda_rng_state is a list of tensors (one for each GPU), so we loop and move them to CPU
        cuda_states_cpu = [state.cpu() for state in ckpt["cuda_rng_state"]]
        torch.cuda.set_rng_state_all(cuda_states_cpu)

    np.random.set_state(ckpt["numpy_rng_state"])
    random.setstate(ckpt["python_rng_state"])
    logger.info(f"Resumed from epoch {ckpt['epoch']+1}, best Dice so far: {ckpt['best_metric']:.4f}")
    return ckpt["epoch"] +1 , ckpt["best_metric"], ckpt["best_metric_epoch"]

def setup_logger(run_dir):
    """Sets up a logger that outputs to both the terminal and a train.log file."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.FileHandler(run_dir / "train.log"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger

def setup_run_dir(root):
    """Creates a timestamped run directory and returns its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "runs" / f"{timestamp}_spleen_unetr_pp"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
import argparse
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

# MONAI Imports
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.losses import DeepSupervisionLoss


# Custom Imports
from data.dataset import SpleenDataset
from unetr_pp.unetr_pp_synapse import UNETR_PP
from utils import load_config
from utils import load_checkpoint, save_checkpoint, setup_logger, setup_run_dir
from peft_methods import apply_lora_to_model, freeze_for_lora, freeze_model, set_train_mode, trainable_param_summary,save_lora_state

set_determinism(seed=42)


def load_pretrained_synapse(model_path,logger):
    model = UNETR_PP(in_channels=1, out_channels=2, img_size=(64, 128, 128),
                      feature_size=16, num_heads=4, depths=[3, 3, 3, 3],
                      dims=[32, 64, 128, 256], do_ds=True)
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    pretrained_dict = checkpoint["state_dict"]
    # strip DataParallel prefix if present
    pretrained_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in pretrained_dict.items()}
    
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in pretrained_dict.items()
                     if k in model_dict and not k.startswith("out") and v.shape == model_dict[k].shape}

    matched, total = len(filtered_dict), len(model_dict)
    logger.info(f"Matched {matched}/{total} model keys from Synapse checkpoint.")

    if matched < 0.5 * total:
        raise RuntimeError(f"Only {matched}/{total} keys matched — backbone is likely NOT loaded. Check key naming.")

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=True)
    
    return model

def train_model():

    ROOT = Path(__file__).resolve().parent

    config_path = args.config if args.config else ROOT / "configs" / "config.yaml"
    config = load_config(str(config_path))
    resume_dir_str = args.resume if args.resume else config.training.get('resume_dir', "")
    
    if resume_dir_str and (ROOT / resume_dir_str).exists():
        run_dir = ROOT / resume_dir_str
        logger = setup_logger(run_dir)
        logger.info(f"RESUMING RUN in: {run_dir}")
    else:
        run_dir = setup_run_dir(ROOT)
        logger = setup_logger(run_dir)
        logger.info(f"NEW RUN in: {run_dir}")
        shutil.copy(config_path, run_dir / "config.yaml")


    data_dir = config.paths['data_dir']
    weights_path = config.paths['weights_path']

    max_epochs = args.epochs if args.epochs else config.training['max_epochs']
    val_interval = config.training['val_interval']
    patience_limit = config.training['early_stopping_patience']
    roi_size = tuple(config.training['roi_size'])

    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler(amp_device)

    writer = SummaryWriter(log_dir=str(run_dir))
    logger.info(f"Training on device: {device}")

    # DataLoaders
    spleen_dataset = SpleenDataset(data_dir=str(data_dir), batch_size=config.training['batch_size'], val_fraction=0.2, num_workers=4)
    train_loader, val_loader = spleen_dataset.get_spleen_dataloaders()

    model = load_pretrained_synapse(str(weights_path),logger=logger)

    method = config.peft.get('method', 'full_finetune')

    lora_layers = []

    if method == "linear_probe":
        frozen_bn = freeze_model(model, lambda name: name.startswith(("out1", "out2", "out3")))
    elif method == "lora":
        lcfg = config.peft['lora']
        lora_layers = apply_lora_to_model(model, target_attr=lcfg['target_attr'],
                                           r=lcfg['r'], alpha=lcfg['alpha'], dropout=lcfg.get('dropout', 0.0))
        frozen_bn = freeze_for_lora(model, trainable_prefixes=("out1", "out2", "out3"))
        print(frozen_bn)
    else:
        frozen_bn = []

    model = model.to(device)
    trainable, total, pct = trainable_param_summary(model)
    logger.info(f"Mode: {method} | Trainable: {trainable:,}/{total:,} ({pct:.2f}%)")


    optimizer = AdamW(model.parameters(), lr=config.training['learning_rate'], weight_decay=config.training['weight_decay'])

    steps_per_epoch = len(train_loader)
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)   

    warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    base_loss = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_function = DeepSupervisionLoss(loss=base_loss, weight_mode="exp")

    # Metrics & Post-Processing (From MONAI standard)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])


    full_ckpt_path = run_dir / "last_full_checkpoint.pth"
    small_ckpt_path = run_dir / "best_model_weights.pth"

    if full_ckpt_path.exists():
        start_epoch, best_metric, best_metric_epoch = load_checkpoint(
            full_ckpt_path, model, optimizer, scheduler, scaler, device, logger
        )
    else:
        start_epoch, best_metric, best_metric_epoch = 0, -1, -1

    patience_counter = 0
    logger.info("Starting Training Loop")

    # Training Loop
    for epoch in range(start_epoch,max_epochs):
        print("-" * 40)
        print(f"Epoch {epoch + 1}/{max_epochs}")
        
        # --- TRAINING ---
        set_train_mode(model, frozen_bn)
            
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad()
            with autocast(amp_device):
                outputs = model(inputs)  # list of 3 logits at different resolutions
                total_loss = loss_function(outputs, labels) # # downsampling + weighting + summing, all handled internally
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            
            epoch_loss += total_loss.item()
            logger.info(f"Step {step}/{len(train_loader)}, Train Loss: {total_loss.item():.4f}")
            writer.add_scalar("Train/Step_Loss", total_loss.item(), epoch * len(train_loader) + step)
            
        epoch_loss /= step
        writer.add_scalar("Train/Epoch_Average_Loss", epoch_loss, epoch + 1)
        logger.info(f"Epoch {epoch + 1} Average Loss: {epoch_loss:.4f}")


        # --- VALIDATION ---
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    
                    # Inference Wrapper: Forces sliding_window to only look at outputs[0]
                    def inference_wrapper(x):
                        with autocast(amp_device):
                            return model(x)[0]

                    val_outputs = sliding_window_inference(
                        inputs=val_inputs, 
                        roi_size=roi_size, 
                        sw_batch_size=config.training['sw_batch_size'], 
                        predictor=inference_wrapper,
                        overlap=0.5
                    )
                    
                    # Post-process batch into lists of separate discrete tensors
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                
                writer.add_scalar("Validation/Mean_Dice", metric, epoch + 1)

                logger.info(f"Validation Mean Dice: {metric:.4f}")
                
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    patience_counter = 0
      
                    if method == "lora":
                        save_lora_state(model, small_ckpt_path, trainable_prefixes=("out1", "out2", "out3"))
                    else:
                        torch.save(model.state_dict(), small_ckpt_path)
                    logger.info(f" New Best Model Saved! (Score: {best_metric:.4f})")

                else:
                    patience_counter += 1
                    logger.info(f"No improvement. Patience Counter: {patience_counter}/{patience_limit}")

                logger.info(f"Best Mean Dice: {best_metric:.4f} at Epoch: {best_metric_epoch}")

                if patience_counter >= patience_limit:
                    logger.info(f"Halting training early.")
                    break

        save_checkpoint(full_ckpt_path, epoch, model, optimizer, scheduler, scaler, best_metric, best_metric_epoch)

    writer.close()
    logger.info("Training Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, required=False, help="number of epochs to train")
    parser.add_argument("--resume", type=str, required=False, help="Path to resume training from")

    args = parser.parse_args()
    train_model()
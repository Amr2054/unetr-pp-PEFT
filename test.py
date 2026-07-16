"""
test.py — run inference with a trained (full fine-tune / linear probe / LoRA)
model, on either your dataset's own held-out validation volumes or any
arbitrary NIfTI file, and save labeled comparison images for the README.

Usage:
    # Run on your dataset's own held-out validation volumes:
    python test.py --checkpoint runs/<run>/best_model_weights.pth

    # Run on any arbitrary CT volume, with ground truth for comparison:
    python test.py --checkpoint runs/<run>/best_model_weights.pth \\
        --image_path /path/to/scan.nii.gz --label_path /path/to/mask.nii.gz

    # Run on an arbitrary volume with NO ground truth available:
    python test.py --checkpoint runs/<run>/best_model_weights.pth \\
        --image_path /path/to/scan.nii.gz
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    CropForegroundd, Orientationd, Spacingd, ToTensord,
)
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference

from data.dataset import SpleenDataset
from unetr_pp.unetr_pp_synapse import UNETR_PP
from peft_methods.lora import apply_lora_to_model
from utils import load_config

def build_model(config, checkpoint_path, device):
    method = config.peft.get('method', 'full_finetune')

    if method == "lora":
        model = UNETR_PP(in_channels=1, out_channels=2, img_size=(64, 128, 128),
                          feature_size=16, num_heads=4, depths=[3, 3, 3, 3],
                          dims=[32, 64, 128, 256], do_ds=True)
        original_ckpt = torch.load(config.paths['weights_path'], map_location="cpu", weights_only=False)
        base_state = original_ckpt["state_dict"]
        base_state = {(k[7:] if k.startswith("module.") else k): v for k, v in base_state.items()}
        model_dict = model.state_dict()
        matched = {k: v for k, v in base_state.items()
                   if k in model_dict and not k.startswith("out") and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict, strict=True)

        lcfg = config.peft['lora']
        apply_lora_to_model(model, target_attr=lcfg['target_attr'], r=lcfg['r'], alpha=lcfg['alpha'])

        trained_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(trained_state, strict=False)
        print(f"[LoRA checkpoint] loaded {len(trained_state)} keys "
              f"(lora_ params + output heads); {len(missing)} backbone keys left at "
              f"their pretrained values, as expected.")
        if unexpected:
            print(f"WARNING -- unexpected keys in checkpoint that don't exist in the model: {unexpected}")

    else:
        model = UNETR_PP(in_channels=1, out_channels=2, img_size=(64, 128, 128),
                          feature_size=16, num_heads=4, depths=[3, 3, 3, 3],
                          dims=[32, 64, 128, 256], do_ds=True)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)

    model.to(device).eval()
    return model


def get_single_volume_transforms(has_label: bool):
    keys = ["image", "label"] if has_label else ["image"]
    mode = ("bilinear", "nearest") if has_label else "bilinear"
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=mode),
        ToTensord(keys=keys),
    ])


def run_inference(model, image, roi_size, sw_batch_size, device):
    image = image.to(device)
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image, roi_size=roi_size, sw_batch_size=sw_batch_size,
            predictor=lambda x: model(x)[0], overlap=0.5,
        )
    return logits.argmax(dim=1, keepdim=True)  # (B,1,H,W,D), values 0 or 1


def dice_score(pred, gt):
    pred, gt = pred.float(), gt.float()
    inter = (pred * gt).sum().item()
    return (2 * inter) / (pred.sum().item() + gt.sum().item() + 1e-8)



def overlay(ct_slice, mask_slice, color):
    rgb = np.stack([ct_slice] * 3, axis=-1)
    alpha = 0.45
    for c in range(3):
        rgb[..., c] = np.where(mask_slice > 0, (1 - alpha) * rgb[..., c] + alpha * color[c], rgb[..., c])
    return rgb


def save_comparison_image(ct_vol, pred_vol, gt_vol, out_path, title_extra=""):
    ct_vol = ct_vol.squeeze().cpu().numpy()
    pred_vol = pred_vol.squeeze().cpu().numpy()
    gt_vol = gt_vol.squeeze().cpu().numpy() if gt_vol is not None else None

    ref_vol = gt_vol if gt_vol is not None else pred_vol
    areas = ref_vol.sum(axis=(0, 1))  # assumes (H, W, D); sum foreground per axial slice
    slice_idx = int(np.argmax(areas)) if areas.max() > 0 else ref_vol.shape[-1] // 2

    ct_slice = ct_vol[:, :, slice_idx]
    ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
    pred_slice = pred_vol[:, :, slice_idx]

    n_panels = 4 if gt_vol is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 2:
        axes = list(axes)

    axes[0].imshow(ct_norm, cmap="gray")
    axes[0].set_title("CT slice")

    axes[1].imshow(overlay(ct_norm, pred_slice, color=(0, 255, 0)))
    axes[1].set_title("Prediction")

    if gt_vol is not None:
        gt_slice = gt_vol[:, :, slice_idx]
        axes[2].imshow(overlay(ct_norm, gt_slice, color=(255, 0, 0)))
        axes[2].set_title("Ground truth")

        tp = (pred_slice > 0) & (gt_slice > 0)
        fp = (pred_slice > 0) & (gt_slice == 0)
        fn = (pred_slice == 0) & (gt_slice > 0)
        error_rgb = np.stack([ct_norm] * 3, axis=-1)
        error_rgb[tp] = [0, 1, 0]
        error_rgb[fp] = [1, 0, 0]
        error_rgb[fn] = [1, 1, 0]
        axes[3].imshow(error_rgb)
        axes[3].set_title("TP=green FP=red FN=yellow")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Slice {slice_idx}" + (f" | {title_extra}" if title_extra else ""))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a trained model checkpoint (best_model_weights.pth).")
    parser.add_argument("--image_path", type=str, required=False,
                         help="Optional: run on a specific NIfTI file instead of the dataset's own val split.")
    parser.add_argument("--label_path", type=str, required=False,
                         help="Optional ground truth for --image_path (skip if unavailable).")
    parser.add_argument("--num_samples", type=int, default=None,
                         help="Limit how many validation volumes to process (default: all).")
    parser.add_argument("--output_dir", type=str, default="test_outputs")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parent
    config_path = args.config if args.config else ROOT / "configs" / "config.yaml"
    config = load_config(str(config_path))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Method: {config.peft.get('method', 'full_finetune')}")

    model = build_model(config, args.checkpoint, device)
    roi_size = tuple(config.training['roi_size'])
    sw_batch_size = config.training['sw_batch_size']

    dice_scores = []

    if args.image_path:
        # --- Single arbitrary sample ---
        has_label = args.label_path is not None
        data = {"image": args.image_path}
        if has_label:
            data["label"] = args.label_path
        transforms = get_single_volume_transforms(has_label)
        sample = transforms(data)
        image = sample["image"].unsqueeze(0)  # add batch dim
        gt = sample["label"].unsqueeze(0) if has_label else None

        pred = run_inference(model, image, roi_size, sw_batch_size, device)

        title = ""
        if has_label:
            d = dice_score(pred, gt.to(device))
            dice_scores.append(d)
            title = f"Dice: {d:.4f}"
            print(f"{Path(args.image_path).name}: Dice = {d:.4f}")

        save_comparison_image(image, pred, gt, out_dir / f"{Path(args.image_path).stem}.png", title)

    else:
        # --- Your dataset's own held-out validation volumes ---
        spleen_dataset = SpleenDataset(data_dir=str(config.paths['data_dir']),
                                        batch_size=1, val_fraction=0.2, num_workers=4)
        _, val_loader = spleen_dataset.get_spleen_dataloaders()

        for i, val_data in enumerate(val_loader):
            if args.num_samples is not None and i >= args.num_samples:
                break
            image = val_data["image"]
            gt = val_data["label"]

            pred = run_inference(model, image, roi_size, sw_batch_size, device)
            d = dice_score(pred, gt.to(device))
            dice_scores.append(d)
            print(f"Volume {i}: Dice = {d:.4f}")

            save_comparison_image(image, pred, gt, out_dir / f"val_volume_{i}.png", f"Dice: {d:.4f}")

    if dice_scores:
        print(f"\nMean Dice across {len(dice_scores)} volume(s): {sum(dice_scores)/len(dice_scores):.4f}")
    print(f"Saved comparison images to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
"""Evaluate the saved test split or predict on a NIfTI image or directory."""
import argparse
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from utils.dice_reporting import save_dice_scores
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference

from data.dataset import get_split_test_dataloader, get_test_dataloader
from unetr_pp.unetr_pp_synapse import UNETR_PP
from peft_methods.lora import apply_lora_to_model
from utils import load_config
from utils.compute_profile import ComputeProfile

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


def run_inference(model, image, roi_size, sw_batch_size, device, spleen_class=None):
    image = image.to(device)
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image, roi_size=roi_size, sw_batch_size=sw_batch_size,
            predictor=lambda x: model(x)[0], overlap=0.5,
        ) #TODO Switch to do_ds=False in the model and remove the predictor lambda if you want to use the model's built-in sliding_window_inference.
    prediction = logits.argmax(dim=1, keepdim=True)
    return prediction if spleen_class is None else prediction == spleen_class


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


def evaluate_cases(model, samples, roi_size, sw_batch_size, device, out_dir, method,
                   split, enable_profile, spleen_class=None):
    profile = ComputeProfile(device) if enable_profile else None
    dice_scores = {}

    for sample in samples:
        image = sample["image"].to(device)
        gt = sample.get("label")
        filename = sample["filename"][0]
        if profile is not None:
            if not profile.samples:
                run_inference(model, image, roi_size, sw_batch_size, device, spleen_class)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            profile.start()
        pred = run_inference(model, image, roi_size, sw_batch_size, device, spleen_class)
        if profile is not None:
            profile.stop(filename=filename, volume_shape=list(image.shape[2:]))

        title = ""
        if gt is not None:
            d = dice_score(pred, gt.to(device))
            dice_scores[filename] = d
            title = f"Dice: {d:.4f}"
            print(f"{filename}: {title}")

        save_comparison_image(sample["image"], pred, gt, out_dir / f"{Path(filename).stem}.png", title)

        del pred, image

    save_dice_scores(out_dir, dice_scores)
    if profile is not None:
        print(profile.save(
            out_dir, "inference", method=method, split=split, precision=str(next(model.parameters()).dtype),
            roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=0.5,
        ))


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a trained model checkpoint (best_model_weights.pth).")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--split", choices=("validation", "test"), help="Manifest patients to evaluate (default: test).")
    source.add_argument("--image_path", type=Path,
                         help="A NIfTI image or directory. Defaults to the run's saved test list.")
    parser.add_argument("--label_path", type=Path,
                         help="Optional NIfTI mask or directory of masks with matching filenames.")
    parser.add_argument("--num_samples", type=int, default=None,
                         help="Limit how many images to process (default: all).")
    parser.add_argument("--profile", action="store_true", help="Measure prediction time and peak GPU memory per CT.")
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()

    if args.num_samples is not None and args.num_samples < 1:
        parser.error("--num_samples must be at least 1")
    ROOT = Path(__file__).resolve().parents[1]
    run_dir = Path(args.checkpoint).parent
    run_config = run_dir / "config.yaml"
    default_config = run_config if run_config.is_file() else ROOT / "configs/training/local.yaml"
    config = load_config(args.config or default_config)
    split = "explicit" if args.image_path else (args.split or "test")
    if args.image_path is None:
        if args.label_path is not None:
            parser.error("--label_path requires --image_path")
        split_file = run_dir / "split.json"
        if not split_file.is_file():
            if not (args.config and config.paths.get("split_file")):
                parser.error(f"Missing {split_file}. Supply --image_path or a --config with paths.split_file.")
            split_file = Path(config.paths["split_file"])
        test_loader = get_split_test_dataloader(config.paths["data_dir"], split_file, split)
    else:
        test_loader = get_test_dataloader(args.image_path, args.label_path)

    default_dir = Path("outputs/finetuned")
    out_dir = args.output_dir or (default_dir / "validation" if split == "validation" else default_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Method: {config.peft.get('method', 'full_finetune')}")

    model = build_model(config, args.checkpoint, device)
    roi_size = tuple(config.training['roi_size'])
    sw_batch_size = config.training['sw_batch_size']

    evaluate_cases(model, islice(test_loader, args.num_samples), roi_size, sw_batch_size, device,
                   out_dir, config.peft.get("method", "full_finetune"), split, args.profile)
    print(f"Saved comparison images to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

"""Evaluate the original Synapse spleen class with the project's MONAI preprocessing."""
import argparse
from pathlib import Path
import shutil

import torch

from data.dataset import get_split_test_dataloader, get_test_dataloader
from scripts.infer_finetuned import evaluate_cases
from unetr_pp.unetr_pp_synapse import UNETR_PP
from utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/heads_only.yaml"),
                        help="YAML containing the dataset, split and pretrained checkpoint paths.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--split", choices=("validation", "test"), help="Manifest patients to evaluate (default: test).")
    source.add_argument("--image_path", type=Path, help="NIfTI image or directory. Defaults to the configured test list.")
    parser.add_argument("--label_path", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Override the configured pretrained checkpoint.")
    parser.add_argument("--profile", action="store_true", help="Measure prediction time and peak GPU memory per CT.")
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()

    paths = load_config(args.config).paths
    split = "explicit" if args.image_path else (args.split or "test")
    default_dir = Path("outputs/frozen")
    out_dir = args.output_dir or (default_dir / "validation" if split == "validation" else default_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.image_path is None:
        if args.label_path is not None:
            parser.error("--label_path requires --image_path")
        split_file = Path(paths["split_file"])
        snapshot = out_dir / "split.json"
        if split_file.resolve() != snapshot.resolve():
            shutil.copyfile(split_file, snapshot)
        loader = get_split_test_dataloader(paths["data_dir"], snapshot, split)
    else:
        loader = get_test_dataloader(args.image_path, args.label_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roi = (64, 128, 128)
    model = UNETR_PP(in_channels=1, out_channels=14, img_size=roi, feature_size=16,
                     num_heads=4, depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=True)
    checkpoint = torch.load(args.checkpoint or paths["weights_path"], map_location="cpu", weights_only=False)
    model.load_state_dict({k.removeprefix("module."): v for k, v in checkpoint["state_dict"].items()},
                          strict=True)
    del checkpoint
    model.to(device).eval().requires_grad_(False)

    evaluate_cases(model, loader, roi, 4, device, out_dir, "frozen", split, args.profile, spleen_class=1)


if __name__ == "__main__":
    main()

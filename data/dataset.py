import json
from pathlib import Path
from monai.transforms import ( 
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    Spacingd, Orientationd, 
    ScaleIntensityRanged, 
    CropForegroundd, 
    RandCropByPosNegLabeld, 
    RandAffined, 
    ToTensord,
    SpatialPadd) 

from monai.data import CacheDataset, DataLoader, Dataset


def get_evaluation_transforms(has_label: bool):
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


def get_test_cases(image_path: Path, label_path: Path | None = None):
    """Load one NIfTI or a folder of NIfTIs, with labels matched by filename for folders."""
    nifti_extensions = (".nii", ".nii.gz")
    is_directory = image_path.is_dir()
    if is_directory:
        images = sorted(
            path for path in image_path.iterdir()
            if path.is_file() and not path.name.startswith(".")
            and path.name.endswith(nifti_extensions)
        )
        if not images:
            raise ValueError(f"No NIfTI images found in {image_path}")
        if label_path is not None and not label_path.is_dir():
            raise ValueError(f"Expected a label directory for a directory of images: {label_path}")
    elif image_path.is_file() and image_path.name.endswith(nifti_extensions):
        images = [image_path]
        if label_path is not None and not (label_path.is_file() and label_path.name.endswith(nifti_extensions)):
            raise ValueError(f"Expected a NIfTI label file for a single image: {label_path}")
    else:
        raise ValueError(f"Expected a NIfTI image or image directory: {image_path}")

    cases = []
    for image in images:
        case = {"image": str(image), "filename": image.name}
        if label_path is not None:
            label = label_path / image.name if is_directory else label_path
            if not label.is_file():
                raise ValueError(f"Missing label for {image.name}: {label}")
            case["label"] = str(label)
        cases.append(case)

    return cases


def get_test_dataloader(image_path: Path, label_path: Path | None = None):
    cases = get_test_cases(image_path, label_path)
    dataset = Dataset(cases, transform=get_evaluation_transforms(has_label=label_path is not None))
    return DataLoader(dataset, batch_size=1, shuffle=False)


def load_split_cases(data_dir, split_file, *groups):
    """Resolve named patient lists against the original MSD image and mask folders."""
    root = Path(data_dir)
    split = json.loads(Path(split_file).read_text())
    if not isinstance(split, dict) or set(split) != {"train", "validation", "test"}:
        raise ValueError(f"{split_file} must contain train, validation and test lists")

    seen = set()
    cases = {group: [] for group in groups}
    for group, names in split.items():
        if not isinstance(names, list) or not names:
            raise ValueError(f"Expected a nonempty patient list for {group}")
        for name in names:
            if (not isinstance(name, str) or Path(name).name != name or name.startswith(".")
                    or not name.endswith((".nii", ".nii.gz"))):
                raise ValueError(f"Expected a NIfTI filename in {group}: {name!r}")
            patient = name.removesuffix(".gz").removesuffix(".nii")
            if patient in seen:
                raise ValueError(f"Repeated patient in {split_file}: {name}")
            seen.add(patient)
            if group in cases:
                image, label = root / "imagesTr" / name, root / "labelsTr" / name
                if not image.is_file() or not label.is_file():
                    raise FileNotFoundError(f"Missing image or mask for {name} under {root}")
                cases[group].append({"image": str(image), "label": str(label), "filename": name})
    return cases


def get_split_test_dataloader(data_dir, split_file, split="test"):
    if split not in ("validation", "test"):
        raise ValueError(f"Expected a validation or test split, got {split!r}")
    cases = load_split_cases(data_dir, split_file, split)[split]
    dataset = Dataset(cases, transform=get_evaluation_transforms(has_label=True))
    return DataLoader(dataset, batch_size=1, shuffle=False)


class SpleenDataset:
    def __init__(self, data_dir, split_file, batch_size: int = 2, num_workers: int = 0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        cases = load_split_cases(data_dir, split_file, "train", "validation")
        self.train_files = cases["train"]
        self.val_files = cases["validation"]

    def get_spleen_dataloaders(self):
        """
        Builds and returns the PyTorch DataLoaders for the MSD Spleen dataset.
        
        Returns:
            train_loader, val_loader
        """

        # TRAINING TRANSFORMS
        train_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image",allow_smaller=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            SpatialPadd(keys=["image", "label"],spatial_size=(64,128,128)),

            # 3D Patch Extraction
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64,128,128),
                pos=1, neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # Spatial Augmentations to prevent overfitting
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=0.5,
                spatial_size=(64, 128, 128),
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1)
            ),
            ToTensord(keys=["image", "label"]),
        ])

        val_transforms = get_evaluation_transforms(has_label=True)

        print("Caching Training Data into RAM")
        train_ds = CacheDataset(data=self.train_files, transform=train_transforms, cache_rate=1.0, num_workers=self.num_workers)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        print("Caching Validation Data into RAM")
        val_ds = CacheDataset(data=self.val_files, transform=val_transforms, cache_rate=1.0, num_workers=self.num_workers)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=self.num_workers)

        return train_loader,val_loader


if __name__ == "__main__":
    
    # Test the SpleenDataset class by loading the dataset and printing the shape of one batch
    ROOT = Path(__file__).resolve().parents[1] 
    dataset_path = ROOT / "dataset" / "Task09_Spleen" / "Task09_Spleen"

    if dataset_path.exists():
        spleen_dataset = SpleenDataset(dataset_path, ROOT / "configs/splits/msd_spleen.json",
                                       batch_size=1, num_workers=4)
        tr_loader, vl_loader =spleen_dataset.get_spleen_dataloaders()
        
        # Grab one batch to verify shapes
        for batch in tr_loader:
            print(f"\n Train Batch Loaded!")
            print(f"Image Shape: {batch['image'].shape}")
            print(f"Label Shape: {batch['label'].shape}") 
            break
    else:
        print(f"Dataset not found at {dataset_path}. Please download it first.")

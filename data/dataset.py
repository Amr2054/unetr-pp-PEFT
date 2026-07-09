import os
import glob
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

from monai.data import CacheDataset, DataLoader

class SpleenDataset:
    def __init__(self, data_dir, batch_size: int = 2, val_fraction: float = 0.2, num_workers: int = 0):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self.train_files, self.val_files = self._prepare_data()

    def _prepare_data(self):
        images = sorted(glob.glob(os.path.join(self.data_dir, "imagesTr", "*.nii.gz")))
        labels = sorted(glob.glob(os.path.join(self.data_dir, "labelsTr", "*.nii.gz")))
        
        assert len(images) > 0, f"No images found in {self.data_dir}/imagesTr!"
        assert len(images) == len(labels), "Mismatch between number of images and labels!"

        data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]
        
        num_total = len(data_dicts)
        num_val = int(num_total * self.val_fraction)
        num_train = num_total - num_val

        train_files = data_dicts[:num_train]
        val_files = data_dicts[num_train:]
        # train_files,val_files = data_dicts[0:1],data_dicts[1:2]  # For testing purposes, using only 1 sample for train and val
        print(f"Found {num_total} total patients.")
        print(f"   -> Training on {len(train_files)} patients.")
        print(f"   -> Validating on {len(val_files)} patients.")
        
        return train_files, val_files

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

        # VALIDATION TRANSFORMS TODO : sliding_window_inference
        val_transforms = Compose([
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
            ToTensord(keys=["image", "label"]),
        ])

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

    if os.path.exists(dataset_path):
        spleen_dataset = SpleenDataset(data_dir=dataset_path, batch_size=1,val_fraction=0.2, num_workers=4)
        tr_loader, vl_loader =spleen_dataset.get_spleen_dataloaders()
        
        # Grab one batch to verify shapes
        for batch in tr_loader:
            print(f"\n Train Batch Loaded!")
            print(f"Image Shape: {batch['image'].shape}")
            print(f"Label Shape: {batch['label'].shape}") 
            break
    else:
        print(f"Dataset not found at {dataset_path}. Please download it first.")

    
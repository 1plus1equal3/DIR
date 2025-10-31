import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
import numpy as np
import glob

class Inv3DDataset(Dataset):
    def __init__(self, root_dir, training=True, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        self.scan_images = []
        self.warp_images = []
        self.other_images = {
            'flat_information_delta': [], 
            'flat_text_mask': [],
            'flat_template': [],
            'warped_recon': [],
            'warped_albedo': []
        }
        
        if not training:
            self.scan_images += glob.glob(os.path.join(self.root_dir, "val", "scan", "*.png"))
        else:
            for folder in os.listdir(self.root_dir):
                if folder != "val":
                    self.scan_images += glob.glob(os.path.join(self.root_dir, folder, "scan", "*.png"))
        self.warp_images += [path.replace("scan", "warp").replace("flat", "warped") for path in self.scan_images]

        for metadata_type in self.other_images.keys():
            self.other_images[metadata_type] += [path.replace("scan", "other").replace("flat_document", metadata_type) for path in self.scan_images]

    def __len__(self):
        return len(self.scan_images)
    
    def __getitem__(self, idx):
        scan_image_path = self.scan_images[idx]
        warp_image_path = self.warp_images[idx]
        flat_information_delta_path = self.other_images['flat_information_delta'][idx]
        flat_text_mask_path = self.other_images['flat_text_mask'][idx]
        flat_template_path = self.other_images['flat_template'][idx]
        warped_recon_path = self.other_images['warped_recon'][idx]
        warped_albedo_path = self.other_images['warped_albedo'][idx]

        scan_image = Image.open(scan_image_path).convert("RGB")
        warp_image = Image.open(warp_image_path).convert("RGB")
        flat_information_delta = Image.open(flat_information_delta_path).convert("RGB")
        flat_text_mask = Image.open(flat_text_mask_path).convert("RGB")
        flat_template = Image.open(flat_template_path).convert("RGB")
        warped_recon = Image.open(warped_recon_path).convert("RGB")
        warped_albedo = Image.open(warped_albedo_path).convert("RGB")

        if self.transform:
            scan_image = self.transform(scan_image)
            warp_image = self.transform(warp_image)
            flat_information_delta = self.transform(flat_information_delta)
            flat_text_mask = self.transform(flat_text_mask)
            flat_template = self.transform(flat_template)
            warped_recon = self.transform(warped_recon)
            warped_albedo = self.transform(warped_albedo)

        return scan_image, warp_image, flat_information_delta, flat_text_mask, flat_template, warped_recon, warped_albedo
    
class DocWarpTrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        self.scan_images = glob.glob(os.path.join(self.root_dir, "docwarp", "docwarp", "Dewarping", "*", "scan", "*.png"))
        self.scan_images += glob.glob(os.path.join(self.root_dir, "docwarp", "docwarp", "Dewarping", "*", "scan", "*.jpg"))
        self.warp_images = [path.replace("scan", "warp").replace("target", "origin") for path in self.scan_images]
    
    def __len__(self):
        return len(self.scan_images)
    
    def __getitem__(self, idx):
        scan_image_path = self.scan_images[idx]
        warp_image_path = self.warp_images[idx]
        
        scan_image = Image.open(scan_image_path).convert("RGB")
        warp_image = Image.open(warp_image_path).convert("RGB")
        
        if self.transform:
            scan_image = self.transform(scan_image)
            warp_image = self.transform(warp_image)
        
        return scan_image, warp_image

class TrainDataset(Dataset):
    def __init__(self, root_dir, chosen_data=['inv3d', 'docwarp'], transform=None, augmentations=['rotate', 'hflip', 'vflip']):
        
        self.root_dir = root_dir
        self.transform = transform
        self.augmentations = augmentations
        self.augmentation_map = {
            'rotate': self.random_rotate,
            'hflip': self.random_hflip,
            'vflip': self.random_vflip,
        }
        self.scan_images = []
        self.warp_images = []

        if 'inv3d' in chosen_data:
            inv3d_dataset = Inv3DDataset(root_dir, True, transform)
            self.scan_images += inv3d_dataset.scan_images
            self.warp_images += inv3d_dataset.warp_images

        if 'docwarp' in chosen_data:
            docwarp_dataset = DocWarpTrainDataset(root_dir, transform)
            self.scan_images += docwarp_dataset.scan_images
            self.warp_images += docwarp_dataset.warp_images
    
    def __len__(self):
        return len(self.scan_images)
    
    def random_rotate(self, scan_image, warp_image):
        angle = float(np.random.choice([0, 90, 180, 270]))
        scan_image = T.functional.rotate(scan_image, angle)
        warp_image = T.functional.rotate(warp_image, angle)
        return scan_image, warp_image

    def random_hflip(self, scan_image, warp_image):
        if np.random.rand() > 0.5:
            scan_image = T.functional.hflip(scan_image)
            warp_image = T.functional.hflip(warp_image)
        return scan_image, warp_image

    def random_vflip(self, scan_image, warp_image):
        if np.random.rand() > 0.5:
            scan_image = T.functional.vflip(scan_image)
            warp_image = T.functional.vflip(warp_image)
        return scan_image, warp_image
    
    def __getitem__(self, idx):
        scan_image_path = self.scan_images[idx]
        warp_image_path = self.warp_images[idx]
        
        scan_image = Image.open(scan_image_path).convert("RGB")
        warp_image = Image.open(warp_image_path).convert("RGB")

        for aug in self.augmentations:
            scan_image, warp_image = self.augmentation_map[aug](scan_image, warp_image)

        if self.transform:
            scan_image = self.transform(scan_image)
            warp_image = self.transform(warp_image)

        return scan_image, warp_image
    
class ValDataset(Dataset):
    def __init__(self, root_dir, chosen_data=['inv3d'], transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.scan_images = []
        self.warp_images = []

        if 'inv3d' in chosen_data:
            inv3d_dataset = Inv3DDataset(root_dir, training=False, transform=transform)
            self.scan_images += inv3d_dataset.scan_images
            self.warp_images += inv3d_dataset.warp_images
    
    def __len__(self):
        return len(self.scan_images)

    def __getitem__(self, idx):
        scan_image_path = self.scan_images[idx]
        warp_image_path = self.warp_images[idx]
        
        scan_image = Image.open(scan_image_path).convert("RGB")
        warp_image = Image.open(warp_image_path).convert("RGB")
        
        if self.transform:
            scan_image = self.transform(scan_image)
            warp_image = self.transform(warp_image)
        
        return scan_image, warp_image

# Example usage
if __name__ == "__main__":

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    
    train_data = ['inv3d', 'docwarp']
    test_data = ['inv3d']
    
    train_dataset = TrainDataset("/path/to/dataset", chosen_data=train_data, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    val_dataset = ValDataset("/path/to/dataset", chosen_data=test_data, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
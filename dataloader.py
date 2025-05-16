import torch
from torch.utils.data import DataLoader, random_split
from dataset import CustomImageDataset, SubsetWithTransform
from transforms import augment_pil, to_tensor_transform

# Paths
csv_path = "fitzpatrick17k.csv"
img_dir = "fitz_master"  # or your preferred directory

# Set random seed
seed = 11
torch.manual_seed(seed)
generator = torch.Generator().manual_seed(seed)

# Load dataset
raw_dataset = CustomImageDataset(csv_path, img_dir)

# Split sizes
total_size = len(raw_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

# Dataset splits
train_subset, val_subset, test_subset = random_split(raw_dataset, [train_size, val_size, test_size], generator=generator)

# Wrap with transform
train_dataset = SubsetWithTransform(train_subset, augment_pil=augment_pil, to_tensor=to_tensor_transform)
val_dataset   = SubsetWithTransform(val_subset, augment_pil=None, to_tensor=to_tensor_transform)
test_dataset  = SubsetWithTransform(test_subset, augment_pil=None, to_tensor=to_tensor_transform)

# Worker seed init
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, worker_init_fn=worker_init_fn)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, worker_init_fn=worker_init_fn)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, worker_init_fn=worker_init_fn)

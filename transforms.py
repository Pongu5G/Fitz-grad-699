from torchvision import transforms

# ✅ These will be updated if you re-compute mean/std from your local data
mean_dataset = [0.6082, 0.4710, 0.4226]
std_dataset  = [0.1906, 0.1618, 0.1612]

# ✅ Augmentations for training set (PIL → tensor)
augment_pil = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
])

# ✅ ToTensor + Normalize (used for all sets after optional augment)
to_tensor_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_dataset, std=std_dataset)
])
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import torch

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.target_transform = target_transform

        # ✅ Filter rows with valid Fitzpatrick scale
        self.img_labels = self.img_labels[
            self.img_labels['fitzpatrick_scale'].between(1, 6)
        ].copy()

        # ✅ Encode disease labels
        unique_diseases = sorted(self.img_labels['label'].dropna().unique())
        self.disease_to_idx = {name: idx for idx, name in enumerate(unique_diseases)}
        self.img_labels['disease_encoded'] = self.img_labels['label'].map(self.disease_to_idx)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        try:
            row = self.img_labels.iloc[idx]
            img_path = os.path.join(self.img_dir, row['md5hash'])

            image = Image.open(img_path).convert("RGB")

            # ✅ Safe Fitzpatrick parsing
            fitz_value = row['fitzpatrick_scale']
            if pd.isna(fitz_value) or int(fitz_value) not in range(1, 7):
                raise ValueError(f"Invalid Fitzpatrick value: {fitz_value} at idx {idx}")
            fitz_label = int(fitz_value) - 1

            disease_label = int(row['disease_encoded'])

            if self.target_transform:
                fitz_label = self.target_transform(fitz_label)
                disease_label = self.target_transform(disease_label)

            return image, {
                "fitzpatrick": torch.tensor(fitz_label, dtype=torch.long),
                "disease": torch.tensor(disease_label, dtype=torch.long)
            }

        except Exception as e:
            print(f"⚠️ Skipping image at idx {idx} due to error: {e}")
            return self.__getitem__((idx + 1) % len(self))


class SubsetWithTransform(Dataset):
    def __init__(self, subset, augment_pil=None, to_tensor=None):
        self.subset = subset
        self.augment_pil = augment_pil
        self.to_tensor = to_tensor

    def __getitem__(self, index):
        image, labels = self.subset[index]

        if self.augment_pil and isinstance(image, Image.Image):
            image = self.augment_pil(image)

        if self.to_tensor:
            image = self.to_tensor(image)

        return image, labels

    def __len__(self):
        return len(self.subset)
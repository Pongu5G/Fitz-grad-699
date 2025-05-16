import matplotlib.pyplot as plt
import torch

# ✅ Helper: Denormalize image tensor for plotting
def denormalize(img_tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(img_tensor.device)
    return img_tensor * std + mean

# ✅ Visualize a batch of 4 images from a DataLoader
def visualize_batch(data_loader, class_to_label, mean, std, title="Batch Samples"):
    images, labels = next(iter(data_loader))

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img = denormalize(images[i], mean, std).clamp(0, 1)
        img_np = img.permute(1, 2, 0).cpu().numpy()

        disease_idx = labels["disease"][i].item()
        fitz_val = labels["fitzpatrick"][i].item() + 1

        disease_label = class_to_label[disease_idx]
        ax.imshow(img_np)
        ax.set_title(f"{disease_label} (Type {fitz_val})", fontsize=10)
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
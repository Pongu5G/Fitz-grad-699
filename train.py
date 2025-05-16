import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from model import MultiTaskResNet
from dataloader import train_loader, val_loader
from dataloader import raw_dataset

import os
checkpoint_path = "checkpoint_latest.pth"
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print("✅ Removed old checkpoint for fresh training")

# ✅ Ensure reproducibility
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Initialize model
model_sgd = MultiTaskResNet(num_disease_classes=len(raw_dataset.disease_to_idx), num_fitz_classes=6)

# ✅ Define loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_sgd.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

# ✅ Resume support
checkpoint_path = "checkpoint_latest.pth"
start_epoch = 0
train_losses, val_losses = [], []
train_disease_acc, val_disease_acc = [], []
train_fitz_acc, val_fitz_acc = [], []

if os.path.exists(checkpoint_path):
    print("🔄 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model_sgd.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    train_disease_acc = checkpoint['train_disease_acc']
    val_disease_acc = checkpoint['val_disease_acc']
    train_fitz_acc = checkpoint['train_fitz_acc']
    val_fitz_acc = checkpoint['val_fitz_acc']
    print(f"✅ Resumed from epoch {start_epoch}")

num_epochs = 100

try:
    for epoch in range(start_epoch, num_epochs):
        print(f"\n🔁 Starting Epoch {epoch+1}/{num_epochs}")
        model_sgd.train()
        running_loss = 0.0
        correct_disease = correct_fitz = total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            disease_labels = labels["disease"].to(device)
            fitz_labels = labels["fitzpatrick"].to(device)

            optimizer.zero_grad()
            outputs = model_sgd(images)

            loss = criterion(outputs["disease"], disease_labels) + \
                   criterion(outputs["fitzpatrick"], fitz_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred_disease = torch.max(outputs["disease"], 1)
            _, pred_fitz = torch.max(outputs["fitzpatrick"], 1)
            correct_disease += (pred_disease == disease_labels).sum().item()
            correct_fitz += (pred_fitz == fitz_labels).sum().item()
            total += disease_labels.size(0)

            if batch_idx % 10 == 0:
                print(f"📦 Batch {batch_idx} | Loss: {loss.item():.4f}")

        train_losses.append(running_loss / len(train_loader))
        train_disease_acc.append(100 * correct_disease / total)
        train_fitz_acc.append(100 * correct_fitz / total)

        # Validation
        model_sgd.eval()
        val_loss = 0.0
        correct_disease = correct_fitz = total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                disease_labels = labels["disease"].to(device)
                fitz_labels = labels["fitzpatrick"].to(device)

                outputs = model_sgd(images)
                loss = criterion(outputs["disease"], disease_labels) + \
                       criterion(outputs["fitzpatrick"], fitz_labels)
                val_loss += loss.item()

                _, pred_disease = torch.max(outputs["disease"], 1)
                _, pred_fitz = torch.max(outputs["fitzpatrick"], 1)
                correct_disease += (pred_disease == disease_labels).sum().item()
                correct_fitz += (pred_fitz == fitz_labels).sum().item()
                total += disease_labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_disease_acc.append(100 * correct_disease / total)
        val_fitz_acc.append(100 * correct_fitz / total)

        print(f"✅ Epoch {epoch+1} Complete | Train Loss: {train_losses[-1]:.4f} | \
              Disease Acc: {train_disease_acc[-1]:.2f}% | Fitz Acc: {train_fitz_acc[-1]:.2f}% | \
              Val Loss: {val_losses[-1]:.4f} | Val Disease Acc: {val_disease_acc[-1]:.2f}% | Val Fitz Acc: {val_fitz_acc[-1]:.2f}%")

        scheduler.step()

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_sgd.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_disease_acc': train_disease_acc,
            'val_disease_acc': val_disease_acc,
            'train_fitz_acc': train_fitz_acc,
            'val_fitz_acc': val_fitz_acc,
        }, checkpoint_path)
        print(f"💾 Checkpoint saved at epoch {epoch}")

except KeyboardInterrupt:
    print("⏹️ Training interrupted. Saving checkpoint...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_sgd.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_disease_acc': train_disease_acc,
        'val_disease_acc': val_disease_acc,
        'train_fitz_acc': train_fitz_acc,
        'val_fitz_acc': val_fitz_acc,
    }, checkpoint_path)
    print(f"💾 Checkpoint saved at epoch {epoch} on interrupt")

# Final save
torch.save(model_sgd.state_dict(), "model_final.pth")
print("✅ Training complete. Final model saved as model_final.pth")

# ✅ Plotting
plt.figure(figsize=(16, 5))
plt.subplot(1, 3, 1)
plt.plot(train_disease_acc, label="Train Disease")
plt.plot(val_disease_acc, label="Val Disease")
plt.title("Disease Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_fitz_acc, label="Train Fitz")
plt.plot(val_fitz_acc, label="Val Fitz")
plt.title("Fitzpatrick Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

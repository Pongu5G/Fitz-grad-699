import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
from model import MultiTaskResNet
from dataloader import test_loader, raw_dataset
from tqdm import tqdm

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = MultiTaskResNet(num_disease_classes=len(raw_dataset.disease_to_idx), num_fitz_classes=6)
model.load_state_dict(torch.load("model_final.pth", map_location=device))
model.to(device)
model.eval()

# Reverse disease map
idx_to_disease = {v: k for k, v in raw_dataset.disease_to_idx.items()}

# Track predictions and labels
all_preds = []
all_labels = []
all_probs = []
fitz_types = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)

        probs = F.softmax(outputs["disease"], dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels["disease"].numpy())
        all_probs.extend(probs.cpu().numpy())
        fitz_types.extend(labels["fitzpatrick"].numpy())

# Classification report
print("\n📊 Classification Report")
print(classification_report(all_labels, all_preds, target_names=[idx_to_disease[i] for i in range(len(idx_to_disease))]))

# Confusion matrix
conf_mat = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=idx_to_disease.values(), yticklabels=idx_to_disease.values())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC curve
y_true = np.eye(len(raw_dataset.disease_to_idx))[all_labels]
y_scores = np.array(all_probs)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(raw_dataset.disease_to_idx)):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(len(raw_dataset.disease_to_idx)):
    plt.plot(fpr[i], tpr[i], label=f"{idx_to_disease[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Disease Classification)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Accuracy by Fitzpatrick Type
fitz_types = np.array(fitz_types)
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

per_type_acc = []
for i in range(6):
    mask = fitz_types == i
    acc = (all_preds[mask] == all_labels[mask]).mean() * 100
    per_type_acc.append(acc)

plt.figure(figsize=(8, 5))
plt.bar([f"Type {i+1}" for i in range(6)], per_type_acc, color="skyblue")
plt.title("Disease Classification Accuracy by Fitzpatrick Type")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.show()

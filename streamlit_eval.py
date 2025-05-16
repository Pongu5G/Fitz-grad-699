# streamlit_eval.py
import streamlit as st

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import plotly.express as px
import plotly.graph_objects as go
from dataset import CustomImageDataset
from model import MultiTaskResNet
from dataloader import test_loader, raw_dataset
import warnings
warnings.filterwarnings("default")  # ensure warnings show

# --- Load model and state dict ---
st.title("📊 Fitzpatrick17K Evaluation Dashboard")
st.write("✅ App is running...")
st.markdown("---")

@st.cache_resource
def load_model(path="model_final.pth"):
    model = MultiTaskResNet(
        num_disease_classes=len(raw_dataset.disease_to_idx),
        num_fitz_classes=6
    )
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
device = torch.device("cpu")

# --- Evaluate model ---
all_preds, all_labels, all_fitz = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        preds = torch.argmax(outputs["disease"], dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels["disease"].numpy())
        all_fitz.extend(labels["fitzpatrick"].numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_fitz = np.array(all_fitz)

label_map = {v: k for k, v in raw_dataset.disease_to_idx.items()}
class_names = [label_map[i] for i in sorted(label_map)]

# --- Classification report ---
st.subheader("📋 Classification Report")
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format("{:.2f}"))

# --- Confusion matrix ---
st.subheader("🔀 Confusion Matrix")
cm = confusion_matrix(all_labels, all_preds)
fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=class_names, y=class_names, aspect="auto",
                   title="Interactive Confusion Matrix")
fig_cm.update_layout(width=800, height=800)
st.plotly_chart(fig_cm)

# --- ROC Curve (macro average) ---
st.subheader("📈 ROC Curve (Macro Avg)")
y_true_bin = label_binarize(all_labels, classes=list(range(len(class_names))))
y_pred_bin = label_binarize(all_preds, classes=list(range(len(class_names))))
fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
roc_auc = auc(fpr, tpr)
st.markdown(f"**ROC AUC Score (macro avg):** `{roc_auc:.3f}`")

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Macro ROC AUC = {roc_auc:.2f}'))
fig_roc.update_layout(title="Macro-Average ROC Curve", xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate", width=700)
st.plotly_chart(fig_roc)

# --- Accuracy by Fitzpatrick Type ---
st.subheader("🎨 Accuracy Breakdown by Fitzpatrick Skin Type")
acc_by_fitz = []
for i in range(6):
    idxs = all_fitz == i
    correct = (all_preds[idxs] == all_labels[idxs]).sum()
    total = idxs.sum()
    acc = 100 * correct / total if total > 0 else 0
    acc_by_fitz.append(acc)

fig_bar = px.bar(x=[f"Type {i+1}" for i in range(6)], y=acc_by_fitz,
                 labels={'x': 'Fitzpatrick Type', 'y': 'Accuracy (%)'},
                 title="Validation Accuracy by Fitzpatrick Skin Type")
st.plotly_chart(fig_bar)

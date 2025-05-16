# streamlit_eval.py
import streamlit as st

import torch
import pandas as pd
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

# --- Save predictions ---
pd.DataFrame(all_preds, columns=['all_preds']).to_csv('all_preds.csv', index=False)
pd.DataFrame(all_labels, columns=['all_labels']).to_csv('all_labels.csv', index=False)
pd.DataFrame(all_fitz, columns=['all_fitz']).to_csv('all_fitz.csv', index=False)

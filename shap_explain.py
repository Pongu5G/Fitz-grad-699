import torch
import shap
import matplotlib.pyplot as plt
from model import MultiTaskResNet
from dataloader import test_loader, raw_dataset

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load trained model
model = MultiTaskResNet(
    num_disease_classes=len(raw_dataset.disease_to_idx),
    num_fitz_classes=6
).to(device)
model.load_state_dict(torch.load("model_final.pth", map_location=device))
model.eval()

# ✅ Get a small batch of images
images, labels = next(iter(test_loader))
images = images[:4].to(device)  # just use first 4 for speed

# ✅ Define output function
def model_disease_output(x):
    return model(x)["disease"]

# ✅ Create a masking-aware image masker
masker = shap.maskers.Image("inpaint_telea", images[0].shape)

# ✅ Create the SHAP explainer
explainer = shap.Explainer(model_disease_output, masker, output_names=list(raw_dataset.disease_to_idx.keys()))

# ✅ Run SHAP
shap_values = explainer(images)

# ✅ Plot results
shap.image_plot(shap_values, images.cpu())
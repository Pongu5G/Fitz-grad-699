import torch.nn as nn
from torchvision import models

class MultiTaskResNet(nn.Module):
    def __init__(self, num_disease_classes, num_fitz_classes):
        super(MultiTaskResNet, self).__init__()
        base_model = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # Remove FC
        self.fc_disease = nn.Linear(base_model.fc.in_features, num_disease_classes)
        self.fc_fitz = nn.Linear(base_model.fc.in_features, num_fitz_classes)

    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)
        return {
            "disease": self.fc_disease(x),
            "fitzpatrick": self.fc_fitz(x)
        }
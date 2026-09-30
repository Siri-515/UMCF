import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def create_resnet50(num_classes=2):

    # Load ImageNet-pretrained ResNet50
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Replace final classification layer
    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        num_classes
    )

    return model


if __name__ == "__main__":

    print("=" * 50)
    print("RESNET50 BASELINE TEST")
    print("=" * 50)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = create_resnet50()
    model = model.to(device)

    # Test input
    x = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    print("Model test successful!")
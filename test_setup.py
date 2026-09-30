import torch
import torchvision
import numpy
import cv2
import PIL
import sklearn

print("===================================")
print("UCMF-Net Environment Test")
print("===================================")

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("Pillow:", PIL.__version__)
print("Scikit-learn:", sklearn.__version__)

print("-----------------------------------")
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU not detected - currently using CPU")

print("-----------------------------------")
print("All packages imported successfully!")
print("===================================")
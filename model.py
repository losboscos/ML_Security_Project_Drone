import torch
import torch.nn as nn
from torchvision import models

def get_drone_model(device):
    print("Loading ResNet-18 (Pre-trained on ImageNet)...")
    
    # 1. Taking the original ResNet-18 model (it expects 3 RGB channels)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # --- STEP 1: modifying the Input Layer ---
    # ResNet starts with: Conv2d(3, 64, ...) -> 3 input channels (RGB)
    # We modify the input layer: Conv2d(1, 64, ...) -> 1 input channel (grayscale spectrogram)
    
    original_first_layer = model.conv1
    
    #We create a new layer with the same characteristics, but in_channels=1
    model.conv1 = nn.Conv2d(
        in_channels=1,  # <--- KEY CHANGE: 1 channel for grayscale spectrograms
        out_channels=original_first_layer.out_channels,
        kernel_size=original_first_layer.kernel_size,
        stride=original_first_layer.stride,
        padding=original_first_layer.padding,
        bias=original_first_layer.bias
    )
    
    # Adapting the original weights to the new single-channel input, without losing the trained weights.
    # We take the mean of the original weights across the RGB channels and assign it to the new single channel.
    # In this way, the network doesn't "forget" how to recognize edges and patterns.
    with torch.no_grad():
        model.conv1.weight[:] = torch.mean(original_first_layer.weight, dim=1, keepdim=True)

    # --- STEP 2: (Output Layer) ---
    #  The original ResNet gives us 1000 output features (cats, dogs, ... etc),
    #  but we only need 2 (drone vs background).
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2) # transfer learning
    model = model.to(device)
    
    print("Modified model: 1 Channel Input, 2 Channels Output.")
    return model

# --- TEST  ---
if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        try:
            import torch_directml
            DEVICE = torch_directml.device()
        except ImportError:
            DEVICE = torch.device("cpu")

    model = get_drone_model(DEVICE)
    
    # Simulating a batch of 4 grayscale images (1 channel, 224x224)
    # Batch=4, Channel=1, Height=224, Width=224
    dummy_input = torch.randn(4, 1, 224, 224).to(DEVICE)
    
    # CHECKING
    output = model(dummy_input)
    
    print("\n--- Output dimensions check ---")
    print(f"Input Shape:  {dummy_input.shape}") 
    print(f"Output Shape: {output.shape}") 
    # It should print [4, 2]
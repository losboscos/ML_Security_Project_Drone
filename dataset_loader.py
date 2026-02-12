import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# --- CONFIGURATION ---
DATA_DIR = "Spectrograms"  # Where PNGs are
BATCH_SIZE = 32              # How many images to process at once
IMG_SIZE = (224, 224)        # ResNet Standard

class DroneDataset(Dataset):
    """
    Custom Dataset to load Drone Spectrograms.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir)) # ['unknown', 'yes_drone']
        # Map class names to numbers: {'unknown': 0, 'yes_drone': 1}
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.all_files = []

        # Load all file paths
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path): continue
            
            for fname in os.listdir(class_path):
                if fname.endswith('.png'):
                    self.all_files.append((os.path.join(class_path, fname), class_name))

        # Define transformations (Convert Image -> PyTorch Tensor)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), # Ensure 1 Channel
            transforms.ToTensor(), # Converts 0-255 -> 0.0-1.0
        ])

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img_path, class_name = self.all_files[idx]
        
        # 1. Load Image
        image = Image.open(img_path)
        
        # 2. Apply Transformations
        image = self.transform(image)
        
        # 3. Get Label ID
        label = self.class_to_idx[class_name]
        
        return image, label

# --- EXECUTION & SPLITTING ---

# 1. Instantiate the Dataset
full_dataset = DroneDataset(DATA_DIR)
print(f"✅ Total Images Found: {len(full_dataset)}")
print(f"Classes: {full_dataset.class_to_idx}")

# 2. Split Train (80%) vs Test (20%)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

print(f"Train Set: {len(train_dataset)} images")
print(f"Test Set:  {len(test_dataset)} images")

# 3. Create DataLoaders (The pipeline that feeds the GPU)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- SANITY CHECK (Visual Verification) ---
# Let's grab one batch to see if the shape is correct for ResNet
images, labels = next(iter(train_loader))

print("\n--- BATCH SHAPE CHECK ---")
print(f"Images Shape: {images.shape}") 
# Expected: [32, 1, 224, 224] -> (Batch, Channel, Height, Width)
print(f"Labels Shape: {labels.shape}")
# Expected: [32]

if images.shape[1] == 1:
    print("✅ Success: Images are 1-Channel (Grayscale) as planned.")
else:
    print("❌ Error: Images are not 1-Channel.")
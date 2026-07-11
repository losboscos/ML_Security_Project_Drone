import os
import shutil
import random

# --- CONFIGURATION ---
SOURCE_DIR = "Binary_Drone_Audio"
TRAIN_DIR = "WAV_Train"
TEST_DIR = "WAV_Test"
SPLIT_RATIO = 0.8

def split_dataset():
    # Define classes based on your existing folders
    classes = ["unknown", "yes_drone"]
    
    # Create target directories if they don't exist
    for category in classes:
        os.makedirs(os.path.join(TRAIN_DIR, category), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, category), exist_ok=True)

    for category in classes:
        src_cat_dir = os.path.join(SOURCE_DIR, category)
        if not os.path.exists(src_cat_dir):
            print(f"Warning: Folder {src_cat_dir} not found. Skipping.")
            continue
            
        # Get all wav files and shuffle them
        files = [f for f in os.listdir(src_cat_dir) if f.endswith('.wav')]
        random.shuffle(files)
        
        # Calculate split index
        train_size = int(len(files) * SPLIT_RATIO)
        train_files = files[:train_size]
        test_files = files[train_size:]
        
        print(f"{category}: {len(train_files)} to Train, {len(test_files)} to Test.")
        
        # Copy files to Train folder
        for f in train_files:
            shutil.copy(os.path.join(src_cat_dir, f), os.path.join(TRAIN_DIR, category, f))
            
        # Copy files to Test folder
        for f in test_files:
            shutil.copy(os.path.join(src_cat_dir, f), os.path.join(TEST_DIR, category, f))

    print(" Split completed successfully!")

if __name__ == "__main__":
    split_dataset()
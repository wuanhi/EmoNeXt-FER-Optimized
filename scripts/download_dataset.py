import os
import shutil
import random
from kaggle.api.kaggle_api_extended import KaggleApi

os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser("~/.kaggle")
api = KaggleApi()
api.authenticate()

print("Downloading and extracting dataset 'msambare/fer2013'...")

api.dataset_download_files("msambare/fer2013", unzip=True)


dataset_dir = "FER2013"
val_dir = os.path.join(dataset_dir, "val")
test_dir = os.path.join(dataset_dir, "test")

os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


if os.path.exists("train"):
    print("Moving 'train' directory...")
    
    if os.path.exists(os.path.join(dataset_dir, "train")):
        shutil.rmtree(os.path.join(dataset_dir, "train"))
    shutil.move("train", os.path.join(dataset_dir, "train"))


if os.path.exists("test"):
    if os.path.exists("temp_test"):
        shutil.rmtree("temp_test")
    os.rename("test", "temp_test")


print("Splitting original 'test' into 'val' and 'test' (50/50)...")
classes = os.listdir("temp_test")

for cls in classes:
    cls_path = os.path.join("temp_test", cls)
    if not os.path.isdir(cls_path):
        continue
        
    images = os.listdir(cls_path)
    
    
    random.seed(42) 
    random.shuffle(images)
    
    
    split_idx = len(images) // 2
    val_images = images[:split_idx]
    test_images = images[split_idx:]
    
    
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
    
    
    for img in val_images:
        shutil.move(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))
    for img in test_images:
        shutil.move(os.path.join(cls_path, img), os.path.join(test_dir, cls, img))


if os.path.exists("temp_test"):
    shutil.rmtree("temp_test")

print(f"Done! Dataset is cleanly prepared at ./{dataset_dir}/ with 'train', 'val', and 'test' splits.")
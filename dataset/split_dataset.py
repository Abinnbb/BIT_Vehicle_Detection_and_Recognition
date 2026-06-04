import os
import random
import shutil

def split_dataset(image_dir, output_dir, test_ratio=0.1):
    os.makedirs(f"{output_dir}/train/images", exist_ok=True)
    os.makedirs(f"{output_dir}/val/images", exist_ok=True)

    image_list = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    random.shuffle(image_list)
    split_idx = int(len(image_list) * test_ratio)

    val_images = image_list[:split_idx]
    train_images = image_list[split_idx:]

    for img in train_images:
        shutil.copy(os.path.join(image_dir, img), f"{output_dir}/train/images/{img}")
    for img in val_images:
        shutil.copy(os.path.join(image_dir, img), f"{output_dir}/val/images/{img}")

    return train_images, val_images

if __name__ == "__main__":
    split_dataset("BIT-Vehicle/BITVehicle_Dataset", "../yolov5/images", test_ratio=0.1)

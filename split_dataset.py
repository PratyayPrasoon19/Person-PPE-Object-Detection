import os
import random
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_data(image_dir, label_dir, output_root, split_ratio=0.8):
    # Setup target directories
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(output_root, folder), exist_ok=True)

    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(42) # Ensure reproducibility
    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    def move_files(file_list, subset):
        for img_name in file_list:
            base_name = os.path.splitext(img_name)[0]
            lbl_name = f"{base_name}.txt"
            
            # Paths
            src_img = os.path.join(image_dir, img_name)
            src_lbl = os.path.join(label_dir, lbl_name)
            
            dst_img = os.path.join(output_root, 'images', subset, img_name)
            dst_lbl = os.path.join(output_root, 'labels', subset, lbl_name)

            if os.path.exists(src_img) and os.path.exists(src_lbl):
                shutil.copy(src_img, dst_img)
                shutil.copy(src_lbl, dst_lbl)

    move_files(train_images, 'train')
    move_files(val_images, 'val')
    logging.info(f"Splitting complete! Train: {len(train_images)}, Val: {len(val_images)}")

if __name__ == "__main__":
    # Update these paths to match your system paths
    split_data(
        image_dir="datasets/images", 
        label_dir="yolo_annotations", 
        output_root="person_model_training/person_data"
    )
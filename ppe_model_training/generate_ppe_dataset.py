import os
import cv2
import shutil
from pathlib import Path

# --- CONFIGURATION ---
ORIGINAL_IMAGES_DIR = Path("..\person_model_training\person_data\images") 
GLOBAL_LABELS_DIR = Path("..\yolo_annotations")
OUTPUT_PPE_DIR = Path("ppe_data")

# Class mapping definitions
# Original classes: 0: person, 1: hard-hat, 2: gloves, 3: mask, 4: glasses, 5: boots, 6: vest, 7: ppe-suit, 8: ear-protector, 9: safety-harness
# For the PPE model, we strip out 'person' (0) and shift all tracking indices down by 1.
PPE_CLASSES_SHIFT = {
    1: 0,  # hard-hat
    2: 1,  # gloves
    3: 2,  # mask
    4: 3,  # glasses
    5: 4,  # boots
    6: 5,  # vest
    7: 6,  # ppe-suit
    8: 7,  # ear-protector
    9: 8   # safety-harness
}

def setup_directories():
    for split in ['train', 'val']:
        (OUTPUT_PPE_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_PPE_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

def parse_yolo_file(file_path):
    boxes = []
    if not file_path.exists():
        return boxes
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:])
                boxes.append({'cls': cls_id, 'x_c': x_c, 'y_c': y_c, 'w': w, 'h': h})
    return boxes

def generate_crops():
    setup_directories()
    print("🚀 Starting coordinate extraction and cropping engine...")
    
    # Process splits based on where your original source data structures reside
    # (Assuming split structures match your tracking environment)
    for split in ['train', 'val']:
        img_split_dir = ORIGINAL_IMAGES_DIR / split
        if not img_split_dir.exists():
            # Fallback if your images aren't sub-divided into train/val folders yet
            img_split_dir = ORIGINAL_IMAGES_DIR 
            
        img_files = list(img_split_dir.glob("*.jpg")) + list(img_split_dir.glob("*.png"))
        crop_counter = 0

        for img_path in img_files:
            label_path = GLOBAL_LABELS_DIR / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
                
            all_boxes = parse_yolo_file(label_path)
            persons = [b for b in all_boxes if b['cls'] == 0]
            ppe_items = [b for b in all_boxes if b['cls'] != 0]
            
            if not persons:
                continue
                
            # Read image to compute pixel conversions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w, _ = img.shape
            
            for p_idx, p in enumerate(persons):
                # Convert normalized person box coordinates back to raw pixels
                p_w_px = p['w'] * img_w
                p_h_px = p['h'] * img_h
                p_x1 = int((p['x_c'] * img_w) - (p_w_px / 2))
                p_y1 = int((p['y_c'] * img_h) - (p_h_px / 2))
                p_x2 = int(p_x1 + p_w_px)
                p_y2 = int(p_y1 + p_h_px)
                
                # Boundary clipping defenses
                p_x1, p_y1 = max(0, p_x1), max(0, p_y1)
                p_x2, p_y2 = min(img_w, p_x2), min(img_h, p_y2)
                
                crop_w = p_x2 - p_x1
                crop_h = p_y2 - p_y1
                if crop_w <= 0 or crop_h <= 0:
                    continue
                
                # Check for any PPE elements physically inside this specific person bounding container
                local_labels = []
                for item in ppe_items:
                    # Target center point coordinates of the item in pixels
                    i_xc_px = item['x_c'] * img_w
                    i_yc_px = item['y_c'] * img_h
                    
                    if (p_x1 <= i_xc_px <= p_x2) and (p_y1 <= i_yc_px <= p_y2):
                        # Coordinate transformation math relative to person crop dimensions
                        local_xc = (i_xc_px - p_x1) / crop_w
                        local_yc = (i_yc_px - p_y1) / crop_h
                        local_w = (item['w'] * img_w) / crop_w
                        local_h = (item['h'] * img_h) / crop_h
                        
                        # Clip standard scale parameters
                        local_xc = min(1.0, max(0.0, local_xc))
                        local_yc = min(1.0, max(0.0, local_yc))
                        local_w = min(1.0, local_w)
                        local_h = min(1.0, local_h)
                        
                        shifted_cls = PPE_CLASSES_SHIFT.get(item['cls'], None)
                        if shifted_cls is not None:
                            local_labels.append(f"{shifted_cls} {local_xc:.6f} {local_yc:.6f} {local_w:.6f} {local_h:.6f}")
                
                # Save crop and label mappings if valid items are captured
                crop_name = f"{img_path.stem}_p{p_idx}"
                crop_img_path = OUTPUT_PPE_DIR / 'images' / split / f"{crop_name}.jpg"
                crop_txt_path = OUTPUT_PPE_DIR / 'labels' / split / f"{crop_name}.txt"
                
                # Execute crop and disk commit
                person_crop = img[p_y1:p_y2, p_x1:p_x2]
                cv2.imwrite(str(crop_img_path), person_crop)
                
                with open(crop_txt_path, 'w') as out_f:
                    out_f.write("\n".join(local_labels) + "\n")
                    
                crop_counter += 1
                
        print(f"✅ Extracted split '{split}' processed successfully. Generated {crop_counter} person sub-crops.")

if __name__ == "__main__":
    generate_crops()
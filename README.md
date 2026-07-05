# PPE Detection with a Two-Stage YOLOv8 Pipeline

This project implements an end-to-end computer vision solution for industrial Personal Protective Equipment (PPE) detection. The pipeline follows the assignment requirements by using two YOLOv8 models:

1. A person detector that runs on the full image.
2. A PPE detector that runs on cropped person regions to detect PPE items such as hard-hat, gloves, mask, glasses, boots, vest, ppe-suit, ear-protector, and safety-harness.

The solution converts PascalVOC annotations to YOLO format, trains both detectors, generates person-based crops for PPE training, and performs batch inference while drawing boxes and labels with OpenCV.

## Project Objective

The goal is to detect:
- people in full-scene images
- PPE items on each detected person

This approach is useful for safety monitoring workflows where PPE compliance must be checked at the person level.

## Assignment Coverage

This repository addresses the required workflow:
- Annotation conversion from PascalVOC to YOLO format
- Person detection training on full images
- PPE detection training on cropped person images
- Dual-stage inference over a folder of test images
- Custom OpenCV-based box rendering and labeling

## Repository Structure

```text
Person-PPE-Object-Detection/
├── datasets/
│   ├── classes.txt
│   ├── images/
│   └── labels/
├── person_model_training/
│   ├── person_data/
│   ├── person_data.yaml
│   ├── person_train.py
│   └── runs/
├── ppe_model_training/
│   ├── generate_ppe_dataset.py
│   ├── ppe_data/
│   ├── ppe_data.yaml
│   ├── ppe_train.py
│   └── runs/
├── inference.py
├── pascalVOC_to_yolo.py
├── split_dataset.py
├── yolo_annotations/
└── inference_outputs/
```

## Dataset and Classes

The dataset contains annotations for the following classes:
- person
- hard-hat
- gloves
- mask
- glasses
- boots
- vest
- ppe-suit
- ear-protector
- safety-harness

### Assumptions used in this implementation
- The first stage trains only on the person class.
- The second stage trains on PPE classes only, using person crops generated from the full-image annotations.
- If a person has multiple PPE items, each item is preserved in the crop-based label file.
- For each full image containing $N$ detected persons, the PPE dataset generation creates $N$ individual person crops.

## Environment Setup

Install the required dependencies:

```bash
pip install --upgrade torch torchvision ultralytics opencv-python
```

If you are using a CUDA-enabled Windows setup, the following command is also suitable:

```bash
pip install --upgrade torch torchvision ultralytics --extra-index-url https://download.pytorch.org/whl/cu121
```

## End-to-End Workflow

### Step 1: Convert PascalVOC annotations to YOLO format

```bash
python pascalVOC_to_yolo.py datasets/labels yolo_annotations
```

This reads the XML annotations from the input folder and writes YOLO-style `.txt` files into the output directory.

### Step 2: Split the dataset for person training

```bash
python split_dataset.py
```

This creates train/validation folders for the person detection model under:
- person_model_training/person_data/images/train
- person_model_training/person_data/images/val
- person_model_training/person_data/labels/train
- person_model_training/person_data/labels/val

### Step 3: Train the person detector

```bash
cd person_model_training
python person_train.py
```

This trains a YOLOv8 nano model for person detection on the full image.

### Step 4: Generate PPE crops from person detections

```bash
cd ../ppe_model_training
python generate_ppe_dataset.py
```

This script:
- loads the YOLO-format labels
- finds person boxes
- crops each person region from the full image
- converts PPE labels into crop-relative coordinates
- saves the corresponding cropped images and labels for PPE training

### Step 5: Train the PPE detector

```bash
python ppe_train.py
```

This trains a second YOLOv8 nano model on the cropped person images for PPE classification.

### Step 6: Run inference on a folder of images

```bash
cd ..
python inference.py datasets\images inference_outputs person_model_training\runs\detect\syook_workspace\person_detector-4\weights\best.pt ppe_model_training\runs\detect\syook_workspace\ppe_detector-2\weights\best.pt
```

This script:
- loads the person detector
- detects people in each image
- crops those person regions
- runs the PPE detector on the crops
- maps the PPE predictions back to the original image coordinates
- saves the annotated outputs in the output directory

## Model Training Configuration

### Person detector
- Base model: YOLOv8n
- Input size: 640
- Epochs: 50
- Batch size: 16
- Device: CPU by default (can be changed to GPU if available)

### PPE detector
- Base model: YOLOv8n
- Input size: 320
- Epochs: 50
- Batch size: 16
- Device: CPU by default (can be changed to GPU if available)

## Output Artifacts

After running the workflow, the repository will contain:
- converted YOLO labels in yolo_annotations/
- person training splits in person_model_training/person_data/
- PPE training crops in ppe_model_training/ppe_data/
- training logs and weights in the respective runs folders
- annotated inference images in inference_outputs/

## Notes

- The inference script uses OpenCV for drawing boxes and placing text labels, as required by the assignment.
- The project is designed to be robust to multiple people in a single image by producing one crop per person.
- The current implementation uses the provided repository structure and relative paths, so the commands above should be run from the project root unless otherwise noted.

## Expected Result

The final pipeline produces annotated images that show:
- detected persons in the full frame
- PPE detections mapped back to the original image coordinates
- confidence values rendered next to each bounding box

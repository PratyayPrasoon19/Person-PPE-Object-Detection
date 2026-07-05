# Step 1: Annotation Format Conversion (pascalVOC_to_yolo.py)
## sample: 
python pascalVOC_to_yolo.py /path/to/input_dataset_dir /path/to/output_yolo_annotations_dir
## current command:
python pascalVOC_to_yolo.py datasets/labels yolo_annotations

# Step 2: Split Dataset for Person Detector
## command:
python split_dataset.py

# Step 3: Stage-1 Model Training (Person Detector)
## command:
cd person_model_training
python person_train.py

# Step 4: Sub-Bounding Box Generation & Dataset Cropping
## command
cd ppe_model_training
python generate_ppe_dataset.py

# Step 5: Stage-2 Model Training (PPE Detector)
## command
cd ppe_model_training
python ppe_train.py

# Step 6: Unified Batch Inference (inference.py)
## sample: 
python inference.py --input_dir ./test_images --output_dir ./annotated_results --person_det_model ./weights/person_best.pt --ppe_detection_model ./weights/ppe_best.pt
## current command
python inference.py datasets\images inference_outputs person_model_training\runs\detect\syook_workspace\person_detector-4\weights\best.pt ppe_model_training\runs\detect\syook_workspace\ppe_detector-2\weights\best.pt --classes_file datasets\classes.txt

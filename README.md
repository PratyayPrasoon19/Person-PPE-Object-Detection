# for running pascalVOC_to_yolo.py:

python pascalVOC_to_yolo.py datasets/labels yolo_annotations --classes_file datasets/classes.txt

# for running inference.py

python inference.py datasets\images inference_outputs yolov8n.pt yolov8n.pt --classes_file datasets\classes.txt

python inference.py datasets\images inference_outputs person_model_training\runs\detect\syook_workspace\person_detector-4\weights\best.pt ppe_model_training\runs\detect\syook_workspace\ppe_detector-2\weights\best.pt --classes_file datasets\classes.txt

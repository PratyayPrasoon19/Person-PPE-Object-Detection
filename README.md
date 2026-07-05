# for running pascalVOC_to_yolo.py:

python pascalVOC_to_yolo.py datasets/labels yolo_annotations --classes_file datasets/classes.txt

# for running inference.py

python inference.py datasets\images inference_outputs yolov8n.pt yolov8n.pt --classes_file datasets\classes.txt

python inference.py datasets\images inference_outputs .\weights\person_detect.pt .\weights\ppe_detect.pt --classes_file datasets\classes.txt
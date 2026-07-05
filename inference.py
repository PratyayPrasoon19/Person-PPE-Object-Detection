import os
import cv2
import argparse
import logging
from ultralytics import YOLO

# Set up logging for tracking pipeline performance and troubleshooting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TwoStageInferencePipeline:
    def __init__(self, person_model_path, ppe_model_path):
        logging.info("Initializing models...")
        self.person_model = YOLO(person_model_path)
        self.ppe_model = YOLO(ppe_model_path)
        
        # Color mapping palette for custom OpenCV rendering (BGR format)
        self.colors = {
            'person': (255, 0, 0),        # Blue
            'hard-hat': (0, 255, 255),    # Yellow
            'boots': (255, 0, 165),       # Pink/Purple
            'vest': (0, 165, 255),        # Orange
            'ppe-suit': (0, 255, 0),      # Green
            'gloves': (255, 255, 0),      # Cyan
            'default': (0, 0, 255)        # Red for everything else
        }

    def get_ppe_class_name(self, class_id):
        # Extract class names dynamically directly from the trained model dictionary
        if class_id in self.ppe_model.names:
            return self.ppe_model.names[class_id]
        return f"class_{class_id}"

    def process_image(self, img_path, output_dir, conf_threshold=0.1):
        # Read the raw full image
        frame = cv2.imread(img_path)
        if frame is None:
            logging.error(f"Could not read image: {img_path}")
            return

        # Keep a completely clean copy for cropping to prevent drawing artifacts from leaking into other crops
        crop_canvas = frame.copy()
        render_canvas = frame.copy()

        # STAGE 1: Detect Persons on the full image
        person_results = self.person_model(frame, conf=conf_threshold, verbose=False)[0]
        
        # Parse detected persons
        for person_box in person_results.boxes:
            # Check if class matches person (assuming person index is 0 in default YOLO models)
            if int(person_box.cls[0]) != 0:
                continue

            # Extract absolute coordinates of the person bounding box
            p_xyxy = person_box.xyxy[0].cpu().numpy()
            p_xmin, p_ymin, p_xmax, p_ymax = map(int, p_xyxy)
            p_conf = float(person_box.conf[0])

            # Draw the Person Bounding Box directly using OpenCV on the render canvas
            cv2.rectangle(render_canvas, (p_xmin, p_ymin), (p_xmax, p_ymax), self.colors['person'], 2)
            label = f"person {p_conf:.2f}"
            cv2.putText(render_canvas, label, (p_xmin, max(p_ymin - 10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['person'], 2)

            # Extract image crop safely ensuring coordinates remain within frame limits from clean canvas
            h, w, _ = crop_canvas.shape
            p_xmin_safe, p_ymin_safe = max(0, p_xmin), max(0, p_ymin)
            p_xmax_safe, p_ymax_safe = min(w, p_xmax), min(h, p_ymax)
            
            person_crop = crop_canvas[p_ymin_safe:p_ymax_safe, p_xmin_safe:p_xmax_safe]
            if person_crop.size == 0:
                continue

            # STAGE 2: Infer PPE items inside the cropped person window
            ppe_results = self.ppe_model(person_crop, conf=conf_threshold, verbose=False)[0]

            # Parse detected PPE items within the crop
            for ppe_box in ppe_results.boxes:
                ppe_cls_id = int(ppe_box.cls[0])
                ppe_cls_name = self.get_ppe_class_name(ppe_cls_id)
                ppe_conf = float(ppe_box.conf[0])

                # Bounding box dimensions relative to the crop
                c_xyxy = ppe_box.xyxy[0].cpu().numpy()
                c_xmin, c_ymin, c_xmax, c_ymax = map(int, c_xyxy)

                # INVERSE COORDINATE MAPPING: Map relative crop boxes back to full canvas coordinates
                global_ppe_xmin = p_xmin_safe + c_xmin
                global_ppe_ymin = p_ymin_safe + c_ymin
                global_ppe_xmax = p_xmin_safe + c_xmax
                global_ppe_ymax = p_ymin_safe + c_ymax

                # Choose specific rendering color based on item class
                color = self.colors.get(ppe_cls_name, self.colors['default'])

                # Draw the mapped PPE box onto the main rendering canvas using OpenCV
                cv2.rectangle(render_canvas, (global_ppe_xmin, global_ppe_ymin), 
                              (global_ppe_xmax, global_ppe_ymax), color, 2)
                
                ppe_label = f"{ppe_cls_name} {ppe_conf:.2f}"
                cv2.putText(render_canvas, ppe_label, (global_ppe_xmin, max(global_ppe_ymin - 10, 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Save the finalized canvas with all rendered layers
        out_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, render_canvas)

    def run_directory_pipeline(self, input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
        
        logging.info(f"Processing directory: Found {len(images)} images to process.")
        
        for idx, img_name in enumerate(images):
            img_path = os.path.join(input_dir, img_name)
            try:
                self.process_image(img_path, output_dir)
                if (idx + 1) % 50 == 0 or (idx + 1) == len(images):
                    logging.info(f"Progress: [{idx + 1}/{len(images)}] images successfully processed.")
            except Exception as e:
                logging.error(f"Bypassed processing on image {img_name} due to unexpected runtime error: {e}")

if __name__ == "__main__":
    # Explicitly matches requirements from Submission Guidelines (Question 4 and 5)
    parser = argparse.ArgumentParser(description="Two-Stage Vision Pipeline: Person & PPE Detection Engine.")
    parser.add_argument("input_dir", type=str, help="Directory path targeting the raw test images.")
    parser.add_argument("output_dir", type=str, help="Directory path where annotated frames will save.")
    parser.add_argument("person_det_model", type=str, help="File path pointing to the Person Detection Model weights (.pt)")
    parser.add_argument("ppe_detection_model", type=str, help="File path pointing to the PPE Detection Model weights (.pt)")

    args = parser.parse_args()

    try:
        pipeline = TwoStageInferencePipeline(
            person_model_path=args.person_det_model,
            ppe_model_path=args.ppe_detection_model
        )
        pipeline.run_directory_pipeline(args.input_dir, args.output_dir)
        logging.info("Two-Stage Pipeline batch processing executed completely.")
    except Exception as err:
        logging.critical(f"Pipeline crashed during initialization: {err}")
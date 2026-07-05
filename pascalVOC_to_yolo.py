import os
import xml.etree.ElementTree as ET
import argparse
import logging

# Set up logging to match industrial standards preferred by the JD
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_classes(classes_file):
    with open(classes_file, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def convert_box(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x_center * dw, y_center * dh, w * dw, h * dh)

def convert_xml_to_yolo(input_dir, output_dir, classes):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
    logging.info(f"Found {len(xml_files)} XML files to process.")

    for xml_file in xml_files:
        file_path = os.path.join(input_dir, xml_file)
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            size_elem = root.find('size')
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
            
            out_txt_path = os.path.join(output_dir, xml_file.replace('.xml', '.txt'))
            
            with open(out_txt_path, 'w') as out_file:
                for obj in root.iter('object'):
                    cls_name = obj.find('name').text
                    if cls_name not in classes:
                        continue
                    cls_id = classes.index(cls_name)
                    
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                         float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    
                    bb = convert_box((width, height), b)
                    out_file.write(f"{cls_id} " + " ".join([f"{x:.6f}" for x in bb]) + '\n')
                    
        except Exception as e:
            logging.error(f"Failed to process file {xml_file} due to error: {e}")
            raise e # Fail-fast as per assignment rules to ensure data validity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PascalVOC XML annotations to YOLO text format.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing XML files.")
    parser.add_argument("output_dir", type=str, help="Path to save the converted YOLO .txt files.")
    
    args = parser.parse_args()
    
    try:
        class_list = load_classes("datasets/classes.txt")
        convert_xml_to_yolo(args.input_dir, args.output_dir, class_list)
        logging.info("Successfully converted all annotations!")
    except Exception as err:
        logging.critical(f"Data conversion pipeline aborted: {err}")
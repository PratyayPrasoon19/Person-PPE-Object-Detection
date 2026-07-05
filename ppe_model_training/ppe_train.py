import logging
from ultralytics import YOLO
import torch.utils._pytree

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Monkey-patch the missing attribute for PyTorch compatibility
if not hasattr(torch.utils._pytree, 'register_pytree_node'):
    if hasattr(torch.utils._pytree, '_register_pytree_node'):
        torch.utils._pytree.register_pytree_node = torch.utils._pytree._register_pytree_node

def run_training():
    logging.info("Loading YOLOv8 nano model weights...")
    model = YOLO('yolov8n.pt') 

    logging.info("Starting PPE Detection training...")
    model.train(
        data='ppe_data.yaml',
        epochs=50,
        imgsz=320,          # Optimized lower resolution for tight person crops
        batch=16,
        device='cpu',       # Use 0 if CUDA GPU is available, else 'cpu'
        project='syook_workspace',
        name='ppe_detector'
    )
    logging.info("Training pipeline successfully finished!")

if __name__ == "__main__":
    run_training()
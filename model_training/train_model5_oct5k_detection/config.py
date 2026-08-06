import os

DATASET_ROOT = os.getenv("OCT5K_DETECTION_DIR", "/Users/nikhilmundhra/Downloads/Capstone/DataSets/Segmented/OCT5K_Object_Detection")
CHECKPOINT_DIR = "./checkpoints/model5_detection"

# Hyperparameters
NUM_CLASSES = 10  # 9 Pathology classes + 1 Background
EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

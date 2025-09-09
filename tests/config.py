"""
Central configuration for the test suite.
"""
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_CONFIG_PATH = 'models/detect/hyper/model.yaml'
NUM_CLASSES = 80
REG_MAX = 16

IMG_SIZE = 640

TEST_LOG_DIR = 'runs/tests'

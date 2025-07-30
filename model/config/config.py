import torch

# -- Project Configuration --
PROJECT_NAME = "DeltaDepthEstimation"
PROJECT_VERSION = "0.1.0"

# -- Device Configuration --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -- Dataset Configuration --
# Path to the directory containing the training images
TRAIN_IMG_DIR = 'data/raw' 
# Path to the directory containing the validation images
VAL_IMG_DIR = 'data/raw'

# Data Split ratio  (training/(training+validation))
DATA_SPLIT  = 0.7
IMG_HEIGHT = 384
IMG_WIDTH = 384


# -- Model Configuration --
# Teacher Model (Hugging Face identifier or local path)
TEACHER_MODEL_ID = 'depth-anything/depth-anything-v2-small-hf'

# Student Model Configuration
# Encoder to use for the student model (e.g., 'mobilevit_xs')
STUDENT_ENCODER = 'mobilevit_xs'
# Indices of the feature maps to extract from the encoder
STUDENT_FEATURE_INDICES = [0, 1, 2, 3]
# Channel dimensions for the MiniDPT decoder blocks
STUDENT_DECODER_CHANNELS = [256, 128, 96, 64]


# -- Training Configuration --
# Number of training epochs
EPOCHS = 60
# Base learning rate for the encoder (lower than decoder's, fine  tuning)
LEARNING_RATE_ENCODER = 1e-5
# Base learning rate for the decoder
LEARNING_RATE_DECODER = 1e-3
# Weight decay for the optimizer
WEIGHT_DECAY = 1e-3
# Min learning rate for the schedular
MIN_LEARNING_RATE = 1e-6
# Batch size for training and validation
BATCH_SIZE = 8
# Number of workers for dataset loader
NUM_WORKERS = 2

# -- Loss Function Weights --
# Weight for the Scale-Invariant Log (SILog) loss
LAMBDA_SILOG = 1.0
# Weight for the Gradient Matching loss
LAMBDA_GRAD = 1.0
# Weight for the Feature Matching (distillation) loss
LAMBDA_FEAT = 0.5
# Weight for the Attention Matching (distillation) loss
LAMBDA_ATTN = 0.5
ALPHA = 0.5


# -- Data Transformation --
FLIP_PROP = 0.5
ROTATION_DEG = 10
MIN_SCALE = 0.8
MAX_SCALE = 10
BRIGHTNESS = 0.4
CONTRAST = 0.4
SATURATION = 0.4
HUE = 0.1
IMGNET_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMGNET_NORMALIZE_STD = [0.229, 0.224, 0.225]

# -- Checkpoint and Seeds --
# Directory to save model checkpoints
CHECKPOINT_DIR = 'checkpoints'
# Random seed for reproducibility
RANDOM_SEED = 42

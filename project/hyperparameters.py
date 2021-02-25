import torch
# 250 bird species
# DATA_DIR = 'D:/datasets/bird_species'
# IMGMEAN = [0.507, 0.487, 0.441]
# IMGSTD = [0.267, 0.256, 0.276]
# DS_SPLIT = [33000, 4715]
# NUM_CLASSES = 250

# Imagenet
DATA_DIR = 'D:/datasets/imagenet'
IMGMEAN = [0.485, 0.456, 0.406]
IMGSTD = [0.229, 0.224, 0.225]
DS_SPLIT = [1000000, 281167]
NUM_CLASSES = 1000

IMAGE_SIZE = 224
CHANNELS = 3


BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 128
N_EPOCHS = 50
N_WORKERS = 2
PREFETCH_FACTOR = 2


LEARNING_RATE = 2e-4
GAMMA = 0.7
EPSILON = 2e-7
WEIGHT_DECAY = 0.0001
LR_DECAY_PER = 2
DECAY_TYPE = 'step'  # or 'cosine'
COS_T0 = 6
COS_T_MULT = 3

EARLY_STOP_PATIENCE = 5
EARLY_STOP_DELTA = 0.001
CHECKPOINT_PATH = 'checkpoints'
CHECKPOINT_MAX_HIST = 5

TRANSFORMER_HEADS = 8
TRANSFORMER_DEPTH = 8
VTR_MLP_DIM = 1536
VTR_DIM = 768

VTR_DROPOUT = 0.15
ATTN_DROPOUT = 0.1
EMB_DROPOUT = 0.1

opt_level = 'O1'

# one of resnet 50-101-152, resnext 50-101, resnest 50-101-200-269

net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest200', pretrained=True, verbose=False)
# BACKBONE = models.resnext101_32x8d(pretrained=True)
BACKBONE = net
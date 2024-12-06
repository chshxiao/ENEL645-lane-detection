import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet_model2 import UNET
from dice_loss import DiceLoss
from tversky_loss import TverskyLoss
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


# hyperparameter
LEARNING_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCH = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 540
IMAGE_WIDTH = 960
PIN_MEMORY = True
LOAD_MODEL = True
ALPHA = 0.1                             # Tversky loss - False positive
BETA = 0.9                              # Tversky loss - False negative
TRAIN_IMG_DIR = 'train_folder/'
TRAIN_MASK_DIR = 'train_masks/'
VAL_IMG_DIR = 'val_folder/'
VAL_MASK_DIR = 'val_masks/'
TEST_IMG_DIR = 'test_folder/'
TEST_MASK_DIR = 'test_masks/'
PATH = 'check_point.pth.tar'
SAVE_FOLDER = 'saved_images_91/'


train_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

test_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)


model = UNET(in_channels=1, out_channels=1).to(DEVICE)
# model = UNET(in_channels=1, out_channels=1).to(DEVICE)                # greyscale
criterion = TverskyLoss(alpha=ALPHA, beta=BETA)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_loader, val_loader, test_loader = get_loaders(
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    TEST_IMG_DIR,
    TEST_MASK_DIR,
    BATCH_SIZE,
    train_transforms,
    val_transforms,
    test_transforms,
    NUM_WORKERS,
    PIN_MEMORY,
)

checkpoint = torch.load(PATH)

load_checkpoint(checkpoint, model)
test_loss = check_accuracy(test_loader, model, criterion, device=DEVICE)
save_predictions_as_imgs(val_loader, model, folder=SAVE_FOLDER, device=DEVICE)
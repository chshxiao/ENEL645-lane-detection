import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
from torchinfo import summary
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
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
# LEARNING_RATE = 0.0001
LEARNING_RATE = 0.001                               # with exponential scheduler
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCH = 20
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
PATH = 'check_point_exponential.pth.tar'
# PATH = 'check_poing_greyscale.pth.tar'                          # greyscale
# PATH = 'check_poing_grey_schedule.pth.tar'                        # greyscale + scheduler
SAVE_FOLDER = 'saved_images_exponential/'


def train(loader, model, optimizer, loss, scaler):
    model.train()
    pixel_acc = 0
    num_pixel = 0
    intersection = 0
    label_sum = 0
    dice_loss = 0
  
    # loop = tqdm(loader)

    # for batch_index, (data, targets) in enumerate(loop):
    for batch_index, (data, targets) in enumerate(loader):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data).squeeze()
            loss_value = loss(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss_value).backward()
        scaler.step(optimizer)
        scaler.update()

        # get accuracy
        # predictions = torch.sigmoid(predictions)                      # sigmoid before loss
        # predictions = (predictions > 0.5).float()
        predictions = (predictions > 0.7).float()                       # increase threshold

        pixel_acc += (predictions == targets).sum().item()
        num_pixel += torch.numel(predictions)

        # get dice loss component
        intersection += (predictions * targets).sum()
        label_sum += predictions.sum() + targets.sum()
        dice_loss += loss_value

        # # update tqdm loop
        # loop.set_postfix(loss=loss_value.item())

    print("train:")
    print(f"Accuracy: {pixel_acc}/{num_pixel} {pixel_acc/num_pixel*100}%")
    print(f"Dice Intersection: {intersection}")
    print(f"Dice Sum: {label_sum}")
    print(f"Dice score: {2. * intersection / label_sum}")
    print(f"Dice Loss: {dice_loss/len(loader)}")
    

train_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.AdvancedBlur(),
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

# model = UNET(in_channels=3, out_channels=1).to(DEVICE)
model = UNET(in_channels=1, out_channels=1).to(DEVICE);                     # greyscale
# loss = nn.BCEWithLogitsLoss()
# criterion = DiceLoss()
criterion = TverskyLoss(alpha=ALPHA, beta=BETA)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

summary(model, input_size=(BATCH_SIZE, 1, IMAGE_HEIGHT, IMAGE_WIDTH))

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

scaler = torch.cuda.amp.GradScaler()
scheduler = ExponentialLR(optimizer, gamma=0.7)             # exponential scheduler
best_loss = 1e+10
for epoch in range(NUM_EPOCH):
  train(train_loader, model, optimizer, criterion, scaler)
  scheduler.step()                                          # with exponential scheduler

  # check accuracy
  loss_this_epoch, correct_this_epoch = check_accuracy(val_loader, model, criterion, device=DEVICE)

  if correct_this_epoch < 1000:
    break

  if loss_this_epoch < best_loss:
    # save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, filename=PATH)


# print some examples to a folder
save_predictions_as_imgs(test_loader, model, folder=SAVE_FOLDER, device=DEVICE)
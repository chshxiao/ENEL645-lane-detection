import os
import torch
import torchvision
from dataset2 import LaneDataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_check_point.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoing")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    test_dir,
    test_maskdir,
    batch_size,
    train_transforms,
    val_transforms,
    test_transforms,
    num_workers=4,
    pin_memory=True,
):
    train_ds = LaneDataset(image_dir=train_dir, mask_dir=train_maskdir, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = LaneDataset(image_dir=val_dir, mask_dir=val_maskdir, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    test_ds = LaneDataset(image_dir=test_dir, mask_dir=test_maskdir, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


def check_accuracy(loader, model, criterion, device="cuda"):
    num_correct = 0
    num_pixels = 0
    intersection = 0
    label_sum = 0
    dice_loss = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            predictions = model(x).squeeze()
            loss_value = criterion(predictions, y)

            dice_loss += loss_value.item()

            # get accuracy
            predictions = torch.sigmoid(predictions)
            # predictions = (predictions > 0.5).float()
            predictions = (predictions > 0.7).float()               # increase threshold
            num_correct += (predictions == y).sum().item()
            num_pixels += torch.numel(predictions)

            # get dice score component
            intersection += (predictions * y).sum()
            label_sum += predictions.sum() + y.sum()
    
    # update dice loss
    dice_loss = dice_loss / len(loader)

    print("Evaluate:")
    print(f"{num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}%")
    print(f"Dice Intersection: {intersection}")
    print(f"Dice Sum: {label_sum}")
    print(f"Dice Score: {2. * intersection / label_sum}")
    print(f"Dice Loss: {dice_loss}")
    model.train()

    return dice_loss, num_correct


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    # create a new directory if it doesn't exist
    if not os.path.exists(folder):
        os.mkdir(folder)

    model.eval()
    for idx, (x, y) in enumerate(loader):
        if idx < 2:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                # preds = (preds > 0.5).float()
                preds = (preds > 0.7).float()                   # increase threshold

            torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
            torchvision.utils.save_image(x, f"{folder}/y_{idx}.png")
        else:
            break

    model.train()
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        :param smooth: prevent division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        :param inputs: predicted segmentation map, shape (B, C, H, W)
        :param targets: ground truth segmentation map, shape (B, H, W) or (B, C, H, W)
        :return: dice_loss: scalar value of the Dice Loss
        """
        # # if inputs are logits, apply softmax
        if inputs.dim() > 2 and inputs.shape[1] > 1:
            inputs = torch.softmax(inputs, dim=1)

        # flatten the tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        # compute the dice score
        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / \
                     (inputs.sum() + targets.sum() + self.smooth)

        # compute dice loss
        dice_loss = 1 - dice_score
        return dice_loss
import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
  def __init__(self, alpha, beta, smooth = 1e-6):
    super().__init__()
    self.alpha = alpha
    self.beta = beta
    self.smooth = smooth
  
  def forward(self, inputs, targets):
    # # if inputs are logits, apply softmax
    if inputs.dim() > 2 and inputs.shape[1] > 1:
      inputs = torch.softmax(inputs, dim=1)
    
    # flatten the tensors
    inputs = inputs.contiguous().view(-1)
    targets = targets.contiguous().view(-1)

    #True Positives, False Positives & False Negatives
    TP = (inputs * targets).sum()    
    FP = ((1-targets) * inputs).sum()
    FN = (targets * (1-inputs)).sum()

    # compute Tversky loss
    Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

    return 1-Tversky
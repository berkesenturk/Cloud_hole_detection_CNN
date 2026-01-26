import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        
        inputs = F.softmax(inputs, dim=1) 

        targets = F.one_hot(targets, num_classes=self.num_classes).float() 

        log_p = torch.log(inputs + 1e-8)  

        ce_loss = -targets * log_p  

        p_t = torch.sum(inputs * targets, dim=1, keepdim=True)  

        focal_weight = (1 - p_t) ** self.gamma  

        loss = focal_weight * ce_loss 
        loss = self.alpha * loss  

        return loss.sum(dim=1).mean()

def train_model(model, train_loader, criterion, optimizer, device):
    """
    Description:
        Train the model using the given training data.
    Parameters:
        model: ResNetForImageClassification
            The model to train.
        train_loader: DataLoader
            The training data loader.
        criterion: nn.Module
            The loss function.
        optimizer: optim.Optimizer
            The optimizer to use for training.
    Returns:
        float: The average training loss.
        float: The training accuracy
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        _, predicted = torch.max(outputs.logits, 1)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_accuracy = 100 * correct / total
    
    return train_loss, train_accuracy

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs.logits, labels).item()
            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    
    return val_loss, val_accuracy

def early_stopping(val_loss, best_val_loss, counter, patience):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered")
    return best_val_loss, counter
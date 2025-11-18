"""Defines the convolutional neural network architecture used for benign versus
malignant histopathology classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BreastCancerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolution blocks with BatchNorm
        #BatchNorm:It is a technique that normalizes the output of a layer in a neural network in a mini-batch.
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1); self.bn4 = nn.BatchNorm2d(256)
        
        #Max-pooling to reduce spatial size
        #pooling:It is a process used in CNCs that reduces the size of the image while preserving important information.
        self.pool = nn.MaxPool2d(2,2)
        #Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        #Fully connected layers after flattening (128→64→32→16→8)
        self.fc1 = nn.Linear(256*8*8, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        #Four conv → BN → ReLU → pool stages
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten feature maps
        x = x.view(-1, 256*8*8)
        
        # Apply dropout and dense layers
        x = self.dropout(F.relu(self.fc1(x)))
        #Final logits (no softmax here, handled by loss function)
        #Softmax, is the activation function that gives the probability distribution for each class.
        return self.fc2(x)

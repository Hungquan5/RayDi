import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        # Expecting a 2D input with 1 channel (e.g., Mel-Spectrogram)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # Adjust according to the flattened size
        self.fc2 = nn.Linear(128, 2)  # Two output classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the feature map
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

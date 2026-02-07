import torch.nn as nn
from transformers import ResNetForImageClassification


class CNN_Model(nn.Module):

    def __init__(self, number_of_classes):
        super(CNN_Model, self).__init__()
        self.conv_layer1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=5
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.1)
        self.relu1 = nn.ReLU()

        self.linear1 = nn.Linear(32 * 110 * 110, 256)
        self.linear2 = nn.Linear(256, number_of_classes)

    def forward(self, x):
        out = self.conv_layer1(x)  # 1
        out = self.relu1(out)  # 2
        out = self.max_pool1(out)  # 3
        out = self.dropout1(out)  # 4

        # Flatten the output before passing to linear layers
        out = out.view(out.size(0), -1)  # Flatten all dimensions except batch
        out = self.linear1(out)  # 5
        out = self.relu1(out)  # 6
        out = self.linear2(out)  # 7

        return out


def get_pretrained(**kwargs) -> ResNetForImageClassification:

    return ResNetForImageClassification.from_pretrained(
        **kwargs
    )

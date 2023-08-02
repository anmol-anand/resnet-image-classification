import torch
from torch.functional import Tensor
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self,
            resnet_size,
            first_num_filters,
            num_classes,
            device
        ):
        super(ResNet, self).__init__()
        self.resnet_size = resnet_size
        self.first_num_filters = first_num_filters
        self.num_classes = num_classes

        self.start_layer = nn.Sequential(
            nn.Conv2d(3, first_num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=first_num_filters, eps=1e-5, momentum=0.997),
            nn.LeakyReLU(0.01)
            ).to(device)

        self.stack_layers = nn.ModuleList()
        for i in range(3):
            filters = self.first_num_filters * (2**i)
            stride = 1 if i == 0 else 2
            self.stack_layers.append(stack_layer(filters, stride, self.resnet_size, self.first_num_filters, device))

        self.output_layer = output_layer(filters, self.num_classes, device)

    def forward(self, inputs):
        outputs = self.start_layer(inputs)
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs

class standard_block(nn.Module):
    def __init__(self, out_channels, stride, in_channels, device) -> None:
        super(standard_block, self).__init__()
        if in_channels != out_channels:
            assert out_channels == 2 * in_channels
            assert stride == 2
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0).to(device),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.997)
                ).to(device)
        else :
            # In our model, when there is no change in #channels, the spatial dimension doesn't change either.
            assert stride == 1
            self.shortcut = nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1).to(device)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.997).to(device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1).to(device)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.997).to(device)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs
        outputs = self.conv1(outputs)
        outputs = self.bn1(outputs)
        outputs = nn.LeakyReLU(0.01)(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = outputs + self.shortcut(inputs)
        outputs = nn.LeakyReLU(0.01)(outputs)
        return outputs

class stack_layer(nn.Module):
    def __init__(self, filters, stride, resnet_size, first_num_filters, device) -> None:
        super(stack_layer, self).__init__()
        self.resnet_size = resnet_size
        out_channels = filters
        if filters == first_num_filters:
            in_channels = out_channels
        else:
            in_channels = out_channels // 2

        self.blocks = nn.ModuleList()
        self.blocks.append(standard_block(out_channels, stride, in_channels, device))
        for _ in range(1, resnet_size):
            self.blocks.append(standard_block(out_channels, 1, out_channels, device))
    
    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs
        for i in range(0, self.resnet_size):
            outputs = self.blocks[i](outputs)
        return outputs

class output_layer(nn.Module):
    def __init__(self, filters, num_classes, device) -> None:
        super(output_layer, self).__init__()
        in_channels = filters

        pooling_stride = 2
        self.avgPool = nn.AvgPool2d(kernel_size=pooling_stride, stride=pooling_stride, padding=0)
        self.fc_input_size = in_channels * ((8 // pooling_stride)**2)
        self.fc1 = nn.Linear(self.fc_input_size, self.fc_input_size // 2).to(device)
        self.fc2 = nn.Linear(self.fc_input_size // 2, num_classes).to(device)
        
    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs
        outputs = self.avgPool(outputs)
        outputs = outputs.view(outputs.shape[0], self.fc_input_size)
        outputs = self.fc1(outputs)
        outputs = nn.Sigmoid()(outputs)
        outputs = self.fc2(outputs)
        return outputs
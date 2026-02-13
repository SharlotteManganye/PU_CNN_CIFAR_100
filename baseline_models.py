from model_components import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class baseline_model_1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,        
        kernel_size,
        stride,
        dropout_rate,
        number_classes,
        fc_dropout_rate
    ):
        super(baseline_model_1, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout
        self.dropout_conv = nn.Dropout(dropout_rate)
        self.fc_dropout_rate = fc_dropout_rate  # save for forward

        # Flatten
        self.flatten = nn.Flatten()

        # Fully connected layers (will init later based on input size)
        self.fc1 = None
        self.fc2 = nn.Linear(out_channels * 2, number_classes)

    def _initialize_fc_layers(self, x):
        in_features = x.size(1)
        self.fc1 = nn.Linear(in_features, 256)  # hidden layer with more capacity
        self.fc2 = nn.Linear(256, self.fc2.out_features)
        self.fc1.to(x.device)
        self.fc2.to(x.device)

    def forward(self, x):
        # First conv block
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Flatten
        x = self.flatten(x)

        # Initialize FC layers once
        if self.fc1 is None:
            self._initialize_fc_layers(x)

        # Fully connected head
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.fc_dropout_rate, training=self.training)
        x = self.fc2(x)

        return x


class baseline_model_2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dropout_rate,
        image_height,
        image_width,
        fc_hidden_size,
        number_classes,
        fc_dropout_rate,
    ):
        super(baseline_model_2, self).__init__()

        self.conv_layers1 = StandardConv2D( in_channels, out_channels, kernel_size, stride, padding, dropout_rate)
        self.conv_layers2 = StandardConv2D(out_channels, out_channels*2, kernel_size, stride, padding, dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, image_height, image_width)
            output_shape = self.conv_layers2(self.conv_layers1(dummy_input)).shape
 
        fc_input_size = output_shape[1]  

        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x):
        x = self.conv_layers1(x)
        x = self.conv_layers2(x)
        x  = self.global_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)

class baseline_model_3(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,        
        kernel_size,
        stride,
        dropout_rate,
        number_classes,
        fc_dropout_rate
    ):
        super(baseline_model_3, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size, stride, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout
        self.dropout_conv = nn.Dropout(dropout_rate)
        self.fc_dropout_rate = fc_dropout_rate  # save for forward

        # Flatten
        self.flatten = nn.Flatten()

        # Fully connected layers (will init later based on input size)
        self.fc1 = None
        self.fc2 = nn.Linear(out_channels * 2, number_classes)

    def _initialize_fc_layers(self, x):
        """Initialize FC layers dynamically based on flattened size."""
        in_features = x.size(1)
        self.fc1 = nn.Linear(in_features, 256)  # hidden layer with more capacity
        self.fc2 = nn.Linear(256, self.fc2.out_features)
        self.fc1.to(x.device)
        self.fc2.to(x.device)

    def forward(self, x):
        # First conv block
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Second conv block
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Flatten
        x = self.flatten(x)

        # Initialize FC layers once
        if self.fc1 is None:
            self._initialize_fc_layers(x)

        # Fully connected head
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.fc_dropout_rate, training=self.training)
        x = self.fc2(x)

        return x

class baseline_model_4(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,        
        kernel_size,
        stride,
        dropout_rate,
        number_classes,
        fc_dropout_rate
    ):
        super(baseline_model_4, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size, stride, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout
        self.dropout_conv = nn.Dropout(dropout_rate)
        self.fc_dropout_rate = fc_dropout_rate  # save for forward

        # Flatten
        self.flatten = nn.Flatten()

        # Fully connected layers (will init later based on input size)
        self.fc1 = None
        self.fc2 = nn.Linear(out_channels * 2, number_classes)

    def _initialize_fc_layers(self, x):
        """Initialize FC layers dynamically based on flattened size."""
        in_features = x.size(1)
        self.fc1 = nn.Linear(in_features, 256)  # hidden layer with more capacity
        self.fc2 = nn.Linear(256, self.fc2.out_features)
        self.fc1.to(x.device)
        self.fc2.to(x.device)

    def forward(self, x):
        # First conv block
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Second conv block
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Flatten
        x = self.flatten(x)

        # Initialize FC layers once
        if self.fc1 is None:
            self._initialize_fc_layers(x)

        # Fully connected head
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.fc_dropout_rate, training=self.training)
        x = self.fc2(x)

        return x


# class baseline_model_4(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride,
#         padding,
#         dropout_rate,
#         image_height,
#         image_width,
#         fc_hidden_size,
#         number_classes,
#         fc_dropout_rate,
#         num_layers,
#     ):
#         super(baseline_model_4, self).__init__()
#         self.num_layers = num_layers
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.conv_layers = StandardConv2D(in_channels, out_channels*2, kernel_size, stride, padding, dropout_rate)

#         with torch.no_grad():
#           dummy_input = torch.randn(1, in_channels, image_height, image_width)
#           output_shape = self.conv_layers(dummy_input).shape
          
#         fc_input_size = output_shape[1]  

#         self.mlp = MLP(
#             fc_input_size=fc_input_size,
#             fc_hidden_size=fc_hidden_size,
#             num_classes=number_classes,
#             fc_dropout_rate=fc_dropout_rate,
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         pooled  = self.global_pool(x)
#         x_flat = pooled.reshape(pooled.size(0), -1)
#         out = self.mlp(x_flat)
#         return F.log_softmax(out, dim=1)

class baseline_model_5(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dropout_rate,
        image_height,
        image_width,
        fc_hidden_size,
        number_classes,
        fc_dropout_rate,
        num_layers,
    ):
        super(baseline_model_5, self).__init__()

        self.num_layers = num_layers
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        layers = []
        current_in_channels = in_channels
        for _ in range(num_layers):
            layers.append(StandardConv2D(current_in_channels, out_channels, kernel_size, stride, padding, dropout_rate))
            current_in_channels = out_channels  

        self.conv_layers = nn.Sequential(*layers)

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, image_height, image_width)
            output_shape = self.conv_layers(dummy_input).shape

        fc_input_size = output_shape[1]  

        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x):
        x = self.conv_layers(x)
        pooled  = self.global_pool(x)
        x_flat = pooled.reshape(pooled.size(0), -1)
        out = self.mlp(x_flat)
        return F.log_softmax(out, dim=1)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def baseline_model_6():
    return ResNet18(num_classes=100)




class ResNet8(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet8, self).__init__()
        self.in_channels = 128
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer3 = self._make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def baseline_model_7():
    return ResNet8(num_classes=100)    
# 
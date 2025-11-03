from model_components import *
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim



class model_1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_rate, number_classes, fc_dropout_rate):
        super(model_1, self).__init__()

        self.prod_block = nn.Sequential(
            ProductUnits(in_channels, out_channels,kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )

        self.flatten = nn.Flatten()
        self.fc1 = None 
        self.fc2 = None  
        self.fc_dropout_rate = fc_dropout_rate
        self.number_classes = number_classes

    def forward(self, x, return_feature_maps=False):
        x = self.prod_block(x)
        x_flat = self.flatten(x)

        if self.fc1 is None:
            in_features = x_flat.size(1)
            self.fc1 = nn.Linear(in_features, 256).to(x.device)
            self.fc2 = nn.Linear(256, self.number_classes).to(x.device)

        output = F.relu(self.fc1(x_flat))
        output = F.dropout(output, p=self.fc_dropout_rate, training=self.training)
        output = self.fc2(output)

        if return_feature_maps:
            return output, x
        return output

        


class model_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_rate, number_classes, fc_dropout_rate):
        super(model_2, self).__init__()

        self.prod_block = nn.Sequential(
            ProductUnits(in_channels, out_channels,kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )

        self.prod_block2 = nn.Sequential(
            ProductUnits( out_channels, out_channels*2,kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )


        self.flatten = nn.Flatten()
        self.fc1 = None 
        self.fc2 = None 
        self.fc_dropout_rate = fc_dropout_rate
        self.number_classes = number_classes

    def forward(self, x, return_feature_maps=False):
        prod1 = self.prod_block(x)  
        prod2 = self.prod_block2 (prod1)
        x_flat = self.flatten(x)

        if self.fc1 is None:
            in_features = x_flat.size(1)
            self.fc1 = nn.Linear(in_features, 256).to(x.device)
            self.fc2 = nn.Linear(256, self.number_classes).to(x.device)

        output = F.relu(self.fc1(x_flat))
        output = F.dropout(output, p=self.fc_dropout_rate, training=self.training)
        output = self.fc2(output)

        if return_feature_maps:
            return output, prod1,prod2
        return output
      


class model_3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_rate, number_classes, fc_dropout_rate,image_height, image_width):
        super(model_3, self).__init__()

        self.prod_block = nn.Sequential(
            ProductUnits(in_channels, out_channels,kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dropout_rate = fc_dropout_rate

        # Temporary dummy input to infer feature size
        dummy_input = torch.zeros(1, in_channels,image_height, image_width)  # change to your real input size
        with torch.no_grad():
            conv_out = self.prod_block(dummy_input)
            conv_out = self.conv_block(conv_out)
            conv_out = torch.flatten(conv_out, 1)
            flattened_size = conv_out.shape[1]

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, number_classes)

    def forward(self, x, return_feature_maps=False):
        prod1 = self.prod_block(x)
        conv1 = self.conv_block(prod1)
        x_flat = torch.flatten(conv1, 1)

        output = F.relu(self.fc1(x_flat))
        output = F.dropout(output, p=self.fc_dropout_rate, training=self.training)
        output = self.fc2(output)

        if return_feature_maps:
            return output, prod1, conv1
        return output



class model_0(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_rate, number_classes, fc_dropout_rate,image_height, image_width):
        super(model_0, self).__init__()

        self.prod_block = nn.Sequential(
            ProductUnits(out_channels, out_channels * 2,kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels , kernel_size,stride=stride, padding=1),
            nn.BatchNorm2d(out_channels ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dropout_rate = fc_dropout_rate

        # Temporary dummy input to infer feature size
        dummy_input = torch.zeros(1, in_channels, image_height, image_width)  # change to your real input size
        with torch.no_grad():
            conv_out = self.conv_block(dummy_input)
            conv_out = self.prod_block(conv_out)
            conv_out = torch.flatten(conv_out, 1)
            flattened_size = conv_out.shape[1]

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, number_classes)

    def forward(self, x, return_feature_maps=False):
        conv1 = self.conv_block(x)
        prod1 = self.prod_block(conv1)
        x_flat = torch.flatten(prod1, 1)

        output = F.relu(self.fc1(x_flat))
        output = F.dropout(output, p=self.fc_dropout_rate, training=self.training)
        output = self.fc2(output)

        if return_feature_maps:
            return output, conv1, prod1
        return output





class model_4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_rate, number_classes, fc_dropout_rate, image_height, image_width):
        super(model_4, self).__init__()

        self.prod_block = nn.Sequential(
            ProductUnits(in_channels, out_channels, kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )

        self.fc_dropout_rate = fc_dropout_rate

        # Infer feature size using dummy input
        dummy_input = torch.zeros(1, in_channels, image_height, image_width)
        with torch.no_grad():
            prod_out = self.prod_block(dummy_input)
            conv_out = self.conv_block(dummy_input)

            if prod_out.shape[2:] != conv_out.shape[2:]:
                conv_out = F.interpolate(conv_out, size=prod_out.shape[2:], mode="bilinear", align_corners=False)

            concat = torch.cat((prod_out, conv_out), dim=1)
            flattened_size = torch.flatten(concat, 1).shape[1]

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, number_classes)

    def forward(self, x, return_feature_maps=False):
        prod1 = self.prod_block(x)
        conv1 = self.conv_block(x)

        if prod1.shape[2:] != conv1.shape[2:]:
            conv1 = F.interpolate(conv1, size=prod1.shape[2:], mode="bilinear", align_corners=False)

        concat = torch.cat((prod1, conv1), dim=1)
        x_flat = torch.flatten(concat, 1)

        output = F.relu(self.fc1(x_flat))
        output = F.dropout(output, p=self.fc_dropout_rate, training=self.training)
        output = self.fc2(output)

        if return_feature_maps:
            return output, prod1, conv1
        return output


class model_5(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_rate, number_classes, fc_dropout_rate, image_height, image_width):
        super(model_5, self).__init__()

        self.prod_block = nn.Sequential(
            ProductUnits(in_channels, 32, kernel_size, stride=stride),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )

        self.prod_block2 = nn.Sequential(
            ProductUnits(64, 64, kernel_size, stride=stride),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )

        self.fc_dropout_rate = fc_dropout_rate

        # Infer feature size using dummy input
        dummy_input = torch.zeros(1, in_channels, image_height, image_width)
        with torch.no_grad():
            prod_out = self.prod_block(dummy_input)
            conv_out = self.conv_block(dummy_input)

            if prod_out.shape[2:] != conv_out.shape[2:]:
                conv_out = F.interpolate(conv_out, size=prod_out.shape[2:], mode="bilinear", align_corners=False)

            concat = torch.cat((prod_out, conv_out), dim=1)

            prod_out2 = self.prod_block2(concat)
            conv_out2 = self.conv_block2(concat)

            if prod_out2.shape[2:] != conv_out2.shape[2:]:
                conv_out2 = F.interpolate(conv_out2, size=prod_out2.shape[2:], mode="bilinear", align_corners=False)

            concat2 = torch.cat((prod_out2, conv_out2), dim=1)

            flattened_size = torch.flatten(concat2, 1).shape[1]

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, number_classes)

    def forward(self, x, return_feature_maps=False):
        prod1 = self.prod_block(x)
        conv1 = self.conv_block(x)

        if prod1.shape[2:] != conv1.shape[2:]:
          conv1 = F.interpolate(conv1, size=prod1.shape[2:], mode="bilinear", align_corners=False)

        concat = torch.cat((prod1, conv1), dim=1)

        # Use prod_block2 and conv_block2 here on concat, NOT on x
        prod2 = self.prod_block2(concat)
        conv2 = self.conv_block2(concat)

        if prod2.shape[2:] != conv2.shape[2:]:
            conv2 = F.interpolate(conv2, size=prod2.shape[2:], mode="bilinear", align_corners=False)

        concat2 = torch.cat((prod2, conv2), dim=1)

        x_flat = torch.flatten(concat2, 1)

        output = F.relu(self.fc1(x_flat))
        output = F.dropout(output, p=self.fc_dropout_rate, training=self.training)
        output = self.fc2(output)

        if return_feature_maps:
            return output, prod2, conv2
        return output

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 128
        self.pi_conv_layers = ProductUnits(3, self.inchannel,2)
        self.bn_prod = nn.BatchNorm2d(self.inchannel)
        self.dropout = nn.Dropout(0.25)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)        
        self.fc = nn.Linear(512, num_classes)
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.pi_conv_layers(x)
        out = self.bn_prod(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = self.dropout(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
def PI_ResNet():
    return ResNet(BasicBlock)

class ResNet2(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10, input_size=(3, 32, 32)):
        super(ResNet2, self).__init__()
        self.inchannel = 3
        self.pi_conv_layers = ProductUnits(3, 128, 2)
        self.bn_prod = nn.BatchNorm2d(128)  
        self.dropout = nn.Dropout(0.25)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)  

        # compute fc input size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_size)  # batch=1, channels=3, HxW = 32x32
            out = self._forward_features(dummy)
            flattened_size = out.view(1, -1).size(1)

        self.fc = nn.Linear(flattened_size, num_classes)
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def _forward_features(self, x):
        y = self.pi_conv_layers(x)
        y = self.bn_prod(y)
        y = F.relu(y)

        z = self.layer3(x)
        z = self.layer4(z)

        if y.shape[2:] != z.shape[2:]:
            z = F.interpolate(z, size=y.shape[2:], mode="bilinear", align_corners=False)

        concat = torch.cat((y, z), dim=1)
        out = F.avg_pool2d(concat, 4)
        return out
    
    def forward(self, x):
        out = self._forward_features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

def PI_ResNet_stacked(num_classes=10, input_size=(3, 32, 32)):
    return ResNet2(BasicBlock, num_classes=num_classes, input_size=input_size)





class PU_ResNet18(nn.Module):
    def __init__(self, ResidualBlock,ResidualBlock_PU, num_classes=10):
        super(PU_ResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            ProductUnits2(3, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock_PU, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)        
        self.fc = nn.Linear(512, num_classes)
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def PU_ResNet_18():
    return PU_ResNet18(ResidualBlock,ResidualBlock_PU)

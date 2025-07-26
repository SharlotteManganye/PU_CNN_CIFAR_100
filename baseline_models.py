from model_components import *
import torch.nn.functional as F
import torch


class baseline_model_1(nn.Module):
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
        super(baseline_model_1, self).__init__()

        self.conv_layers = StandardConv2D(in_channels, out_channels, kernel_size, stride, padding, dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        with torch.no_grad():
          dummy_input = torch.randn(1, in_channels, image_height, image_width)
          x = self.conv_layers(dummy_input)
          x = self.global_pool(x) 
          fc_input_size = x.view(1, -1).size(1) 
      
        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x):
      x = self.conv_layers(x)
      x  = self.global_pool(x)
      x = x.reshape(x.size(0), -1)
      x = self.mlp(x)
      return F.log_softmax(x, dim=1)


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
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

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
        padding,
        dropout_rate,
        image_height,
        image_width,
        # fc_input_size,
        fc_hidden_size,
        number_classes,
        fc_dropout_rate,
    ):
        super(baseline_model_3, self).__init__()

        self.conv_layers1 = StandardConv2D( in_channels, out_channels, kernel_size, stride, padding, dropout_rate )
        self.conv_layers2 = StandardConv2D(out_channels, out_channels * 2, kernel_size, stride, padding, dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))


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

class baseline_model_4(nn.Module):
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
        super(baseline_model_4, self).__init__()
        self.num_layers = num_layers
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_layers = StandardConv2D(in_channels, out_channels*2, kernel_size, stride, padding, dropout_rate)

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

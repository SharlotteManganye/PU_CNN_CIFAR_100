from model_components import *
import torch.nn.functional as F
import torch


class baseline_model_1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        image_height,
        image_width,
        # fc_input_size,
        fc_hidden_size,
        number_classes,
        fc_dropout_rate,
    ):
        super(baseline_model_1, self).__init__()

        self.conv_layers = StandardConv2D(
            in_channels, out_channels
        )
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, image_height, image_width)
            output_shape = self.conv_layers(dummy_input).shape

        fc_input_size = out_channels * output_shape[2] * output_shape[3]
        self.bn_conv = nn.BatchNorm2d(out_channels)

        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.bn_conv(x)
        x = F.relu(x)
      # x = torch.log(torch.clamp(x, min=1e-6))
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)


class baseline_model_2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        image_height,
        image_width,
        fc_hidden_size,
        number_classes,
        fc_dropout_rate,
    ):
        super(baseline_model_2, self).__init__()

        self.conv_layers1 = StandardConv2D(
            in_channels, out_channels
        )

        self.conv_layers2 = StandardConv2D(
            in_channels, out_channels*2
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, image_height, image_width)
            output_shape = self.conv_layers2(self.conv_layers1(dummy_input)).shape

        fc_input_size = out_channels * output_shape[2] * output_shape[3]
        self.bn_conv = nn.BatchNorm2d(out_channels)

        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x):
        x = self.conv_layers1(x)
        x = self.bn_conv(x)
        x = F.relu(x)
        x = self.conv_layers2(x)
        x = self.bn_conv(x)
        x = F.relu(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x


class baseline_model_3(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        image_height,
        image_width,
        fc_hidden_size,
        number_classes,
        fc_dropout_rate,
    ):
        super(baseline_model_3, self).__init__()

        self.conv_layers1 = StandardConv2D(
            out_channels, out_channels
        )

        self.conv_layers2 = StandardConv2D(
            in_channels, out_channels*2
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, image_height, image_width)
            output_shape = self.conv_layers2(self.pi_conv_layers1(dummy_input)).shape

        fc_input_size = out_channels * output_shape[2] * output_shape[3]
        self.bn_conv = nn.BatchNorm2d(out_channels)

        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x):
        x = self.conv_layers1(x)
        x = self.bn_conv(x)
        x = F.relu(x)
        x = self.conv_layers2(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x


class baseline_model_4(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        image_height,
        image_width,
        fc_input_size,
        fc_hidden_size,
        number_classes,
        fc_dropout_rate,
        num_layers,
    ):
        super(baseline_model_4, self).__init__()

        self.num_layers = num_layers

        self.concat_conv_ = StandardConv2D(
            in_channels, out_channels
        )

        self.conv_layers = StandardConv2D(
            out_channels, out_channels
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, image_height, image_width)
            output_shape = self.conv_layers(self.concat_conv_(dummy_input)).shape

        fc_input_size = out_channels * output_shape[2] * output_shape[3]

        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x):
        x = self.concat_conv_(x)
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x


class baseline_model_5(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        image_height,
        image_width,
        fc_input_size,
        fc_hidden_size,
        number_classes,
        fc_dropout_rate,
        num_layers,
    ):
        super(baseline_model_5, self).__init__()

        self.num_layers = num_layers

        self.concat_conv_ = StandardConv2D(
            in_channels, out_channels
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, image_height, image_width)
            output_shape = self.concat_conv_(dummy_input).shape

        fc_input_size = output_shape[1] * output_shape[2] * output_shape[3]

        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x):
        x = self.concat_conv_(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x

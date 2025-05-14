from model_components import *


class model_1(nn.Module):
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
        super(model_1, self).__init__()

        self.pi_conv_layers = ProductUnits(in_channels, out_channels)

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, image_height, image_width)
            output_shape = self.pi_conv_layers(dummy_input).shape

        fc_input_size = out_channels * output_shape[2] * output_shape[3]

        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x):
      x = self.pi_conv_layers(x)
      print(f"Shape after pi_conv_layers: {x.shape}")
      x = x.reshape(x.size(0), -1)
      print(f"Shape after reshape: {x.shape}")
      x = self.mlp(x)
      return x


class model_2(nn.Module):
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
    ):
        super(model_2, self).__init__()

        self.pi_conv_layers1 = ProductUnits(
            in_channels, out_channels
        )

        self.pi_conv_layers2 = ProductUnits(
            out_channels, out_channels
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, image_height, image_width)
            output_shape = self.pi_conv_layers2(self.pi_conv_layers1(dummy_input)).shape

        fc_input_size = out_channels * output_shape[2] * output_shape[3]

        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x):
        x = self.pi_conv_layers1(x)
        x = self.pi_conv_layers2(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x


class model_3(nn.Module):
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
    ):
        super(model_3, self).__init__()

        self.conv_layers1 = StandardConv2D(
            in_channels, out_channels
        )

        self.pi_conv_layers2 = ProductUnits(
            out_channels, out_channels
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, image_height, image_width)
            output_shape = self.pi_conv_layers2(self.conv_layers1(dummy_input)).shape

        fc_input_size = out_channels * output_shape[2] * output_shape[3]
        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x):
        x = self.conv_layers1(x)
        x = self.pi_conv_layers2(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x

class model_4(nn.Module):
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
        super(model_4, self).__init__()

        self.concat_conv_product = ConcatConv2DProductUnits(
            in_channels=in_channels,
            num_layers=num_layers,
            initial_out_channels=out_channels,
        )

        output_channels_concat = out_channels * (2 ** (num_layers - 1)) * 2

        self.conv_layers = StandardConv2D(
            output_channels_concat, out_channels
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, image_height, image_width)
            x = self.concat_conv_product(dummy_input)
            output_shape = self.conv_layers(x).shape  # Calculate shape AFTER conv_layers

        fc_input_size = out_channels * output_shape[2] * output_shape[3]

        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x):
        x = self.concat_conv_product(x)
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x


class model_5(nn.Module):
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
        super(model_5, self).__init__()

        self.num_layers = num_layers

        self.concat_conv_product = ConcatConv2DProductUnits(
            in_channels=in_channels,
            num_layers=num_layers,
            initial_out_channels=out_channels,
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, image_height, image_width)
            output_shape = self.concat_conv_product(dummy_input).shape

        fc_input_size = output_shape[1] * output_shape[2] * output_shape[3]
        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x):
        x = self.concat_conv_product(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x
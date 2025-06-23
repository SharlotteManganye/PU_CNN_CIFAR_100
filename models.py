from model_components import *
import torch.nn.functional as F


class model_1(nn.Module):
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
        super(model_1, self).__init__()

        self.pi_conv_layers = ProductUnits(in_channels, out_channels)
        self.bn_prod = nn.BatchNorm2d(out_channels)
        

        with torch.no_grad():
          dummy_input = torch.randn(1, in_channels, image_height, image_width)
          x = self.pi_conv_layers(dummy_input)
          x = self.bn_prod(x)
          x = F.relu(x)
          x = F.max_pool2d(x, 2)  # <-- include this
          fc_input_size = x.view(1, -1).size(1)  # <-- correct flattened size
      
        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x, return_feature_maps=False):
      x = self.pi_conv_layers(x)
      x = self.bn_prod(x)
      x = F.relu(x)
      x = F.max_pool2d(x, 2)
      # x = torch.log(torch.clamp(x, min=1e-6))
      x_flat  = x.reshape(x.size(0), -1)
      x_out  = self.mlp(x_flat)
      output = F.log_softmax(x_out, dim=1)
      if return_feature_maps:
        return output, x
      else:
        return output



class model_2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        image_height,
        image_width,
        dropout_rate,
        fc_hidden_size,
        number_classes,
        fc_dropout_rate,
    ):
        super(model_2, self).__init__()

        self.pi_conv_layers1 = ProductUnits(
            in_channels, out_channels
        )

        self.pi_conv_layers2 = ProductUnits(
            out_channels, out_channels*2
        )
        self.dropout = nn.Dropout2d(dropout_rate)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels * 2)

        with torch.no_grad():
          dummy_input = torch.randn(1, in_channels, image_height, image_width)
          x = self.pi_conv_layers1(dummy_input)
          x = self.bn1(x)
          x = F.relu(x)
          x = F.max_pool2d(x, 2)
          x = self.pi_conv_layers2(x)
          x = self.bn2(x)
          x = F.relu(x)
          x = F.max_pool2d(x, 2)
          x = self.dropout(x) 
          fc_input_size = x.view(1, -1).size(1) 

        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x,return_feature_maps=False):
      pi_cov = self.pi_conv_layers1(x)
      pi_cov = self.bn1(pi_cov)
      pi_cov = F.relu(pi_cov)
      pi_cov = F.max_pool2d(pi_cov, 2)
      pi_cov2 = self.pi_conv_layers2(pi_cov)
      pi_cov2 = self.bn2(pi_cov2)
      pi_cov2 = F.relu(pi_cov2)
      pi_cov2 = F.max_pool2d(pi_cov2, 2)
      pi_cov2 = self.dropout(pi_cov2) 
      x_flat = x.reshape(pi_cov2.size(0), -1)
      x_out = self.mlp(x_flat)
      output = F.log_softmax(x_out, dim=1)
      if return_feature_maps:
        return output, pi_cov,pi_cov2
      else:
        return output
      


class model_3(nn.Module):
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
        super(model_3, self).__init__()

        self.conv_layers1 = StandardConv2D(
            in_channels, out_channels
        )
        self.pi_conv_layers2 = ProductUnits(out_channels, out_channels * 2)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.bn = nn.BatchNorm2d(out_channels*2)

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, image_height, image_width) 
            x = self.conv_layers1(dummy_input)
            x = self.pi_conv_layers2(x)
            x = self.bn(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout(x) 
            fc_input_size = x.view(1, -1).size(1)    
        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x):
        x = self.conv_layers1(x)
        x = self.pi_conv_layers2(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x) 
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)

class model_0(nn.Module):
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
        super(model_0, self).__init__()

          
        self.pi_conv_layers1 = ProductUnits(in_channels, out_channels )

        self.conv_layers1 = StandardConv2D(
             out_channels, out_channels*2
        )

        
      

        self.dropout = nn.Dropout2d(dropout_rate)
        self.bn = nn.BatchNorm2d(out_channels)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, image_height, image_width) 
            x = self.pi_conv_layers1(dummy_input)
            x = self.bn(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.conv_layers1(x)
            fc_input_size = x.view(1, -1).size(1) 
      
        self.mlp = MLP(
            fc_input_size=fc_input_size,
            fc_hidden_size=fc_hidden_size,
            num_classes=number_classes,
            fc_dropout_rate=fc_dropout_rate,
        )

    def forward(self, x):
        x = self.pi_conv_layers1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv_layers1(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)

class model_4(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        image_height,
        image_width,
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


        with torch.no_grad():
          dummy_input = torch.randn(1, in_channels, image_height, image_width)
          x = self.concat_conv_product(dummy_input)
          output_shape = x.shape  # <- fixed line


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
        return F.log_softmax(x, dim=1)


class model_5(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        image_height,
        image_width,
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
        return F.log_softmax(x, dim=1)
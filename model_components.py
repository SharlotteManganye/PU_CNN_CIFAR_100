import torch
import torch.nn as nn
import torch.nn.functional as F
import math

epsilon = 1e-10
kernel_size = 3
stride = 1
padding = 1
dropout_rate = 0.25
fc_dropout_rate=0.25
num_layers = 1


def set_component_vars(eps, k_size, strd, pad, drop):
    global epsilon, kernel_size, stride, padding, dropout_rate
    epsilon = eps
    kernel_size = k_size
    stride = strd
    padding = pad
    dropout_rate = drop


class MLP(nn.Module):
    def __init__(self, fc_input_size, fc_hidden_size, num_classes, fc_dropout_rate):
        super().__init__()
        self.fc_input_size = fc_input_size
        self.fc_hidden_size = fc_hidden_size
        self.num_classes = num_classes
        self.fc_dropout_rate = fc_dropout_rate

        self.fc_model = nn.Sequential(
            nn.Linear(self.fc_input_size, self.fc_hidden_size),
            nn.BatchNorm1d(self.fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.fc_dropout_rate),
            nn.Linear(self.fc_hidden_size, self.num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc_model(x)



class StandardConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_rate):
        super().__init__()
        kernel_size = int(kernel_size)
        stride = int(stride)
        padding = int(padding)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation_func(x)
        x = self.dropout(x)
        return x


# class ProductUnits(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ProductUnits, self).__init__()
#         self.weights_u = nn.Parameter(
#             torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
#         )  # for positive inputs
#         self.weights_v = nn.Parameter(
#             torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
#         )  # for indicator function
#         self.stride = stride
#         self.padding = padding  # Store the padding value
#         # Initialize weights (important!)
#         nn.init.kaiming_uniform_(self.weights_u, a=math.sqrt(5))
#         nn.init.kaiming_uniform_(self.weights_v, a=math.sqrt(5))

#     def forward(self, x):
#         # Apply padding before unfolding
#         x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

#         batch_size, in_channels, in_height, in_width = x.size()
#         kernel_size = self.weights_u.shape[2]
#         out_channels = self.weights_u.shape[0]

#         out_height = (in_height - kernel_size) // self.stride + 1
#         out_width = (in_width - kernel_size) // self.stride + 1

#         # Unfold the input tensor
#         unfolded = F.unfold(x, kernel_size=kernel_size, stride=self.stride)
#         unfolded = unfolded.view(
#             batch_size, in_channels * kernel_size * kernel_size, out_height * out_width
#         )

#         # Weights for positive input component
#         weights_u_reshaped = self.weights_u.view(out_channels, -1)

#         # Logarithm of absolute value
#         log_abs_unfolded = torch.log(
#             torch.abs(unfolded) + epsilon
#         )  # Add small constant for numerical stability

#         # Compute the exponential part (corresponding to Eq. 3)
#         exp_term = torch.exp(
#             torch.einsum("oc,bcp->bop", weights_u_reshaped, log_abs_unfolded)
#         )

#         # Weights for indicator function
#         weights_v_reshaped = self.weights_v.view(out_channels, -1)

#         # Indicator function (vectorized)
#         indicator = (unfolded < 0).float()

#         # Compute the cosine term (corresponding to Eq. 4)
#         cosine_term = torch.cos(
#             math.pi * torch.einsum("oc,bcp->bop", weights_v_reshaped, indicator)
#         ).view(batch_size, out_channels, out_height, out_width)

#         # Combine the terms
#         output = (
#             exp_term.view(batch_size, out_channels, out_height, out_width) * cosine_term
#         )

#         return output

class ProductUnits(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ProductUnits, self).__init__()
        self.weights_u = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)) # for positive inputs
        self.weights_v = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)) # for indicator function
        self.stride = stride
        # Initialize weights (important!)
        nn.init.kaiming_uniform_(self.weights_u, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_v, a=math.sqrt(5))

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.size()
        kernel_size = self.weights_u.shape[2]
        out_channels = self.weights_u.shape[0]

        out_height = (in_height - kernel_size) // self.stride + 1
        out_width = (in_width - kernel_size) // self.stride + 1

        # Unfold the input tensor
        unfolded = F.unfold(x, kernel_size=kernel_size, stride=self.stride)
        unfolded = unfolded.view(batch_size, in_channels * kernel_size * kernel_size, out_height * out_width)

        # Weights for positive input component
        weights_u_reshaped = self.weights_u.view(out_channels, -1)

        # Logarithm of absolute value
        log_abs_unfolded = torch.log(torch.abs(unfolded) + 1e-10)  # Add small constant for numerical stability

        # Compute the exponential part (corresponding to Eq. 3)
        exp_term = torch.exp(torch.einsum("oc,bcp->bop", weights_u_reshaped, log_abs_unfolded))

        # Weights for indicator function
        weights_v_reshaped = self.weights_v.view(out_channels, -1)

        # Indicator function (vectorized)
        indicator = (unfolded < 0).float()

        # Compute the cosine term (corresponding to Eq. 4)
        cosine_term = torch.cos(math.pi * torch.einsum("oc,bcp->bop", weights_v_reshaped, indicator)).view(batch_size, out_channels, out_height, out_width)

        # Combine the terms
        output = exp_term.view(batch_size, out_channels, out_height, out_width) * cosine_term

        return output


class ConcatConv2DProductUnits(nn.Module):
    def __init__(
        self, in_channels, num_layers, initial_out_channels=16, kernel_size=3, dropout_rate=0.25
    ):
        super(ConcatConv2DProductUnits, self).__init__()
        self.kernel_size = kernel_size
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.num_layers = num_layers
        self.initial_out_channels = initial_out_channels

        in_channels_ = in_channels
        out_channels_ = initial_out_channels
        for i in range(num_layers):
            self.layers.append(
                nn.ModuleList([
                    ProductUnits(in_channels_, out_channels_),
                    StandardConv2D(
                        in_channels_, out_channels_,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                        dropout_rate=dropout_rate
                    ),
                ])
            )
            self.bn_layers.append(
                nn.ModuleList([
                    nn.BatchNorm2d(out_channels_),
                    nn.Identity(),  # Already handled in StandardConv2D
                ])
            )
            in_channels_ = out_channels_ * 2
            out_channels_ = out_channels_ * 2

        self.out_channels_list = [
            initial_out_channels * (2**i) for i in range(num_layers)
        ]

    def forward(self, x):
        for i, (product_conv, conv) in enumerate(self.layers):
            bn_prod, _ = self.bn_layers[i]

            y = product_conv(x)
            y = bn_prod(y)
            y = F.relu(y)
            y = F.dropout(y, p=0.5, training=self.training)

            z = conv(x)  # StandardConv2D handles BN, ReLU, Dropout

            if y.shape[2:] != z.shape[2:]:
                z = F.interpolate(z, size=y.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat((y, z), dim=1)

        return x

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

## This code refers the code in SIREN(https://github.com/vsitzmann/siren)
class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

def positional_encoding(x, d):

    pi = 3.1415927410125732

    dd = torch.arange(0,d).to(x.device)
    phase = torch.pow(2, dd)*pi
    base = torch.ones_like(x).unsqueeze(-1)
    dd = dd[None, None, :]
    dd = dd * base
    phase = torch.pow(2, dd)*pi
    phase = phase*x[...,None]

    sinp = torch.sin(phase)
    cosp = torch.cos(phase)
    pe = torch.stack([sinp, cosp], dim=-1)
    pe = pe.reshape(x.shape[0], d*3*2)

    return pe


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def last_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)


class MLPNet(nn.Module):
    def __init__(
            self,
            args
    ):
        super(MLPNet, self).__init__()

        self.args = args
        self.use_pe = self.args.use_pe
        if self.use_pe:
            self.xyz_dims = self.args.pe_dimen * 3 * 2
        else:
            self.xyz_dims = 3

        # Init the network structure
        dims_init = [self.xyz_dims] + self.args.init_dims + [self.args.output_dims]
        self.num_layers_init = len(dims_init)
  
        for layer in range(0, self.num_layers_init - 1):
            in_dim = dims_init[layer]
            if layer + 1 in self.args.init_latent_in:
                out_dim = dims_init[layer + 1] - dims_init[0]
            else:
                out_dim = dims_init[layer + 1]
                if self.args.xyz_in_all and layer != self.num_layers_init - 2:
                    out_dim -= self.xyz_dims
            if self.args.weight_norm and layer in self.args.init_norm_layers:
                setattr(
                    self,
                    "lin_" + str(layer),
                    nn.utils.weight_norm(nn.Linear(in_dim, out_dim)),
                )
            else:
                setattr(self, "lin_" + str(layer), nn.Linear(in_dim, out_dim))
            if (
                    (not self.args.weight_norm)
                    and self.args.init_norm_layers is not None
                    and layer in self.args.init_norm_layers
            ):
                setattr(self, "bn_" + str(layer), nn.LayerNorm(out_dim))

        # Activate function
        if self.args.activation == "relu":
            self.activation = nn.ReLU()
        elif self.args.activation == "softplus":
            self.activation = nn.Softplus(beta=100)
        else:
            self.activation = Sine()
            for layer in range(0, self.num_layers_init - 1):
                lin = getattr(self, "lin_" + str(layer))
                if layer == 0:
                    first_layer_sine_init(lin)
                elif layer == self.num_layers_init - 2:
                    last_layer_sine_init(lin)
                else:
                    sine_init(lin)

        # Setup the switch for the last layer output
        self.sp = nn.Softplus(beta=100)
        self.relu = nn.ReLU()

        # Setup dropouts
        if self.args.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)


    # input: N x (L+3)
    def forward(self, input, write_debug=False):
        xyz = input[..., -3:]

        x = xyz

        if self.use_pe:
            pe = positional_encoding(x, self.args.pe_dimen)
            x = pe

        # Forward the network
        for layer in range(0, self.num_layers_init - 1):
            lin = getattr(self, "lin_" + str(layer))
            if layer != 0 and self.args.xyz_in_all:
                x = torch.cat([x, xyz], -1)
            x = lin(x)
            # bn and activation
            if layer < self.num_layers_init - 2:
                if (
                        self.args.init_norm_layers is not None
                        and layer in self.args.init_norm_layers
                        and not self.args.weight_norm
                ):
                    bn = getattr(self, "bn_" + str(layer))
                    x = bn(x)
                x = self.activation(x)
                if self.args.dropout is not None and layer in self.args.dropout:
                    x = F.dropout(x, p=self.args.dropout_prob, training=self.training)
        if self.args.last_activation == 'softplus':
            x = self.sp(x)
        else:
            x = self.relu(x)
        return x


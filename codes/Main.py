import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

import torch
import torch.nn.functional as F
import numpy as np
import math
import inspect
import functools
from torch.distributions.normal import Normal

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.ndimage import binary_closing
import cv2
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

import sys
import numpy as np
import os
from os.path import join
import pandas as pd

from datetime import datetime

import argparse
import tqdm



import random

import os
from typing_extensions import Self
import numpy as np
import pandas as pd
from datetime import datetime


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda
import matplotlib
import seaborn as sns
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
#from torchsummary import summary
import time
import os
import copy

from tqdm import tqdm
import torchio

address_tag = '/home/guests/mohammad_kalbasi/Dataset/MNI/MNI/'
save_path = '/home/guests/mohammad_kalbasi/Feature_morph2.0/results'


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x
import torch
import torch.nn.functional as F
import numpy as np
import math


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None,device = 'cuda'):
        self.win = win
        self.device = device

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()
def store_config_args(func):
    """
    Class-method decorator that saves every argument provided to the
    function as a dictionary in 'self.config'. This is used to assist
    model loading - see LoadableModel.
    """

    argspec = inspect.getfullargspec(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.config = {}

        # first save the default values
        if argspec.defaults:
            for attr, val in zip(reversed(argspec.args), reversed(argspec.defaults)):
                self.config[attr] = val

        # next handle positional args
        for attr, val in zip(argspec.args[1:], args):
            self.config[attr] = val

        # lastly handle keyword args
        if kwargs:
            for attr, val in kwargs.items():
                self.config[attr] = val

        return func(self, *args, **kwargs)
    return wrapper


class LoadableModel(nn.Module):
    """
    Base class for easy pytorch model loading without having to manually
    specify the architecture configuration at load time.

    We can cache the arguments used to the construct the initial network, so that
    we can construct the exact same network when loading from file. The arguments
    provided to __init__ are automatically saved into the object (in self.config)
    if the __init__ method is decorated with the @store_config_args utility.
    """

    # this constructor just functions as a check to make sure that every
    # LoadableModel subclass has provided an internal config parameter
    # either manually or via store_config_args
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'config'):
            raise RuntimeError('models that inherit from LoadableModel must decorate the '
                               'constructor with @store_config_args')
        super().__init__(*args, **kwargs)

    def save(self, path):
        """
        Saves the model configuration and weights to a pytorch file.
        """
        # don't save the transformer_grid buffers - see SpatialTransformer doc for more info
        sd = self.state_dict().copy()
        grid_buffers = [key for key in sd.keys() if key.endswith('.grid')]
        for key in grid_buffers:
            sd.pop(key)
        torch.save({'config': self.config, 'model_state': sd}, path)

    @classmethod
    def load(cls, path, device):
        """
        Load a python model configuration and weights.
        """
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        return model
class BlurPool(nn.Module):
    def __init__(self, ch, stride=2) -> None:
        super().__init__()

        self.f = None
        self.ch = ch
        self.filter_1d = torch.FloatTensor([1., 4., 6., 4., 1.])
        self.filter = self.filter_1d.view(1, 1, -1, 1, 1) * self.filter_1d.view(1, 1, 1, -1, 1) * self.filter_1d.view(1, 1, 1, 1, -1)
        self.filter = (self.filter / torch.sum(self.filter)).repeat(self.ch, 1, 1, 1, 1)
        self.register_buffer('filter3D', self.filter)
        self.pad = 2
        self.stride = stride
        

    def __repr__(self):
        return f"BlurPool(stride={self.stride}, radius={self.f.size(0) // 2})"

    

    def forward(self, x):
        ch = x.size(1)

        y = F.pad(x, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), mode="replicate")
        y = F.conv3d(y, self.filter3D, groups=ch, stride=self.stride)

        return y


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False,
                 feature_emb = 8):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = [[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]]


        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [BlurPool(ch).to(device) for ch in enc_nf]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf
        Conv_embedding = getattr(nn, 'Conv%dd' % ndims)
        self.conv_embedding = Conv_embedding(prev_nf, feature_emb,1)
        

        # cache final number of features
        self.final_nf = feature_emb

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        #return self.conv_embedding(x)
        return x
class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-2).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out
class FlowExtract2D(nn.Module):
    def __init__(self,size = (112,112), max_displacement=20,temperature = 0.0001,device = 'cuda'):
      super().__init__()
      self.si1 = size[0]
      self.si2 = size[1]
      self.max_disp = max_displacement
      self.padder = nn.ConstantPad2d(max_displacement, 0)
      x_dir,y_dir = torch.meshgrid([torch.arange(-1*max_displacement,max_displacement+1),
                                    torch.arange(-1*max_displacement,max_displacement+1)])
      self.x_dir,self.y_dir = x_dir.reshape(-1),y_dir.reshape(-1)
      self.flow_guide = torch.stack((self.x_dir,self.y_dir), dim=-1).float().to(device)
      self.t = temperature
    def forward(self,in1,in2):
      in1 = F.normalize(in1,dim = 1)
      in2 = F.normalize(in2,dim = 1)


      in2_padded = self.padder(in2)
      similarity = torch.cat([
                  torch.sum(in1 * in2_padded[:, :, self.max_disp+dx:self.max_disp+dx+self.si1, self.max_disp+dy:self.max_disp+dy+self.si2], 1, keepdim=True)
                  for dx, dy in zip( self.x_dir,self.y_dir)], 1)
      weights =   F.softmax(similarity/self.t, dim=1)
      weights = weights.permute(0,2,3,1)
      flow =  torch.matmul(weights, self.flow_guide)
      return flow.permute(0,3,1,2)
class FlowExtract3D(nn.Module):
    def __init__(self,size = (56,56,56), max_displacement=6,temperature = 0.0001,device = 'cuda'):
      super().__init__()
      self.si1 = size[0]
      self.si2 = size[1]
      self.si3 = size[2]
      self.max_disp = max_displacement
      self.padder = nn.ConstantPad3d(max_displacement, 0)
      x_dir,y_dir,z_dir = torch.meshgrid([torch.arange(-1*max_displacement,max_displacement+1),
                                          torch.arange(-1*max_displacement,max_displacement+1),
                                          torch.arange(-1*max_displacement,max_displacement+1)])
      self.x_dir,self.y_dir,self.z_dir = x_dir.reshape(-1),y_dir.reshape(-1),z_dir.reshape(-1)
      self.flow_guide = torch.stack((self.x_dir,self.y_dir,self.z_dir), dim=-1).float().to(device)
      self.t = temperature
    def forward(self,in1,in2):
      in1 = F.normalize(in1,dim = 1)
      in2 = F.normalize(in2,dim = 1)


      in2_padded = self.padder(in2)
      similarity = torch.cat([
                  torch.sum(in1 * in2_padded[:, :, self.max_disp+dx:self.max_disp+dx+self.si1, self.max_disp+dy:self.max_disp+dy+self.si2,self.max_disp+dz:self.max_disp+dz+self.si3], 1, keepdim=True)
                  for dx, dy,dz in zip( self.x_dir,self.y_dir,self.z_dir)], 1)
      weights =   F.softmax(similarity/self.t, dim=1)
      weights = weights.permute(0,2,3,4,1)
      flow =  torch.matmul(weights, self.flow_guide)
      return flow.permute(0,4,1,2,3)
class FeatureNet3D(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 unet_average_pool = True):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        self.unet_average_pool = unet_average_pool

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        if unet_average_pool:
          self.flow = FlowExtract3D(size = (32,32,32),max_displacement=5,device = device) # it is not generalized
          self.average_pool = nn.AvgPool3d(kernel_size=4, stride=4)
        else:
          self.flow = FlowExtract3D(size = inshape)


        self.unet_half_res = unet_half_res


        # configure optional resize layers (downsize)
        if  unet_average_pool:
            self.resize = ResizeTransform(0.25, ndims)
        else:
            self.resize = None



        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate =  None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet

        x1 = self.unet_model(source)
        x1 = self.average_pool(x1)
        x2 = self.unet_model(target)
        x2 = self.average_pool(x2)

        # transform into flow field
        flow_field = self.flow(x1,x2)

        # resize flow for integration
        pos_flow = flow_field
        if self.unet_average_pool:
            pos_flow = self.resize(pos_flow)
            flow_field = pos_flow

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow

def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def mse_loss(x, y):
    return torch.mean( (x - y) ** 2 )


def diceLoss(y_true, y_pred):
    top = 2 * (y_true * y_pred, [1, 2, 3]).sum()
    bottom = torch.max((y_true + y_pred, [1, 2, 3]).sum(), 50)
    dice = torch.mean(top / bottom)
    return -dice


def ncc_loss(I, J, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    conv_fn = getattr(F, 'conv%dd' % ndims)
    I2 = I*I
    J2 = J*J
    IJ = I*J

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0]/2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1,1)
        padding = (pad_no, pad_no)
    else:
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross*cross / (I_var*J_var + 1e-5)

    return -1 * torch.mean(cc)



def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross

def pdist_squared(x):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist

def MINDSSC(img, radius=2, dilation=2):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

    # kernel size
    kernel_size = radius * 2 + 1

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor([[0,1,1],
                                      [1,1,0],
                                      [1,0,1],
                                      [1,1,2],
                                      [2,1,1],
                                      [1,2,1]]).long()

    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask,:]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask,:]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)

    # compute patch-ssd
    ssd = F.avg_pool3d(rpad2((F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2), kernel_size, stride=1)

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean()*0.001, mind_var.mean()*1000)
    mind /= mind_var
    mind = torch.exp(-mind)

    #permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

    return mind

def mind_loss(x, y):
    return torch.mean( (MINDSSC(x) - MINDSSC(y)) ** 2 )
class MINDLoss3D(torch.nn.Module):

    def __init__(self, radius=2, dilation=2):
        super(MINDLoss3D, self).__init__()
        self.radius = radius
        self.dilation = dilation


    def forward(self, input, target):
        in_mind = MINDSSC(input,self.radius,self.dilation)
        tar_mind =MINDSSC(target,self.radius,self.dilation)

        return torch.mean( (in_mind - tar_mind) ** 2 )

class FeatureNet3D_two_level(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 unet_average_pool = True):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        self.unet_average_pool = unet_average_pool

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        if unet_average_pool:
          self.flow_aggr = FlowExtract3D(size = (32,32,32),max_displacement=5,device = device) # it is not generalized
          self.average_pool_aggr = nn.AvgPool3d(kernel_size=4, stride=4)
          self.flow_mel = FlowExtract3D(size = (64,64,64),max_displacement=2,device = device) # it is not generalized
          self.average_pool_mel = nn.AvgPool3d(kernel_size=2, stride=2)
        else:
          self.flow = FlowExtract3D(size = inshape)


        self.unet_half_res = unet_half_res




        self.resize_aggr = ResizeTransform(0.25, ndims)
        self.resize_mel = ResizeTransform(0.5, ndims)





        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate =  None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet

        x1 = self.unet_model(source)
        x1_agg = self.average_pool_aggr(x1)
        x2 = self.unet_model(target)
        x2_agg = self.average_pool_aggr(x2)
        x2_mel = self.average_pool_mel(x2)
        if registration == True:
          return x1,x2

        # transform into flow field
        flow_field = self.flow_aggr(x1_agg,x2_agg)

        # resize flow for integration
        pos_flow_agg = flow_field
        pos_flow_agg = self.resize_aggr(pos_flow_agg)

        flow_field_agg = pos_flow_agg
        # warp image with flow field
        x1 = self.transformer(x1, pos_flow_agg)
        x1_mel = self.average_pool_mel(x1)
        pos_flow_mel = self.flow_mel(x1_mel,x2_mel)
        pos_flow_mel = self.resize_mel(pos_flow_mel)
        pos_flow = pos_flow_agg + pos_flow_mel



        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None



        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow


import SimpleITK as sitk

class Elastic_transform(nn.Module):
  def __init__(self,size,control_points,max_displacement,num_dimensions = 3,num_locked_borders =2,SPLINE_ORDER = 3):
    super().__init__()
    self.size = size
    self.control_points = control_points
    self.max_displacement = max_displacement
    self.num_dimensions = num_dimensions
    self.num_locked_borders = num_locked_borders
    self.SPLINE_ORDER = SPLINE_ORDER
    self.num_control_points = np.array(control_points, np.uint32)
    self.mesh_size = self.num_control_points - self.SPLINE_ORDER
    img_template_size = np.zeros(size)
    self.img_template = sitk.GetImageFromArray(img_template_size)
    self.spt_trans = SpatialTransformer(size)
  def forward(self,img):
    grid_shape = self.num_control_points
    coarse_field = torch.rand(*grid_shape, self.num_dimensions)  # [0, 1)
    coarse_field -= 0.5  # [-0.5, 0.5)
    coarse_field *= 2  # [-1, 1]
    for dimension in range(3):
        # [-max_displacement, max_displacement)
        coarse_field[..., dimension] *= self.max_displacement[dimension]
    # Set displacement to 0 at the borders
    for i in range(self.num_locked_borders):
        coarse_field[i, :] = 0
        coarse_field[-1 - i, :] = 0
        coarse_field[:, i] = 0
        coarse_field[:, -1 - i] = 0
    num_control_points = coarse_field.shape[:-1]
    mesh_shape = [n - self.SPLINE_ORDER for n in num_control_points]
    bspline_transform = sitk.BSplineTransformInitializer(self.img_template, mesh_shape)
    coarse_field = coarse_field.numpy()
    parameters = coarse_field.flatten(order='F').tolist()
    bspline_transform.SetParameters(parameters)
    flow_real = sitk.TransformToDisplacementField(bspline_transform
                                    ,sitk.sitkVectorFloat64,
                                    self.img_template.GetSize(),
                              self.img_template.GetOrigin(),
                              self.img_template.GetSpacing(),
                              self.img_template.GetDirection())
    flow_real = torch.tensor(sitk.GetArrayFromImage(flow_real).transpose(3,0,1,2)).float()

    out_spat = self.spt_trans(img.unsqueeze(0),flow_real.unsqueeze(0))
    return out_spat.squeeze(0),flow_real.squeeze()

images_t1 = np.zeros((28,128,128,128))
images_f1 = np.zeros((28,128,128,128))
images_mask = np.zeros((28,128,128,128))

flair_tag = 'pet.nii.gz'
mask_tag = 'pet.nii.gz'
T1_tag = 'T1w.nii.gz'
counter = 0
for i in np.arange(1,10):
  folder_name = 'sub-000'+str(i)+'/'
  start_name = 'sub-000' + str(i)+'_space-MNI_'
  mask_temp = nib.load(address_tag +folder_name+start_name+mask_tag).get_fdata()
  mask_temp = (mask_temp[:,:,:] - np.min(mask_temp[:,:,:]))/(np.max(mask_temp[:,:,:]) - np.min(mask_temp[:,:,:]))
  mask_temp = np.pad(mask_temp, ((18,18),(0, 0),(9, 8)), mode='constant')
  images_mask[counter,:,:,:] = zoom(mask_temp, (0.2633*2,0.2633*2,0.2633*2))
  binary_mask = images_mask[counter,:,:,:]>0.01
  kernel = np.ones((3, 3, 3))
  dilated_mask = binary_closing(binary_mask, iterations=1)

  folder_name = 'sub-000'+str(i)+'/'
  start_name = 'sub-000' + str(i)+'_space-MNI_'
  t1_temp = nib.load(address_tag +folder_name+start_name+T1_tag).get_fdata()
  t1_temp = (t1_temp[:,:,:] - np.min(t1_temp[:,:,:]))/(np.max(t1_temp[:,:,:]) - np.min(t1_temp[:,:,:]))
  t1_temp = np.pad(t1_temp, ((18,18),(0, 0),(9, 8)), mode='constant')
  images_t1[counter,:,:,:] = zoom(t1_temp, (0.2633*2,0.2633*2,0.2633*2))*dilated_mask
  




  f1_temp = nib.load(address_tag +folder_name+start_name+flair_tag).get_fdata()
  f1_temp_s = (f1_temp[:,:,:] - np.min(f1_temp[:,:,:]))/(np.max(f1_temp[:,:,:]) - np.min(f1_temp[:,:,:]))
  f1_temp_s = np.pad(f1_temp_s, ((18,18),(0, 0),(9, 8)), mode='constant')
  images_f1[counter,:,:,:] = zoom(f1_temp_s,  (0.2633*2,0.2633*2,0.2633*2))*dilated_mask

  counter = counter + 1
for i in np.arange(10,29):
  folder_name = 'sub-00'+str(i)+'/'
  start_name = 'sub-00' + str(i)+'_space-MNI_'
  mask_temp = nib.load(address_tag +folder_name+start_name+mask_tag).get_fdata()
  mask_temp = (mask_temp[:,:,:] - np.min(mask_temp[:,:,:]))/(np.max(mask_temp[:,:,:]) - np.min(mask_temp[:,:,:]))
  mask_temp = np.pad(mask_temp, ((18,18),(0, 0),(9, 8)), mode='constant')
  images_mask[counter,:,:,:] = zoom(mask_temp, (0.2633*2,0.2633*2,0.2633*2))
  binary_mask = images_mask[counter,:,:,:]>0.01
  kernel = np.ones((3, 3, 3))
  dilated_mask = binary_closing(binary_mask, iterations=1)

  folder_name = 'sub-00'+str(i)+'/'
  start_name = 'sub-00' + str(i)+'_space-MNI_'
  t1_temp = nib.load(address_tag +folder_name+start_name+T1_tag).get_fdata()
  t1_temp = (t1_temp[:,:,:] - np.min(t1_temp[:,:,:]))/(np.max(t1_temp[:,:,:]) - np.min(t1_temp[:,:,:]))
  t1_temp = np.pad(t1_temp, ((18,18),(0, 0),(9, 8)), mode='constant')
  images_t1[counter,:,:,:] = zoom(t1_temp, (0.2633*2,0.2633*2,0.2633*2))*dilated_mask
  




  f1_temp = nib.load(address_tag +folder_name+start_name+flair_tag).get_fdata()
  f1_temp_s = (f1_temp[:,:,:] - np.min(f1_temp[:,:,:]))/(np.max(f1_temp[:,:,:]) - np.min(f1_temp[:,:,:]))
  f1_temp_s = np.pad(f1_temp_s, ((18,18),(0, 0),(9, 8)), mode='constant')
  images_f1[counter,:,:,:] = zoom(f1_temp_s,  (0.2633*2,0.2633*2,0.2633*2))*dilated_mask

  counter = counter + 1
transform2 = Elastic_transform((128,128,128),(7,7,7),(16,16,16))
class CustomImageDataset(Dataset):
    def __init__(self, f_img, t_img,dephormatic, transform=None):
        self.f_img = f_img
        self.t_img = t_img
        self.transform = transform
        self.dep = dephormatic

    def __len__(self):
        return len(self.f_img)

    def __getitem__(self, idx):
      f1_img = self.f_img[idx]
      t1_img = self.t_img[idx]


      if self.transform:
          f1_img = torch.from_numpy(f1_img).float()
          t1_img = torch.from_numpy(t1_img).float()
          f1_img = f1_img.unsqueeze(0)
          t1_img = t1_img.unsqueeze(0)
          t1_img_t,flow_real = self.dep(t1_img)

      return f1_img, t1_img,t1_img_t,flow_real
transform=transforms.Compose([transforms.ToTensor(),])
data_set = CustomImageDataset(images_f1,images_t1,transform2,transform)

dt = DataLoader(data_set, batch_size=1, shuffle=True)



images_t1 = np.zeros((4,128,128,128))
images_f1 = np.zeros((4,128,128,128))
images_mask = np.zeros((4,128,128,128))



counter = 0

for i in np.arange(29,33):
  folder_name = 'sub-00'+str(i)+'/'
  start_name = 'sub-00' + str(i)+'_space-MNI_'
  mask_temp = nib.load(address_tag +folder_name+start_name+mask_tag).get_fdata()
  mask_temp = (mask_temp[:,:,:] - np.min(mask_temp[:,:,:]))/(np.max(mask_temp[:,:,:]) - np.min(mask_temp[:,:,:]))
  mask_temp = np.pad(mask_temp, ((18,18),(0, 0),(9, 8)), mode='constant')
  images_mask[counter,:,:,:] = zoom(mask_temp, (0.2633*2,0.2633*2,0.2633*2))
  binary_mask = images_mask[counter,:,:,:]>0.01
  kernel = np.ones((3, 3, 3))
  dilated_mask = binary_closing(binary_mask, iterations=1)

  folder_name = 'sub-00'+str(i)+'/'
  start_name = 'sub-00' + str(i)+'_space-MNI_'
  t1_temp = nib.load(address_tag +folder_name+start_name+T1_tag).get_fdata()
  t1_temp = (t1_temp[:,:,:] - np.min(t1_temp[:,:,:]))/(np.max(t1_temp[:,:,:]) - np.min(t1_temp[:,:,:]))
  t1_temp = np.pad(t1_temp, ((18,18),(0, 0),(9, 8)), mode='constant')
  images_t1[counter,:,:,:] = zoom(t1_temp, (0.2633*2,0.2633*2,0.2633*2))*dilated_mask
  


  f1_temp = nib.load(address_tag +folder_name+start_name+flair_tag).get_fdata()
  f1_temp_s = (f1_temp[:,:,:] - np.min(f1_temp[:,:,:]))/(np.max(f1_temp[:,:,:]) - np.min(f1_temp[:,:,:]))
  f1_temp_s = np.pad(f1_temp_s, ((18,18),(0, 0),(9, 8)), mode='constant')
  images_f1[counter,:,:,:] = zoom(f1_temp_s,  (0.2633*2,0.2633*2,0.2633*2))*dilated_mask

  counter = counter + 1

data_set = CustomImageDataset(images_f1,images_t1,transform2,transform)

dv = DataLoader(data_set, batch_size=1, shuffle=True)

def train_model_manual(model,spt_transform, criterion_m,criterion_s,criterion_v,grad_loss, optimizer,scheduler,val_beark,clip, num_epochs,dt_local,dv_local):
  """
  function for train our model! in this function we use dataloader directly.
  inputs:
      model: input model
      criterion: desired loss function
      optimizer: our optimizer(!)
      scheduler: for changing learning rate after sum epochs
      num_epochs: number of epoches
      val_beark: threshold for early stopping, if after "val_beark" steps our model don't get better, we end procces
      clip: for gradient clipping
  output:
      model: our trained model!


  """

  since = time.time()
  counter_loss_val = 0
  loss_val_np = np.zeros((num_epochs,))

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  best_loss = 100000000000
  counter_val_beark = 0
  for epoch in range(num_epochs):
    ### Training
    model.train()
    loss_train = 0
    acc_train = 0
    counter = 1
    for dl in tqdm(dt_local, total=len(dt_local), desc="Training... "):


      # 1. Forward pass

      source,source2,target,field_true = dl
      source = source.float()
      source2 = source2.float()
      target = target.float()
      field_true = field_true.float()
      source = source.to(device)
      source2 = source2.to(device)
      target = target.to(device)
      field_true = field_true.to(device)

      y_target1,flow1 = model(target,source)
      y_target3,flow3 = model(source,target)

      y_target2,flow2 = model(source2,target)
      loss = criterion_v.loss(source2,y_target1)+ criterion_m(y_target3,target) + 0.01*grad_loss.loss(None,flow2)+ criterion_s(y_target2,target) + 0.1*grad_loss.loss(None,flow1) +0.1*grad_loss.loss(None,flow3)     # loss = criterion_m(y_target1,source)+ criterion_m(y_target3,target) + criterion_s(goal,similarity_value) + 0.01*grad_loss.loss(_,flow1) +0.01*grad_loss.loss(_,flow3)
      # original 0.001
      #



      #loss = torch.mean(-1*criterion_m(y_target1,target)) + 0.01*grad_loss.loss(_,flow1) + criterion_s(y_target2,target) + 0.01*grad_loss.loss(_,flow2)
     # loss = criterion(y_target,target)






      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backwards
      loss.backward()


      # 5. Optimizer step
      optimizer.step()

      loss.detach()
      loss_train = loss_train + loss

      del source,source2,target,flow1,flow3,flow2
     


    loss_train = loss_train/counter
    scheduler.step()

    ### Testing
    loss_test = 0
    acc_test = 0
    counter = 1
    model.eval()
    with torch.inference_mode():
      for batch in tqdm(dv_local, total=len(dv_local), desc="Validating... "):
      # 1. Forward pass

        with torch.no_grad():

          source,source2,target,field_true = batch

          source = source.to(device)
          source2 = source2.to(device)
          target = target.to(device)
          field_true = field_true.to(device)


          y_target,flow = model(target,source)
          loss_t = criterion_v.loss(source2,y_target)









          loss_test = loss_test + loss_t.detach()
          loss_val_np[counter_loss_val] = loss_t.detach().cpu().numpy()
          
          del source,source2,target

      loss_test = loss_test/counter
      counter_loss_val += 1

    if loss_test <= best_loss:
             #   best_loss = loss_test
                counter_val_beark = 0
             #   best_loss = loss_test
                best_model_wts = copy.deepcopy(model.state_dict())

    if loss_test > best_loss:
                counter_val_beark = counter_val_beark + 1
                if (counter_val_beark > val_beark):
                  print(f"early stopping happend!")
                  break;

    # Print out what's happening
    if epoch % 5 == 0:
      print(f"Epoch: {epoch} | Loss: {loss_train:.5f}, Acc: {acc_train:.2f}% | Test Loss: {loss_test:.5f}, Test Acc: {acc_test:.2f}%")
    if (epoch + 1) % 500 == 0:
            save_file_path = f"{save_path}/featuremorph_epoch_supervised_mri_t1w_pet_blur_{epoch+1}.pth"
            save_file_path_np = f"{save_path}/val_loss_trend.npy"
            torch.save(model.state_dict(), save_file_path)
            np.save(save_file_path_np, loss_val_np)
            print(f"Model saved to {save_file_path}")
  time_elapsed = time.time() - since
  print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Best val loss: {best_loss:4f}')

  # load best model weights
  model.load_state_dict(best_model_wts)

  return model,best_loss,best_acc
device = 'cuda'

nb_features = [
    [16,32, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 16,16]  # decoder features
]
unet_model =FeatureNet3D_two_level((128,128,128),nb_unet_features = nb_features)
unet_model = unet_model.to(device)
#criterion_m = MutualInformation(num_bins=256, sigma=0.5, normalize=True).to(device)
criterion_m = MINDLoss3D().to(device)
criterion_s = torch.nn.MSELoss()
optimizer_conv = optim.Adam(unet_model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=400, gamma=0.5)


spt_transform = SpatialTransformer((128,128,128)).to(device)
model_layer1,best_loss,best_acc = train_model_manual(unet_model,spt_transform ,criterion_m,criterion_s,NCC(),Grad(), optimizer_conv,exp_lr_scheduler,1001,0.0,1001,dt,dv)
def train_model_manual2(model,spt_transform, criterion_m,criterion_s,criterion_v,grad_loss, optimizer,scheduler,val_beark,clip, num_epochs,dt_local,dv_local):
  """
  function for train our model! in this function we use dataloader directly.
  inputs:
      model: input model
      criterion: desired loss function
      optimizer: our optimizer(!)
      scheduler: for changing learning rate after sum epochs
      num_epochs: number of epoches
      val_beark: threshold for early stopping, if after "val_beark" steps our model don't get better, we end procces
      clip: for gradient clipping
  output:
      model: our trained model!


  """

  since = time.time()
  counter_loss_val = 0
  loss_val_np = np.zeros((num_epochs,))

  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  best_loss = 100000000000
  counter_val_beark = 0
  for epoch in range(num_epochs):
    ### Training
    model.train()
    loss_train = 0
    acc_train = 0
    counter = 1
    for dl in tqdm(dt_local, total=len(dt_local), desc="Training... "):


      # 1. Forward pass

      source,source2,target,field_true = dl
      source = source.float()
      source2 = source2.float()
      target = target.float()
      field_true = field_true.float()
      source = source.to(device)
      source2 = source2.to(device)
      target = target.to(device)
      field_true = field_true.to(device)

      y_target1,flow1 = model(target,source)
      y_target3,flow3 = model(source,target)

      y_target2,flow2 = model(source2,target)
      loss = criterion_v.loss(source2,y_target1)+ criterion_m(y_target3,target) + 0.003*grad_loss.loss(None,flow2)+ criterion_s(y_target2,target) + 0.01*grad_loss.loss(None,flow1) +0.01*grad_loss.loss(None,flow3)     # loss = criterion_m(y_target1,source)+ criterion_m(y_target3,target) + criterion_s(goal,similarity_value) + 0.01*grad_loss.loss(_,flow1) +0.01*grad_loss.loss(_,flow3)
      # original 0.001
      #



      #loss = torch.mean(-1*criterion_m(y_target1,target)) + 0.01*grad_loss.loss(_,flow1) + criterion_s(y_target2,target) + 0.01*grad_loss.loss(_,flow2)
     # loss = criterion(y_target,target)






      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backwards
      loss.backward()


      # 5. Optimizer step
      optimizer.step()

      loss.detach()
      loss_train = loss_train + loss

      del source,source2,target,flow1,flow3,flow2
     


    loss_train = loss_train/counter
    scheduler.step()

    ### Testing
    loss_test = 0
    acc_test = 0
    counter = 1
    model.eval()
    with torch.inference_mode():
      for batch in tqdm(dv_local, total=len(dv_local), desc="Validating... "):
      # 1. Forward pass

        with torch.no_grad():

          source,source2,target,field_true = batch

          source = source.to(device)
          source2 = source2.to(device)
          target = target.to(device)
          field_true = field_true.to(device)


          y_target,flow = model(target,source)
          loss_t = criterion_v.loss(source2,y_target)









          loss_test = loss_test + loss_t.detach()
          loss_val_np[counter_loss_val] = loss_t.detach().cpu().numpy()
          
          del source,source2,target

      loss_test = loss_test/counter
      counter_loss_val += 1

    if loss_test <= best_loss:
             #   best_loss = loss_test
                counter_val_beark = 0
             #   best_loss = loss_test
                best_model_wts = copy.deepcopy(model.state_dict())

    if loss_test > best_loss:
                counter_val_beark = counter_val_beark + 1
                if (counter_val_beark > val_beark):
                  print(f"early stopping happend!")
                  break;

    # Print out what's happening
    if epoch % 5 == 0:
      print(f"Epoch: {epoch} | Loss: {loss_train:.5f}, Acc: {acc_train:.2f}% | Test Loss: {loss_test:.5f}, Test Acc: {acc_test:.2f}%")
    if (epoch + 1) % 250 == 0:
            save_file_path = f"{save_path}/featuremorph_epoch_supervised2_mri_t1w_pet_blur_{epoch+1}.pth"
            save_file_path_np = f"{save_path}/val_loss_trend.npy"
            torch.save(model.state_dict(), save_file_path)
            np.save(save_file_path_np, loss_val_np)
            print(f"Model saved to {save_file_path}")
  time_elapsed = time.time() - since
  print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Best val loss: {best_loss:4f}')

  # load best model weights
  model.load_state_dict(best_model_wts)

  return model,best_loss,best_acc

model_layer1 = model_layer1.to(device)
optimizer_conv = optim.Adam(model_layer1.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=300, gamma=0.5)
model_layer1,best_loss,best_acc = train_model_manual2(model_layer1,spt_transform ,criterion_m,criterion_s,NCC(),Grad(), optimizer_conv,exp_lr_scheduler,801,0.0,801,dt,dv)



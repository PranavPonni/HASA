import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple, Any
from torchvision import transforms
import torchvision.ops
import math
import copy
from util import get_activation_fn
import pdb


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel=3,padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = kernel, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        return x
    
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3,padding=0):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.rl = nn.ReLU()

    def forward(self, x):
        x = self.upconv(x)
        x = self.bn(x)
        x = self.rl(x)
        return x
    

#=================================SARNN Layer========================================#


def create_position_encoding(
    width: int, height: int, indexing="xy", normalized=True, data_format="channels_first"
):
    if normalized:
        pos_x, pos_y = np.meshgrid(
            np.linspace(0.0, 1.0, height), np.linspace(0.0, 1.0, width), indexing=indexing
        )
    else:
        pos_x, pos_y = np.meshgrid(
            np.linspace(0, height - 1, height),
            np.linspace(0, width - 1, width),
            indexing="xy",
        )

    if data_format == "channels_first":
        pos_xy = torch.from_numpy(np.stack([pos_x, pos_y], axis=0)).float()  # (2,W,H)
    else:
        pos_xy = torch.from_numpy(np.stack([pos_x, pos_y], axis=2)).float()  # (W,H,2)

    pos_x = torch.from_numpy(pos_x.reshape(height * width)).float()
    pos_y = torch.from_numpy(pos_y.reshape(height * width)).float()
    
    return pos_xy, pos_x, pos_y







class SpatialSoftmax(nn.Module):
    """Spatial Softmax
    Extract XY position from feature map of CNN

    Chelsea Finn, Xin Yu Tan, Yan Duan, Trevor Darrell, Sergey Levine, Pieter Abbeel
    ``Deep spatial autoencoders for visuomotor learning.``
    2016 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2016.
    https://ieeexplore.ieee.org/abstract/document/7487173
    """

    def __init__(self, width: int, height: int, temperature=1e-4, indexing="xy", normalized=True):
        super().__init__()
        self.width = width
        self.height = height
        self.temperature = temperature

        _, pos_x, pos_y = create_position_encoding(width, height, indexing=indexing, normalized=normalized)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, x):
        batch_size, channels, width, height = x.shape
        assert height == self.height
        assert width == self.width

        # flatten, apply softmax
        logit = x.reshape(batch_size, channels, -1)
        att_map = torch.softmax(logit / self.temperature, dim=-1)

        # compute expectation
        expected_x = torch.sum(self.pos_x * att_map, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y * att_map, dim=-1, keepdim=True)
        keys = torch.cat([expected_x, expected_y], -1)

        # keys [[x,y], [x,y], [x,y],...]
        keys = keys.reshape(batch_size, channels, 2)
        att_map = att_map.reshape(-1, channels, width, height)
        return keys, att_map
    


class SinSpatialSoftmax(nn.Module):
    """Spatial Softmax
    Extract XY position from feature map of CNN

    Chelsea Finn, Xin Yu Tan, Yan Duan, Trevor Darrell, Sergey Levine, Pieter Abbeel
    ``Deep spatial autoencoders for visuomotor learning.``
    2016 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2016.
    https://ieeexplore.ieee.org/abstract/document/7487173
    """

    def __init__(self, width: int, height: int, temperature=1e-4, indexing="xy", normalized=True):
        super().__init__()
        self.width = width
        self.height = height
        self.temperature = temperature

        _, pos_x, pos_y = create_position_encoding(width, height, indexing=indexing, normalized=normalized)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, x):
        batch_size, channels, width, height = x.shape
        assert height == self.height
        assert width == self.width

        # flatten, apply softmax
        logit = x.reshape(batch_size, channels, -1)
        
        sin_logit=torch.sin(logit)
        att_map = torch.softmax(sin_logit / self.temperature, dim=-1)

        # compute expectation
        expected_x = torch.sum(self.pos_x * att_map, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y * att_map, dim=-1, keepdim=True)
        keys = torch.cat([expected_x, expected_y], -1)

        # keys [[x,y], [x,y], [x,y],...]
        keys = keys.reshape(batch_size, channels, 2)
        att_map = att_map.reshape(-1, channels, width, height)
        return keys, att_map
    

class StraightThroughSpatialSoftmax(nn.Module):
    """Spatial Softmax
    Extract XY position from feature map of CNN

    Chelsea Finn, Xin Yu Tan, Yan Duan, Trevor Darrell, Sergey Levine, Pieter Abbeel
    ``Deep spatial autoencoders for visuomotor learning.``
    2016 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2016.
    https://ieeexplore.ieee.org/abstract/document/7487173
    """

    def __init__(self, width: int, height: int, temperature=0.1, indexing="xy", normalized=True):
        super().__init__()
        self.width = width
        self.height = height
        self.temperature = temperature

        _, pos_x, pos_y = create_position_encoding(width, height, indexing=indexing, normalized=normalized)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, x):
        batch_size, channels, width, height = x.shape
        assert height == self.height
        assert width == self.width

        # flatten, apply softmax
        logit = x.reshape(batch_size, channels, -1)

        max_indices = torch.argmax(logit, dim=-1, keepdim=True)
        one_hot = torch.zeros_like(logit).scatter_(-1, max_indices, 1)
        probs = F.softmax(logit/self.temperature, dim=-1)
        
        # compute expectation
        expected_x = torch.sum(self.pos_x * probs, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y * probs, dim=-1, keepdim=True)
        arg_expected_x=torch.sum(self.pos_x * one_hot, dim=-1, keepdim=True)
        arg_expected_y=torch.sum(self.pos_y * one_hot, dim=-1, keepdim=True)

        keys = torch.cat([expected_x, expected_y], -1)
        arg_key=torch.cat([arg_expected_x,arg_expected_y],-1)
        key_stg=keys+(arg_key-keys).detach()
        # keys [[x,y], [x,y], [x,y],...]
        key_stg = key_stg.reshape(batch_size, channels, 2)
        att_map = one_hot.reshape(-1, channels, width, height)
        return key_stg, att_map

class WeakStraightThroughSpatialSoftmax(nn.Module):
    """Spatial Softmax
    Extract XY position from feature map of CNN

    Chelsea Finn, Xin Yu Tan, Yan Duan, Trevor Darrell, Sergey Levine, Pieter Abbeel
    ``Deep spatial autoencoders for visuomotor learning.``
    2016 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2016.
    https://ieeexplore.ieee.org/abstract/document/7487173
    """

    def __init__(self, width: int, height: int, temperature=1e-1,stg_temperature=1e-4, indexing="xy", normalized=True):
        super().__init__()
        self.width = width
        self.height = height
        self.temperature = temperature
        self.stg_temperature=stg_temperature

        _, pos_x, pos_y = create_position_encoding(width, height, indexing=indexing, normalized=normalized)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, x):
        batch_size, channels, width, height = x.shape
        assert height == self.height
        assert width == self.width

        # flatten, apply softmax
        logit = x.reshape(batch_size, channels, -1)

        stg_probs=F.softmax(logit/self.stg_temperature,dim=-1)
        probs = F.softmax(logit/self.temperature, dim=-1)
        
        # compute expectation
        expected_x = torch.sum(self.pos_x * probs, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y * probs, dim=-1, keepdim=True)
        arg_expected_x=torch.sum(self.pos_x * stg_probs, dim=-1, keepdim=True)
        arg_expected_y=torch.sum(self.pos_y * stg_probs, dim=-1, keepdim=True)

        keys = torch.cat([expected_x, expected_y], -1)
        arg_key=torch.cat([arg_expected_x,arg_expected_y],-1)
        key_stg=keys+(arg_key-keys).detach()
        # keys [[x,y], [x,y], [x,y],...]
        key_stg = key_stg.reshape(batch_size, channels, 2)
        att_map = stg_probs.reshape(-1, channels, width, height)
        return key_stg, att_map
    
class GumbelSpatialSoftmax(nn.Module):
    """Spatial Softmax
    Extract XY position from feature map of CNN

    Chelsea Finn, Xin Yu Tan, Yan Duan, Trevor Darrell, Sergey Levine, Pieter Abbeel
    ``Deep spatial autoencoders for visuomotor learning.``
    2016 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2016.
    https://ieeexplore.ieee.org/abstract/document/7487173
    """

    def __init__(self, width: int, height: int, temperature=1e-4, indexing="xy", normalized=True):
        super().__init__()
        self.width = width
        self.height = height
        self.temperature = temperature

        _, pos_x, pos_y = create_position_encoding(width, height, indexing=indexing, normalized=normalized)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, x):
        batch_size, channels, width, height = x.shape
        assert height == self.height
        assert width == self.width

        # flatten, apply softmax
        logit = x.reshape(batch_size, channels, -1)/self.temperature
        att_map= F.gumbel_softmax(logit, tau=1.0, hard=True,dim=-1)

        # compute expectation
        expected_x = torch.sum(self.pos_x * att_map, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y * att_map, dim=-1, keepdim=True)
        keys = torch.cat([expected_x, expected_y], -1)

        # keys [[x,y], [x,y], [x,y],...]
        keys = keys.reshape(batch_size, channels, 2)
        att_map = att_map.reshape(-1, channels, width, height)
        return keys, att_map


class InverseSpatialSoftmax(nn.Module):
    """InverseSpatialSoftmax
    Generate heatmap from XY position

    Hideyuki Ichiwara, Hiroshi Ito, Kenjiro Yamamoto, Hiroki Mori, Tetsuya Ogata
    ``Spatial Attention Point Network for Deep-learning-based Robust Autonomous Robot Motion Generation.``
    https://arxiv.org/abs/2103.01598
    """

    def __init__(self, width: int, height: int, heatmap_size=0.1, indexing="xy", normalized=True):
        super().__init__()
        self.width = width
        self.height = height
        self.normalized = normalized
        self.heatmap_size = heatmap_size

        pos_xy, _, _ = create_position_encoding(width, height, indexing=indexing, normalized=normalized)
        self.register_buffer("pos_xy", pos_xy)

    def forward(self, keys):
        squared_distances = torch.sum(
            torch.pow(self.pos_xy[None, None] - keys[:, :, :, None, None], 2.0), axis=2
        )
        heatmap = torch.exp(-squared_distances / self.heatmap_size)
        return heatmap
    
class LearnableInverseSpatialSoftmax(nn.Module):

    def __init__(self, width: int, height: int, k_dim: int,indexing="xy", normalized=True):
        super().__init__()
        self.width = width
        self.height = height
        self.normalized = normalized
        self.heatmap_size = nn.Parameter(torch.ones(k_dim, 1)*0.1)

        pos_xy, _, _ = create_position_encoding(width, height, indexing=indexing, normalized=normalized)
        self.register_buffer("pos_xy", pos_xy)

    def forward(self, keys):
        squared_distances = torch.sum(
            torch.pow(self.pos_xy[None, None] - keys[:, :, :, None, None], 2.0), axis=2
        )
        heatmap = torch.exp(-squared_distances / ((self.heatmap_size[None,:,None])**2+1e-9))
        return heatmap,self.heatmap_size


class ScalableInverseSpatialSoftmax(nn.Module):

    def __init__(self, width: int, height: int, k_dim: int,indexing="xy", normalized=True):
        super().__init__()
        self.width = width
        self.height = height
        self.normalized = normalized
        self.heatmap_size = nn.Parameter(torch.ones(k_dim, 1)*0.1)

        pos_xy, _, _ = create_position_encoding(width, height, indexing=indexing, normalized=normalized)
        self.register_buffer("pos_xy", pos_xy)

    def forward(self, keys ,scale=None):
        squared_distances = torch.sum(
            torch.pow(self.pos_xy[None, None] - keys[:, :, :, None, None], 2.0), axis=2
        )
        if scale is None:
            heatmap = torch.exp(-squared_distances / (self.heatmap_size[None,:,None])**2+1e-9)
        else:
            heatmap = torch.exp(-squared_distances / (scale[:,:,None]*(self.heatmap_size[None,:,None])**2+1e-9))
        return heatmap,self.heatmap_size



class CatCoordChannel(nn.Module):
    def __init__(self, width: int, height: int, indexing="xy", normalized=True):
        super().__init__()
        self.pos_xy, _, _ = create_position_encoding(width, height, indexing=indexing, normalized=normalized)

    def forward(self, x):
        batch_size = x.size(0)
        pos_xy = self.pos_xy.to(x.device).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        x = torch.cat([x, pos_xy], dim=1)
        return x

#=================================SARNN Layer========================================#
#=================================Transformer Layer========================================#

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        # features = d_model
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b
    
class MaskSoftmax(nn.Module):
    def __init__(self):
        super(MaskSoftmax,self).__init__()

    def forward(self,x,mask,dim,avoid_nan=None):
        if avoid_nan is None:
            mask=mask+0.0001
        else:
            mask=mask+avoid_nan

        input_exp = mask*torch.exp(x)
        return input_exp / torch.sum(input_exp, dim=dim, keepdim=True)
    

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            `x`: shape (batch_size, max_len, d_model)

        Returns:
            same shape as input x
        """
        return self.w_2(F.relu(self.w_1(x)))
    

class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward"""

    def __init__(self, size: int, self_attn, feed_forward: FeedForward):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size), 2)
        self.size = size

    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x: torch.FloatTensor, sublayer) -> torch.FloatTensor:
        """Apply residual connection to any sublayer with the same size."""
        return x + sublayer(self.norm(x))
    


class TransformerEncoder(nn.Module):
    """The encoder of transformer

    Args:
        `n_layers`: number of stacked encoder layers
        `d_model`: model dimension
        `d_ff`: hidden dimension of feed forward layer
        `n_heads`: number of heads of self-attention
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int = 1, n_layers: int = 1):
        super(TransformerEncoder, self).__init__()
        self.multi_headed_attention = MultiHeadAttention(n_heads, d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.encoder_layer = EncoderLayer(d_model, self.multi_headed_attention, self.feed_forward)
        self.encoder = Encoder(self.encoder_layer, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.FloatTensor, mask: torch.FloatTensor) -> torch.FloatTensor:
        return self.encoder(x, mask)

class TransformerEncoder(nn.Module):
    """The encoder of transformer

    Args:
        `n_layers`: number of stacked encoder layers
        `d_model`: model dimension
        `d_ff`: hidden dimension of feed forward layer
        `n_heads`: number of heads of self-attention
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int = 1, n_layers: int = 1):
        super(TransformerEncoder, self).__init__()
        self.multi_headed_attention = MultiHeadAttention(n_heads, d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.encoder_layer = EncoderLayer(d_model, self.multi_headed_attention, self.feed_forward)
        self.encoder = Encoder(self.encoder_layer, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.FloatTensor, mask: torch.FloatTensor) -> torch.FloatTensor:
        return self.encoder(x, mask)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.mask_softmax=MaskSoftmax()

    def forward(self, query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor,
                mask: torch.FloatTensor) -> Tuple[
        torch.Tensor, Any]:
        """
        Args:
            `query`: shape (batch_size, n_heads, max_len, d_q)
            `key`: shape (batch_size, n_heads, max_len, d_k)
            `value`: shape (batch_size, n_heads, max_len, d_v)
            `mask`: shape (batch_size, 1 , 1, max_len)

        Returns:
            `weighted value`: shape (batch_size, n_heads, max_len, d_v)
            `weight matrix`: shape (batch_size, n_heads, max_len, max_len)
        """
        d_k = query.size(-1)  # d_k = d_model / n_heads
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # B*H*L*L
        p_attn = self.mask_softmax(scores,mask, dim=-1)  # B*H*L*L
        return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.sdpa = ScaledDotProductAttention()
        self.attn = None

    def forward(self, query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor,
                mask: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args: 
            `query`: shape (batch_size, max_len, d_model)
            `key`: shape (batch_size, max_len, d_model)
            `value`: shape (batch_size, max_len, d_model)
            `mask`: shape (batch_size, max_len)
        
        Returns:
            shape (batch_size, max_len, d_model)
        """
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # x: B x H x L x D_v
        x, self.attn = self.sdpa(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)

        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return x
    
def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])





#=================================Transformer Layer========================================#

class TemperatureSigmoid(nn.Module):
    def __init__(self,temp):
        super(TemperatureSigmoid, self).__init__()
        self.temp=temp
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        return self.sigmoid(x/self.temp)




#data augumentation layer

class ImageAugumentation():
    def __init__(self,device):
        self.transform = nn.Sequential(
            transforms.RandomErasing(),
            transforms.ColorJitter(brightness=0.4),
            transforms.ColorJitter(contrast=[0.6, 1.4]),
            transforms.ColorJitter(hue=[0.0, 0.04]),
            transforms.ColorJitter(saturation=[0.6, 1.4]),
        ).to(device)
    
    def augument(self,input):
        return self.transform(input)



# TDR Conv

class TDRConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, in_ratio=0.5, out_ratio=0.5, exp_times=2,
                 reduction=16, base_g=1, padding=1, dilation=1, bias=False):
        super(TDRConv, self).__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.in_ratio = in_ratio
        self.need_match = False
        base_out_channels = int(math.ceil(out_channels * out_ratio))
        diversity_out_channels = out_channels - base_out_channels
        exp_out_channels = diversity_out_channels * exp_times
        self.main_in = int(math.ceil(in_channels * in_ratio))
        exp_in = in_channels - self.main_in
        diversity_in = self.main_in + exp_out_channels
        base_groups = base_g if base_out_channels % base_g == 0 and self.main_in % base_g == 0 else 1

        if dilation == 1:
            padding = kernel_size // 2
        else:
            padding = dilation
        self.base_branch = nn.Conv2d(in_channels=self.main_in, out_channels=base_out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=base_groups, bias=bias, dilation=dilation)
        if exp_out_channels == 0:
            exp_out_channels = out_channels
        if exp_in != 0:
            self.expand_operation = nn.Conv2d(in_channels=exp_in, out_channels=exp_out_channels, kernel_size=1,
                                              stride=1, padding=0, bias=False)
        else:
            self.expand_operation = None
            diversity_in = self.main_in
        if exp_out_channels != self.out_channels:
            self.need_match = True
            self.match_branch = nn.Conv2d(in_channels=exp_out_channels, out_channels=self.out_channels, kernel_size=1,
                                          stride=1, padding=0, groups=base_out_channels, bias=False)

        diversity_groups = math.gcd(diversity_out_channels, self.main_in)
        if diversity_out_channels != 0:
            self.diversity_branch = nn.Conv2d(in_channels=diversity_in, out_channels=diversity_out_channels,
                                              kernel_size=kernel_size,
                                              stride=1, padding=padding, groups=diversity_groups, bias=bias,
                                              dilation=dilation)
        else:
            self.diversity_branch = None
        # self.bn1 = nn.BatchNorm2d(diversity_in)
        # self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.avgpool_s2_diversity = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool_s2_expand = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        se_out = out_channels
        self.fc = nn.Sequential(
            nn.Linear(se_out, max(2, se_out // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(2, se_out // reduction), se_out),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.ones(in_channels))

    def forward(self, x):
        x_m = x[:, :self.main_in, :, :]
        x_e = x[:, self.main_in:, :, :]
        y_sc = self.base_branch(x_m)
        if self.stride == 2:
            x_m = self.avgpool_s2_diversity(x_m)
            x_e = self.avgpool_s2_expand(x_e)
        if self.expand_operation is not None:
            y_e = self.expand_operation(x_e)
        else:
            y_e = 0
        if self.diversity_branch is not None:
            x_gwc = torch.cat([x_m, y_e], dim=1)
            # x_gwc = self.bn1(x_gwc)
            y_gwc = self.diversity_branch(x_gwc)
            y_m = torch.cat([y_sc, y_gwc], dim=1)
        else:
            y_m = y_sc
        # y_m = self.bn2(y_m)
        b, c, _, _ = y_m.size()
        w = self.avg_pool(y_m).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        y_m = y_m * w

        if self.need_match:
            y_e = self.match_branch(y_e)
        y = y_m + y_e
        return y[:, :self.out_channels, :, :]


# OctaveConv

class OctConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, alphas=(0.5, 0.5)):
        super(OctConv, self).__init__()
        self.alpha_in, self.alpha_out = alphas
        assert 0 <= self.alpha_in <= 1 and 0 <= self.alpha_out <= 1, "Alphas must be in interval [0, 1]"

        self.ch_in_hf = int((1 - self.alpha_in) * ch_in)
        self.ch_in_lf = ch_in - self.ch_in_hf

        self.ch_out_hf = int((1 - self.alpha_out) * ch_out)
        self.ch_out_lf = ch_out - self.ch_out_hf

        self.conv_hf_hf = nn.Conv2d(self.ch_in_hf, self.ch_out_hf, kernel_size, stride, padding=(kernel_size - stride) // 2,padding_mode="replicate")
        self.conv_hf_lf = nn.Conv2d(self.ch_in_hf, self.ch_out_lf, kernel_size, stride, padding=(kernel_size - stride) // 2,padding_mode="replicate")
        self.conv_lf_hf = nn.Conv2d(self.ch_in_lf, self.ch_out_hf, kernel_size, stride, padding=(kernel_size - stride) // 2,padding_mode="replicate")
        self.conv_lf_lf = nn.Conv2d(self.ch_in_lf, self.ch_out_lf, kernel_size, stride, padding=(kernel_size - stride) // 2,padding_mode="replicate")

    def forward(self, input):
        if self.alpha_in == 0:
            hf_input = input
            lf_input = None
        else:
            fmap_size = input.shape[-1]
            hf_input = input[:, :self.ch_in_hf * 4, ...].reshape(-1, self.ch_in_hf, fmap_size * 2, fmap_size * 2)
            lf_input = input[:, self.ch_in_hf * 4:, ...]

        HtoH = HtoL = LtoL = LtoH = 0.
        if self.alpha_in < 1:
            if self.ch_out_hf > 0:
                HtoH = self.conv_hf_hf(hf_input)
            if self.ch_out_lf > 0:
                HtoL = self.conv_hf_lf(F.avg_pool2d(hf_input, 2))
        if self.alpha_in > 0:
            if self.ch_out_hf > 0:
                LtoH = F.interpolate(self.conv_lf_hf(lf_input), scale_factor=2, mode='nearest')
                
            if self.ch_out_lf > 0:
                LtoL = self.conv_lf_lf(lf_input)

        hf_output = HtoH + LtoH
        lf_output = LtoL + HtoL

        if 0 < self.alpha_out < 1:
            fmap_size = hf_output.shape[-1] // 2
            hf_output = hf_output.reshape(-1, 4 * self.ch_out_hf, fmap_size, fmap_size)
            output = torch.cat([hf_output, lf_output], dim=1)
        elif np.isclose(self.alpha_out, 1., atol=1e-8):
            output = lf_output
        elif np.isclose(self.alpha_out, 0., atol=1e-8):
            output = hf_output
        return output
    
class MultiNoiseLoss(nn.Module):
    def __init__(self, n_losses):
        """
        Initialise the module, and the scalar "noise" parameters (sigmas in arxiv.org/abs/1705.07115).
        If using CUDA, requires manually setting them on the device, even if the model is already set to device.
        """
        super(MultiNoiseLoss, self).__init__()
        
        if torch.cuda.is_available():
            self.noise_params = torch.rand(n_losses, requires_grad=True, device="cuda:0")
        else:
            self.noise_params = torch.rand(n_losses, requires_grad=True)
    
    def forward(self, losses):
        """
        Computes the total loss as a function of a list of classification losses.
        TODO: Handle regressions losses, which require a factor of 2 (see arxiv.org/abs/1705.07115 page 4)
        """
        
        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss += (1/torch.square(self.noise_params[i]))*loss + torch.log(self.noise_params[i])
        
        return total_loss
    


#===============================================================================================================#


class CrossMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, add_zero_attn=False, kdim=None, vdim=None):
        super(CrossMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.add_zero_attn = add_zero_attn

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0.)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.constant_(self.k_proj.bias, 0.)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.constant_(self.v_proj.bias, 0.)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.constant_(self.o_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        batch_size, seq_length, _ = query.size()

        # Project input to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to (batch_size, num_heads, seq_length, head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if self.add_zero_attn:
            zero_attn_shape = (batch_size, self.num_heads, 1, self.head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=2)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # Expand to (batch_size, 1, seq_length, seq_length)
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask[:, None, None, :], 0)

        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_length, head_dim)

        # Concatenate attention output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.embed_dim)

        # Final linear layer
        output = self.o_proj(attn_output)

        return output, attn_weights
    

class CosineSimilarityMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, add_zero_attn=False, kdim=None, vdim=None,temperature=0.1):
        super(CosineSimilarityMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.add_zero_attn = add_zero_attn
        self.temperature=temperature

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0.)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.constant_(self.k_proj.bias, 0.)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.constant_(self.v_proj.bias, 0.)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.constant_(self.o_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        batch_size, seq_length, _ = query.size()

        # Project input to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to (batch_size, num_heads, seq_length, head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if self.add_zero_attn:
            zero_attn_shape = (batch_size, self.num_heads, 1, self.head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=2)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=2)

        # Cosine similarity attention
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        attn_weights = torch.matmul(q_norm, k_norm.transpose(-2, -1))/self.temperature

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # Expand to (batch_size, 1, seq_length, seq_length)
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask[:, None, None, :], 0)

        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_length, head_dim)

        # Concatenate attention output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.embed_dim)

        # Final linear layer
        output = self.o_proj(attn_output)
        
        return output, attn_weights


###############################################LayerNorm LSTM################################################
class LayerNormLSTMCell(nn.LSTMCell):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)

        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)

    def forward(self, input, hidden=None):
        if hidden is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) \
                 + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o = gates[:, :(3 * self.hidden_size)].sigmoid().chunk(3, 1)
        g = gates[:, (3 * self.hidden_size):].tanh()

        cy = (f * cx) + (i * g)
        hy = o * self.ln_ho(cy).tanh()
        return hy, cy
    

###################Deformable Conv##############################

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=0,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=0,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=0,  # Set padding to 0 since we are handling padding manually
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        # Apply padding manually using replicate padding
        if self.padding > 0:
            x = F.pad(x, [self.padding, self.padding, self.padding, self.padding], mode='replicate')

        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=0,  # No additional padding needed here
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x


#################MTRNN#######################
class MTRNNCell(nn.Module):
    #:: MTRNNCell
    """Multiple Timescale RNN.

    Implements a form of Recurrent Neural Network (RNN) that operates with multiple timescales.
    This is based on the idea of hierarchical organization in human cognitive functions.

    Arguments:
        input_dim (int): Number of input features.
        fast_dim (int): Number of fast context neurons.
        slow_dim (int): Number of slow context neurons.
        fast_tau (float): Time constant value of fast context.
        slow_tau (float): Time constant value of slow context.
        activation (string, optional): If you set `None`, no activation is applied (ie. "linear" activation: `a(x) = x`).
        use_bias (Boolean, optional): whether the layer uses a bias vector. The default is False.
        use_pb (Boolean, optional): whether the recurrent uses a pb vector. The default is False.

    Yuichi Yamashita, Jun Tani,
    "Emergence of functional hierarchy in a multiple timescale neural network model: a humanoid robot experiment." PLoS computational biology, 2008.
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000220
    """

    def __init__(
        self,
        input_dim,
        fast_dim,
        slow_dim,
        fast_tau,
        slow_tau,
        activation="tanh",
        use_bias=False,
        use_pb=False,
    ):
        super(MTRNNCell, self).__init__()

        self.input_dim = input_dim
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.fast_tau = fast_tau
        self.slow_tau = slow_tau
        self.use_bias = use_bias
        self.use_pb = use_pb

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = get_activation_fn(activation)
        else:
            self.activation = activation

        # Input Layers
        self.i2f = nn.Linear(input_dim, fast_dim, bias=use_bias)

        # Fast context layer
        self.f2f = nn.Linear(fast_dim, fast_dim, bias=False)
        self.f2s = nn.Linear(fast_dim, slow_dim, bias=use_bias)

        # Slow context layer
        self.s2s = nn.Linear(slow_dim, slow_dim, bias=False)
        self.s2f = nn.Linear(slow_dim, fast_dim, bias=use_bias)

    def forward(self, x, state=None, pb=None):
        """Forward propagation of the MTRNN.

        Arguments:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            state (list): Previous states (h_fast, h_slow, u_fast, u_slow), each of shape (batch_size, context_dim).
                   If None, initialize states to zeros.
            pb (bool): pb vector. Used if self.use_pb is set to True.

        Returns:
            new_h_fast (torch.Tensor): Updated fast context state.
            new_h_slow (torch.Tensor): Updated slow context state.
            new_u_fast (torch.Tensor): Updated fast internal state.
            new_u_slow (torch.Tensor): Updated slow internal state.
        """
        batch_size = x.shape[0]
        if state is not None:
            prev_h_fast, prev_h_slow, prev_u_fast, prev_u_slow = state
        else:
            device = x.device
            prev_h_fast = torch.zeros(batch_size, self.fast_dim).to(device)
            prev_h_slow = torch.zeros(batch_size, self.slow_dim).to(device)
            prev_u_fast = torch.zeros(batch_size, self.fast_dim).to(device)
            prev_u_slow = torch.zeros(batch_size, self.slow_dim).to(device)

        new_u_fast = (1.0 - 1.0 / self.fast_tau) * prev_u_fast + 1.0 / self.fast_tau * (
            self.i2f(x) + self.f2f(prev_h_fast) + self.s2f(prev_h_slow)
        )

        _input_slow = self.f2s(prev_h_fast) + self.s2s(prev_h_slow)
        if pb is not None:
            _input_slow += pb

        new_u_slow = (
            1.0 - 1.0 / self.slow_tau
        ) * prev_u_slow + 1.0 / self.slow_tau * _input_slow

        new_h_fast = self.activation(new_u_fast)
        new_h_slow = self.activation(new_u_slow)

        return new_h_fast, new_h_slow, new_u_fast, new_u_slow
'''
ChangeMamba: Remote Sensing Change Detection with Spatio-Temporal State Space Model
Hongruixuan Chen, Jian Song, Chengxi Han, Junshi Xia, Naoto Yokoya
https://github.com/ChenHongruixuan/ChangeMamba

'''

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from flooddamagesense.models.Mamba_backbone import Backbone_VSSM
from classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from .ChangeDecoder import ChangeDecoder
from .SemanticDecoder import SemanticDecoder


class STMambaBDA(nn.Module):
    def __init__(self, output_building, output_damage, output_map, pretrained, **kwargs):
        super(STMambaBDA, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs) 
        # summary(self.encoder.cuda(), (3, 256, 256))
        state_dict = self.encoder.state_dict()
        state_dict['patch_embed.0.weight'] = state_dict['patch_embed.0.weight'].mean(1, keepdim=True)
        state_dict['patch_embed.0.weight'] = state_dict['patch_embed.0.weight'].expand(-1, 4, -1, -1)
        self.encoder.patch_embed[0] = nn.Conv2d(4, 64, kernel_size=3, stride=3, bias=True)
        self.encoder.load_state_dict(state_dict)
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        self.channel_first = self.encoder.channel_first

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)


        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        
        self.decoder_damage = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_building = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_map = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.main_clf = nn.Conv2d(in_channels=128, out_channels=output_damage, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_building, kernel_size=1)
        self.map_clf = nn.Conv2d(in_channels=128, out_channels=output_map, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # Decoder processing - passing encoder outputs to the decoder
        output_building = self.decoder_building(pre_features)
        output_damage = self.decoder_damage(pre_features, post_features)
        output_map = self.decoder_map(pre_features, post_features)
        
        output_building = self.aux_clf(output_building)
        output_building = F.interpolate(output_building, size=pre_data.size()[-2:], mode='bilinear')

        output_damage = self.main_clf(output_damage)
        output_damage = F.interpolate(output_damage, size=post_data.size()[-2:], mode='bilinear')

        output_map = self.map_clf(output_map)
        output_map = F.interpolate(output_map, size=post_data.size()[-2:], mode='bilinear')
       
        return output_building, output_damage, output_map

from .Mamba_backbone import Backbone_VSSM
from classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute

import torch
import torch.nn as nn
import torch.nn.functional as F
from .FeatureFusionDecoder import FeatureFusionDecoder


class FFMambaBDA(nn.Module):
    def __init__(self, output_building, output_damage, output_map, pretrained, **kwargs):
        super(FFMambaBDA, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        encoder_state_dict = self.encoder.state_dict()
        # Extract the pretrained weights and bias from the patch embedding layer
        patch_embed_weight = encoder_state_dict['patch_embed.0.weight']
        patch_embed_bias = encoder_state_dict['patch_embed.0.bias']
        # Load pretrained weights into the new layers
        with torch.no_grad():
            # Define the new patch embedding layers for SAR and RGB data
            self.sar_patch_embed = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=3, bias=True)
            self.rgb_patch_embed = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=3, bias=True)
            # Directly copy weights for the RGB patch embedder since the encoder was pretrained on ImageNet
            self.rgb_patch_embed.weight.copy_(patch_embed_weight)
            self.rgb_patch_embed.bias.copy_(patch_embed_bias)        
            # Adapt pretrained weights for the 4-channel SAR patch embedder
            self.sar_patch_embed.weight.copy_(patch_embed_weight.mean(dim=1, keepdim=True).expand(-1, 4, -1, -1))
            self.sar_patch_embed.bias.copy_(patch_embed_bias)
        del self.encoder.patch_embed[0]


        self.prior_encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        prior_state_dict = self.prior_encoder.state_dict()
        prior_state_dict['patch_embed.0.weight'] = prior_state_dict['patch_embed.0.weight'].mean(1, keepdim=True)
        self.prior_encoder.patch_embed[0] = nn.Conv2d(1, 64, kernel_size=3, stride=3, bias=True)
        self.prior_encoder.load_state_dict(prior_state_dict)

        
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
        
        self.decoder_damage = FeatureFusionDecoder(
            num_features=4,
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_building = FeatureFusionDecoder(
            num_features=2,
            encoder_dims= self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_map = FeatureFusionDecoder(
            num_features=2,
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

    def forward(self, pre_data, post_data, pre_rgb, prior_data):
        # Encoder processing
        pre_rgb_embed = self.rgb_patch_embed(pre_rgb)
        pre_data_embed = self.sar_patch_embed(pre_data)
        post_data_embed = self.sar_patch_embed(post_data)

        pre_rgb_features = self.encoder(pre_rgb_embed)
        pre_features = self.encoder(pre_data_embed)
        post_features = self.encoder(post_data_embed)
        prior_features = self.prior_encoder(prior_data)

        # Decoder processing
        output_building = self.decoder_building(pre_features, pre_rgb_features)   
        output_damage = self.decoder_damage(pre_features, post_features, pre_rgb_features, prior_features)
        output_map = self.decoder_map(pre_features, post_features)
        
        output_building = self.aux_clf(output_building)
        output_building = F.interpolate(output_building, size=pre_data.size()[-2:], mode='bilinear')

        output_damage = self.main_clf(output_damage)
        output_damage = F.interpolate(output_damage, size=post_data.size()[-2:], mode='bilinear')

        output_map = self.map_clf(output_map)
        output_map = F.interpolate(output_map, size=post_data.size()[-2:], mode='bilinear')

        return output_building, output_damage, output_map

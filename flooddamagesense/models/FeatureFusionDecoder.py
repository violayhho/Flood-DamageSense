import torch
import torch.nn as nn
import torch.nn.functional as F
from classification.models.vmamba import VSSBlock, Permute

    
class FeatureFusionDecoder(nn.Module):
    def __init__(self, num_features, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(FeatureFusionDecoder, self).__init__()
        self.num_features = num_features
        self.encoder_dims = encoder_dims
        self.channel_first = channel_first
        self.norm_layer = norm_layer
        self.ssm_act_layer = ssm_act_layer
        self.mlp_act_layer = mlp_act_layer

        # Define which keys from **kwargs are specifically for VSSBlock
        expected_vss_kwargs_keys = [
            'ssm_d_state', 'ssm_ratio', 'ssm_dt_rank', 'ssm_conv',
            'ssm_conv_bias', 'ssm_drop_rate', 'ssm_init', 'forward_type',
            'mlp_ratio', 'mlp_drop_rate', 'gmlp', 'use_checkpoint'
        ]
        self.vss_kwargs = {}
        for key in expected_vss_kwargs_keys:
            if key in kwargs:
                self.vss_kwargs[key] = kwargs.pop(key)
            else:
                print(f"Warning: VSSBlock specific argument '{key}' not found in kwargs.")
        
        num_levels = len(self.encoder_dims)
        if num_levels == 0:
            raise ValueError("encoder_dims cannot be empty.")

        # Initialize ffss_blocks
        for i in range(num_levels):
            level_label = num_levels - i
            encoder_dim = self.encoder_dims[level_label-1] 
            
            # For ffss_parallel_block
            in_channels_parallel = encoder_dim * self.num_features
            setattr(self, f"ffss_parallel_block_{level_label}", self._create_ffss_block(in_channels_parallel))

            # For ffss_cross_block and ffss_sequential_block
            in_channels = encoder_dim
            setattr(self, f"ffss_cross_block_{level_label}", self._create_ffss_block(in_channels))
            setattr(self, f"ffss_sequential_block_{level_label}", self._create_ffss_block(in_channels))
        
        # Initialize fuse_layers
        self.fuse_layers = nn.ModuleList()
        for _ in range(num_levels):
            # Input channels to fuse_layer: 128 (parallel) + num_features * 128 (cross) + num_features * 128 (sequential)
            fuse_in_channels = 128 * (1 + 2 * self.num_features)
            self.fuse_layers.append(nn.Sequential(
                nn.Conv2d(kernel_size=1, in_channels=fuse_in_channels, out_channels=128),
                nn.BatchNorm2d(128), 
                nn.ReLU()
            ))

        # Initialize smooth_layers (N-1 layers, connecting adjacent levels)
        self.smooth_layers = nn.ModuleList()
        if num_levels > 1: 
            for _ in range(num_levels - 1):
                self.smooth_layers.append(ResBlock(in_channels=128, out_channels=128, stride=1))

    def _create_ffss_block(self, in_channels):
        vss_block_params = {
            'hidden_dim': 128,
            'drop_path': 0.1,
            'norm_layer': self.norm_layer,
            'channel_first': self.channel_first,
            'ssm_act_layer': self.ssm_act_layer,
            'mlp_act_layer': self.mlp_act_layer,
            **self.vss_kwargs
        }
        return nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=128),
            Permute(0, 2, 3, 1) if not self.channel_first else nn.Identity(),
            VSSBlock(**vss_block_params),
            Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity(),
        )

    
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, *features):
        num_levels = len(self.encoder_dims)
        o_prev_level = None

        for i in range(num_levels):
            level_label = num_levels - i
        
            # 1. Gather feature maps for the current level from all modalities
            features_cur_level = [feature[level_label-1] for feature in features]
            B, C, H, W = features_cur_level[0].size()

            # 2. Process with ffss_parallel_block
            ffss_parallel_block = getattr(self, f"ffss_parallel_block_{level_label}")
            o_parallel = ffss_parallel_block(torch.cat(features_cur_level, dim=1))

            # 3. Process with ffss_cross_block
            device = o_parallel.device
            tensor_cross = torch.empty(B, C, H, self.num_features * W, device=device)
            for idx, feature in enumerate(features_cur_level):
                tensor_cross[:, :, :, idx::self.num_features] = feature
            ffss_cross_block = getattr(self, f"ffss_cross_block_{level_label}")
            o_cross = ffss_cross_block(tensor_cross)

            # 4. Process with ffss_sequential_block
            ffss_sequential_block = getattr(self, f"ffss_sequential_block_{level_label}")
            o_sequential = ffss_sequential_block(torch.cat(features_cur_level, dim=3))

            # 5. Concatenate block outputs
            o_cat_list = [o_parallel]
            for idx in range(self.num_features):
                o_cat_list.append(o_cross[:, :, :, idx::self.num_features])
            for idx in range(self.num_features):
                o_cat_list.append(o_sequential[:, :, :, idx*W:(idx+1)*W])
            o_cat = torch.cat(o_cat_list, dim=1)

            # 6. Apply fuse_layer
            o_cur_level = self.fuse_layers[i](o_cat)

            # 7. Upsample-add from previous stage
            if o_prev_level is not None: # True for all but the deepest stage
                o_cur_level = self._upsample_add(o_prev_level, o_cur_level)
                o_cur_level = self.smooth_layers[i-1](o_cur_level)
            
            o_prev_level = o_cur_level
        
        return o_prev_level
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

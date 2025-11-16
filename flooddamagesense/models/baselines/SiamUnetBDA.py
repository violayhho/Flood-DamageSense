'''
xview2 1st place solution
Victor Durnov
https://github.com/vdurnov/xview2_1st_place_solution

'''

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models

from .senet import se_resnext50_32x4d, senet154
from .dpn import dpn92


class ConvReluBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvReluBN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)


class SCSEModule(nn.Module):
    # according to https://arxiv.org/pdf/1808.08127.pdf concat is better
    def __init__(self, channels, reduction=16, concat=False):
        super(SCSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

        self.spatial_se = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())
        self.concat = concat

    def forward(self, x):
        module_input = x

        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        chn_se = self.sigmoid(x)
        chn_se = chn_se * module_input

        spa_se = self.spatial_se(module_input)
        spa_se = module_input * spa_se
        if self.concat:
            return torch.cat([chn_se, spa_se], dim=1)
        else:
            return chn_se + spa_se

class SeResNext50_Unet_Encoder(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(SeResNext50_Unet_Encoder, self).__init__()
        self.encoder = se_resnext50_32x4d(pretrained=pretrained)

        state_dict = self.encoder.state_dict()
        state_dict['layer0.conv1.weight'] = state_dict['layer0.conv1.weight'].mean(1, keepdim=True)
        state_dict['layer0.conv1.weight'] = state_dict['layer0.conv1.weight'].expand(-1, 4, -1, -1)
        self.encoder.layer0.conv1 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
        self.encoder.load_state_dict(state_dict)

        self.conv1 = nn.Sequential(self.encoder.layer0.conv1, self.encoder.layer0.bn1, self.encoder.layer0.relu1) #encoder.layer0.conv1
        self.conv2 = nn.Sequential(self.encoder.pool, self.encoder.layer1)
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
    
    def forward(self, x):
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        return [enc1, enc2, enc3, enc4, enc5]

class SeResNext50_Unet_Decoder(nn.Module):
    def __init__(self, encoder_filters, decoder_filters):
        super(SeResNext50_Unet_Decoder, self).__init__()
        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
    
    def forward(self, enc):
        dec6 = self.conv6(F.interpolate(enc[4], scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc[3]], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc[2]], 1))
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc[1]], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, enc[0]], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return dec10


class SeResNext50_Unet_BDA(nn.Module):
    def __init__(self, output_building, output_damage, output_map, pretrained='imagenet'):
        super(SeResNext50_Unet_BDA, self).__init__()
        
        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.encoder = SeResNext50_Unet_Encoder(pretrained=pretrained)
        self.decoder_building = SeResNext50_Unet_Decoder(encoder_filters, decoder_filters)
        self.decoder_damage = SeResNext50_Unet_Decoder(encoder_filters, decoder_filters)
        self.decoder_map = SeResNext50_Unet_Decoder(encoder_filters, decoder_filters)        
        
        self.res_building = nn.Conv2d(decoder_filters[-5], output_building, 1, stride=1, padding=0)
        self.res_damage = nn.Conv2d(decoder_filters[-5] * 2, output_damage, 1, stride=1, padding=0)
        self.res_map = nn.Conv2d(decoder_filters[-5] * 2, output_map, 1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, pre_data, post_data):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        # print([pre_features[i].shape for i in range(5)])
        dec10_building = self.decoder_building(pre_features)
        dec10_damage_0 = self.decoder_damage(pre_features)
        dec10_damage_1 = self.decoder_damage(post_features)
        dec10_damage = torch.cat([dec10_damage_0, dec10_damage_1], 1)
        dec10_map_0 = self.decoder_map(pre_features)
        dec10_map_1 = self.decoder_map(post_features)
        dec10_map = torch.cat([dec10_map_0, dec10_map_1], 1)

        output_building = self.res_building(dec10_building)
        output_damage = self.res_damage(dec10_damage)
        output_map = self.res_map(dec10_map)

        return output_building, output_damage, output_map


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Dpn92_Unet_Encoder(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(Dpn92_Unet_Encoder, self).__init__()
        self.encoder = dpn92(pretrained=pretrained)

        state_dict = self.encoder.state_dict()
        state_dict['features.conv1_1.conv.weight'] = state_dict['features.conv1_1.conv.weight'].sum(1, keepdim=True)
        state_dict['features.conv1_1.conv.weight'] = state_dict['features.conv1_1.conv.weight'].expand(-1, 4, -1, -1)
        self.encoder.blocks['conv1_1'].conv = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
        self.encoder.load_state_dict(state_dict)
        
        self.conv1 = nn.Sequential(
                self.encoder.blocks['conv1_1'].conv,  # conv
                self.encoder.blocks['conv1_1'].bn,  # bn
                self.encoder.blocks['conv1_1'].act,  # relu
            )
        self.conv2 = nn.Sequential(
                self.encoder.blocks['conv1_1'].pool,  # maxpool
                *[b for k, b in self.encoder.blocks.items() if k.startswith('conv2_')]
            )
        self.conv3 = nn.Sequential(*[b for k, b in self.encoder.blocks.items() if k.startswith('conv3_')])
        self.conv4 = nn.Sequential(*[b for k, b in self.encoder.blocks.items() if k.startswith('conv4_')])
        self.conv5 = nn.Sequential(*[b for k, b in self.encoder.blocks.items() if k.startswith('conv5_')])
    
    def forward(self, x):
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        enc1 = (torch.cat(enc1, dim=1) if isinstance(enc1, tuple) else enc1)
        enc2 = (torch.cat(enc2, dim=1) if isinstance(enc2, tuple) else enc2)
        enc3 = (torch.cat(enc3, dim=1) if isinstance(enc3, tuple) else enc3)
        enc4 = (torch.cat(enc4, dim=1) if isinstance(enc4, tuple) else enc4)
        enc5 = (torch.cat(enc5, dim=1) if isinstance(enc5, tuple) else enc5)

        return [enc1, enc2, enc3, enc4, enc5]

class Dpn92_Unet_Decoder(nn.Module):
    def __init__(self, encoder_filters, decoder_filters):
        super(Dpn92_Unet_Decoder, self).__init__()
        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = nn.Sequential(ConvRelu(decoder_filters[-1]+encoder_filters[-2], decoder_filters[-1]), SCSEModule(decoder_filters[-1], reduction=16, concat=True))
        self.conv7 = ConvRelu(decoder_filters[-1] * 2, decoder_filters[-2])
        self.conv7_2 = nn.Sequential(ConvRelu(decoder_filters[-2]+encoder_filters[-3], decoder_filters[-2]), SCSEModule(decoder_filters[-2], reduction=16, concat=True))
        self.conv8 = ConvRelu(decoder_filters[-2] * 2, decoder_filters[-3])
        self.conv8_2 = nn.Sequential(ConvRelu(decoder_filters[-3]+encoder_filters[-4], decoder_filters[-3]), SCSEModule(decoder_filters[-3], reduction=16, concat=True))
        self.conv9 = ConvRelu(decoder_filters[-3] * 2, decoder_filters[-4])
        self.conv9_2 = nn.Sequential(ConvRelu(decoder_filters[-4]+encoder_filters[-5], decoder_filters[-4]), SCSEModule(decoder_filters[-4], reduction=16, concat=True))
        self.conv10 = ConvRelu(decoder_filters[-4] * 2, decoder_filters[-5])
    
    def forward(self, enc):
        dec6 = self.conv6(F.interpolate(enc[4], scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc[3]], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc[2]], 1))

        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc[1]], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, enc[0]], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return dec10

class Dpn92_Unet_BDA(nn.Module):
    def __init__(self, output_building, output_damage, output_map, pretrained='imagenet+5k', **kwargs):
        super(Dpn92_Unet_BDA, self).__init__()
        
        encoder_filters = [64, 336, 704, 1552, 2688]
        decoder_filters = np.asarray([64, 96, 128, 256, 512]) // 2

        self.encoder = Dpn92_Unet_Encoder(pretrained=pretrained)
        self.decoder_building = Dpn92_Unet_Decoder(encoder_filters, decoder_filters)
        self.decoder_damage = Dpn92_Unet_Decoder(encoder_filters, decoder_filters)
        self.decoder_map = Dpn92_Unet_Decoder(encoder_filters, decoder_filters)        
        
        self.res_building = nn.Conv2d(decoder_filters[-5], output_building, 1, stride=1, padding=0)
        self.res_damage = nn.Conv2d(decoder_filters[-5] * 2, output_damage, 1, stride=1, padding=0)
        self.res_map = nn.Conv2d(decoder_filters[-5] * 2, output_map, 1, stride=1, padding=0)
        
        self._initialize_weights()
    
    def forward(self, pre_data, post_data):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        # print([pre_features[i].shape for i in range(5)])
        dec10_building = self.decoder_building(pre_features)
        dec10_damage_0 = self.decoder_damage(pre_features)
        dec10_damage_1 = self.decoder_damage(post_features)
        dec10_damage = torch.cat([dec10_damage_0, dec10_damage_1], 1)
        dec10_map_0 = self.decoder_map(pre_features)
        dec10_map_1 = self.decoder_map(post_features)
        dec10_map = torch.cat([dec10_map_0, dec10_map_1], 1)

        output_building = self.res_building(dec10_building)
        output_damage = self.res_damage(dec10_damage)
        output_map = self.res_map(dec10_map)

        return output_building, output_damage, output_map

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Res34_Unet_Encoder(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(Res34_Unet_Encoder, self).__init__()
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        state_dict = self.encoder.state_dict()
        state_dict['conv1.weight'] = state_dict['conv1.weight'].sum(1, keepdim=True)
        state_dict['conv1.weight'] = state_dict['conv1.weight'].expand(-1, 4, -1, -1)
        self.encoder.conv1 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
        self.encoder.load_state_dict(state_dict)
        
        self.conv1 = nn.Sequential(
                        self.encoder.conv1,
                        self.encoder.bn1,
                        self.encoder.relu)
        self.conv2 = nn.Sequential(
                        self.encoder.maxpool,
                        self.encoder.layer1)
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
    
    def forward(self, x):
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        return [enc1, enc2, enc3, enc4, enc5]

class Res34_Unet_Decoder(nn.Module):
    def __init__(self, encoder_filters, decoder_filters):
        super(Res34_Unet_Decoder, self).__init__()
        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
    
    def forward(self, enc):
        dec6 = self.conv6(F.interpolate(enc[4], scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc[3]], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc[2]], 1))
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc[1]], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, enc[0]], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return dec10

class Res34_Unet_BDA(nn.Module):
    def __init__(self, output_building, output_damage, output_map, pretrained=True, **kwargs):
        super(Res34_Unet_BDA, self).__init__()
        
        encoder_filters = [64, 64, 128, 256, 512]
        decoder_filters = np.asarray([48, 64, 96, 160, 320])

        self.encoder = Res34_Unet_Encoder(pretrained=pretrained)
        self.decoder_building = Res34_Unet_Decoder(encoder_filters, decoder_filters)
        self.decoder_damage = Res34_Unet_Decoder(encoder_filters, decoder_filters)
        self.decoder_map = Res34_Unet_Decoder(encoder_filters, decoder_filters)        
        
        self.res_building = nn.Conv2d(decoder_filters[-5], output_building, 1, stride=1, padding=0)
        self.res_damage = nn.Conv2d(decoder_filters[-5] * 2, output_damage, 1, stride=1, padding=0)
        self.res_map = nn.Conv2d(decoder_filters[-5] * 2, output_map, 1, stride=1, padding=0)

        self._initialize_weights()
    
    def forward(self, pre_data, post_data):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        # print([pre_features[i].shape for i in range(5)])
        dec10_building = self.decoder_building(pre_features)
        dec10_damage_0 = self.decoder_damage(pre_features)
        dec10_damage_1 = self.decoder_damage(post_features)
        dec10_damage = torch.cat([dec10_damage_0, dec10_damage_1], 1)
        dec10_map_0 = self.decoder_map(pre_features)
        dec10_map_1 = self.decoder_map(post_features)
        dec10_map = torch.cat([dec10_map_0, dec10_map_1], 1)

        output_building = self.res_building(dec10_building)
        output_damage = self.res_damage(dec10_damage)
        output_map = self.res_map(dec10_map)

        return output_building, output_damage, output_map
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SeNet154_Unet_Encoder(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(SeNet154_Unet_Encoder, self).__init__()
        self.encoder = senet154(pretrained=pretrained)

        state_dict = self.encoder.state_dict()
        state_dict['layer0.conv1.weight'] = state_dict['layer0.conv1.weight'].sum(1, keepdim=True)
        state_dict['layer0.conv1.weight'] = state_dict['layer0.conv1.weight'].expand(-1, 4, -1, -1)
        self.encoder.layer0.conv1 = nn.Conv2d(4, 64, 3, stride=2, padding=1, bias=False)
        self.encoder.load_state_dict(state_dict)

        self.conv1 = nn.Sequential(self.encoder.layer0.conv1, self.encoder.layer0.bn1, self.encoder.layer0.relu1, self.encoder.layer0.conv2, self.encoder.layer0.bn2, self.encoder.layer0.relu2, self.encoder.layer0.conv3, self.encoder.layer0.bn3, self.encoder.layer0.relu3)
        self.conv2 = nn.Sequential(self.encoder.pool, self.encoder.layer1)
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
    
    def forward(self, x):
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        return [enc1, enc2, enc3, enc4, enc5]

class SeNet154_Unet_Decoder(nn.Module):
    def __init__(self, encoder_filters, decoder_filters):
        super(SeNet154_Unet_Decoder, self).__init__()
        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
    
    def forward(self, enc):
        dec6 = self.conv6(F.interpolate(enc[4], scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc[3]], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc[2]], 1))
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc[1]], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, enc[0]], 1))

        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))

        return dec10

class SeNet154_Unet_BDA(nn.Module):
    def __init__(self, output_building, output_damage, output_map, pretrained='imagenet', **kwargs):
        super(SeNet154_Unet_BDA, self).__init__()
        
        encoder_filters = [128, 256, 512, 1024, 2048]
        decoder_filters = np.asarray([48, 64, 96, 160, 320])

        self.encoder = SeNet154_Unet_Encoder(pretrained=pretrained)
        self.decoder_building = SeNet154_Unet_Decoder(encoder_filters, decoder_filters)
        self.decoder_damage = SeNet154_Unet_Decoder(encoder_filters, decoder_filters)
        self.decoder_map = SeNet154_Unet_Decoder(encoder_filters, decoder_filters)        
        
        self.res_building = nn.Conv2d(decoder_filters[-5], output_building, 1, stride=1, padding=0)
        self.res_damage = nn.Conv2d(decoder_filters[-5] * 2, output_damage, 1, stride=1, padding=0)
        self.res_map = nn.Conv2d(decoder_filters[-5] * 2, output_map, 1, stride=1, padding=0)

        self._initialize_weights()

    def forward(self, pre_data, post_data):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        # print([pre_features[i].shape for i in range(5)])
        dec10_building = self.decoder_building(pre_features)
        dec10_damage_0 = self.decoder_damage(pre_features)
        dec10_damage_1 = self.decoder_damage(post_features)
        dec10_damage = torch.cat([dec10_damage_0, dec10_damage_1], 1)
        dec10_map_0 = self.decoder_map(pre_features)
        dec10_map_1 = self.decoder_map(post_features)
        dec10_map = torch.cat([dec10_map_0, dec10_map_1], 1)

        output_building = self.res_building(dec10_building)
        output_damage = self.res_damage(dec10_damage)
        output_map = self.res_map(dec10_map)

        return output_building, output_damage, output_map

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
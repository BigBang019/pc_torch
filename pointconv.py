import torch
import torch.nn as nn
import numpy as np

from model.pointconv.pointconv_utils import *
from model.base import *

class WeightNet(nn.Module):
    def __init__(self, mlp_channels=[64, 64, 64]):
        super().__init__()
        in_channel = 3
        self.mlps = nn.Sequential()
        for idx, channel in enumerate(mlp_channels):
            self.mlps.append(
                nn.Conv2d(in_channels=in_channel, out_channels=channel, kernel_size=1) \
                    if idx == len(mlp_channels) - 1 else \
                ConvBNRelu2D(in_channels=in_channel, out_channels=channel, kernel_size=1)
            )
            in_channel = channel

    def forward(self, offset):
        return self.mlps(offset)

class PointConv(nn.Module):
    def __init__(self, in_channel, out_channel=128, 
                npoints=256, nsample=32, with_relu=True):
        super().__init__()
        self.npoints = npoints
        self.nsample = nsample

        self.grouper = QueryAndGroup(radius=0.4, nsample=self.nsample, ret_grouped_xyz=True)
        
        weight_mlp_channels = [64,64,64]
        self.weight_net = WeightNet(mlp_channels=weight_mlp_channels)
        self.linear = ConvBNRelu1D(in_channel * weight_mlp_channels[-1], out_channel, kernel_size=1)
        
        self.final = ConvBNRelu1D(in_channels=in_channel + out_channel, out_channels=out_channel, kernel_size=1) \
                        if with_relu else \
                    nn.Conv1d(in_channels=in_channel + out_channel, out_channels=out_channel, kernel_size=1)
        


    def forward(self, xyz, feature):
        '''
            xyz: (B, N, 3)
            feature: (B, D, N)
        '''
        B, N, _ = xyz.shape
        xyz_trans = xyz.permute(0, 2, 1).contiguous()           # [B, 3, N]
        '''
            furthest sampling
        '''
        new_xyz_trans = gather_operation(
            xyz_trans, furthest_point_sample(xyz, self.npoints)
        )                                                       # [B, 3, npoints]
        new_xyz = new_xyz_trans.permute(0, 2, 1).contiguous()   # [B, npoints, 3]
        new_feature, grouped_xyz_trans = self.grouper(xyz, new_xyz, feature)  # [B, D, npoints, nsample] [B, 3, npoints, nsample]
        weights = self.weight_net(grouped_xyz_trans)                          # [B, D', npoints, nsample]
        new_feature = torch.matmul(
            new_feature.permute(0, 2, 1, 3),      # [B, npoints, D, nsample]
            weights.permute(0, 2, 3, 1)           # [B, npoints, nsample, D']
        ).view(B, self.npoints, -1)               # [B, npoints, D*D']
        new_feature = self.linear(new_feature).permute(0, 2, 1).contiguous() #[B, D, npoints]
        '''
            propogate
                new_xyz, new_feature --> xyz, feature
        '''
        dist, idx = three_nn(xyz, new_xyz)
        reverse_density = 1.0 / (dist + 1e-8)
        norm = torch.sum(reverse_density, dim=2, keepdim=True)
        reverse_density = reverse_density / norm

        interpolated_feature = three_interpolate(new_feature, idx, reverse_density) # [B, D, N]
        if feature is not None:
            concat_feature = torch.cat([interpolated_feature, feature], dim=1)         # [B, D+D, N]
        else:
            concat_feature = torch.cat([interpolated_feature, xyz_trans], dim=1)       # [B, D+3, N]

        return self.final(concat_feature)


class PConv(nn.Module):
    def __init__(self, in_channel, out_channel, 
                nsample=9, with_relu=True):
        super().__init__()
        self.nsample = nsample
        weight_mlp = [64, 64, 64]
        self.weight_net = WeightNet(weight_mlp)
        self.final = ConvBNRelu1D(in_channel * weight_mlp[-1], out_channels=out_channel, kernel_size=1)\
                    if with_relu else \
                    nn.Conv1d(in_channel * weight_mlp[-1], out_channels=out_channel, kernel_size=1)
    def forward(self, xyz, feature, idx=None):
        B, _, N = feature.shape
        xyz_trans = xyz.permute(0, 2, 1).contiguous()
        if len(idx.shape)!=3:
            idx = k_neighbor_query(xyz, xyz, self.nsample)
        
        grouped_xyz_trans = grouping_operation(
            xyz_trans, idx
        ) - xyz_trans.view(B, 3, N, 1) # [B, 3, N, nsample]
        feature = grouping_operation(
            feature, idx
        ) # [B, D, N, nsample]

        weights = self.weight_net(grouped_xyz_trans) # [B, D1, nsample, N]
        feature = torch.matmul(
            feature.permute(0, 2, 1, 3),
            weights.permute(0, 2, 3, 1)
        ).view(B, N, -1)
        return self.final(feature.permute(0, 2, 1))

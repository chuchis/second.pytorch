"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn
from torch.nn import functional as F

from second.pytorch.models.voxel_encoder import get_paddings_indicator, register_vfe
from second.pytorch.models.middle import register_middle
from torchplus.nn import Empty
from torchplus.tools import change_default_args
import numpy as np 

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size, num_dims, num_points = x.size()
    device = torch.device('cuda')
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)
    feature = feature[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
    # print(feature.shape)
    return feature

def batch_process(input, fun, num_batches=5):
    num_data = input.shape[0]
    data_per_batch = np.ceil(num_data/num_batches).astype(int)
    for i in range(num_batches):
        if i == 0:
            out = fun(input[:data_per_batch])
        else:
            start = data_per_batch*i
            end = min(data_per_batch*(i+1), num_data)
            out = torch.cat((out,fun(input[start:end])), axis=0)
        # print(out.shape)
    return out

class GCNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'GCNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        # if use_norm:
        #     BatchNorm2d = change_default_args(
        #         eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
        #     Conv2d = change_default_args(kernel_size=1, bias=False)(nn.Conv2d)
        # else:
        #     BatchNorm2d = Empty
        #     Conv2d = change_default_args(kernel_size=1, bias=True)(nn.Conv2d)

        # LeakyReLU = change_default_args(negative_slope=0.2)(nn.LeakyReLU)
        # self.conv = Conv2d(in_channels, self.units)
        # self.norm = BatchNorm2d(self.units)
        # self.relu = LeakyReLU()
        self.seq = nn.Sequential(nn.Conv2d(in_channels, self.units,kernel_size=1, bias=False),
                                 nn.BatchNorm2d(self.units, eps=1e-3, momentum=0.01),
                                 nn.LeakyReLU(negative_slope=0.2))
        self.k = 8

    def forward(self, inputs):
        x = get_graph_feature(inputs.transpose(1,2), k=self.k)
        x = self.seq(x)
        x = x.max(dim=-1, keepdim=False)[0].transpose(1,2)
        return x

@register_vfe
class DeepGCNFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64, ),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'DeepGCNFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                GCNLayer(
                    in_filters*2, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        # print(torch.sum(num_voxels)/2)
        device = features.device

        dtype = features.dtype
        # print(features.shape, num_voxels.shape, coors.shape)
        # print(coors[0])
        # print(num_voxels[0])
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features_out = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features_out.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features_out)
        # print(torch.sum(features[:,:,:3] - (features*mask)[:,:,:3]))
        features_out = features_out * mask
        # Forward pass through PFNLayers
        prev_features_out = torch.zeros_like(mask)
        for pfn in self.pfn_layers:
            features_out = batch_process(features_out, pfn, num_batches=10)
            features_out = features_out * mask
            features_out = features_out + prev_features_out
            features_max = torch.max(features_out, dim=1, keepdim=True)[0]
            if pfn.last_vfe:
                features_out = features_max
            else:
                prev_features_out = features_out
                features_repeat = features_max.repeat(1, features_out.shape[1], 1)
                features_out = torch.cat([features_out, features_repeat], dim=2)
        return features_out.squeeze() 

@register_vfe
class PillarGCNFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64, ),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarGCNFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                GCNLayer(
                    in_filters*2, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        # print(torch.sum(num_voxels)/2)
        device = features.device

        dtype = features.dtype
        # print(features.shape, num_voxels.shape, coors.shape)
        # print(coors[0])
        # print(num_voxels[0])
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features_out = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features_out.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features_out)
        # print(torch.sum(features[:,:,:3] - (features*mask)[:,:,:3]))
        features_out = features_out * mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            # print(features.shape)
            # features = batch_process(features, pfn, num_batches=20)
            features_out = batch_process(features_out, pfn, num_batches=10)
            # features_out = pfn(features_out)
            # print(features.grad_fn)
            # print(mask)
            features_out = features_out * mask
            # print(mask.shape, features.shape)
            features_max = torch.max(features_out, dim=1, keepdim=True)[0]
            if pfn.last_vfe:
                features_out = features_max
            else:
                # features_max = torch.max(x, dim=1, keepdim=True)[0]
                features_repeat = features_max.repeat(1, features_out.shape[1], 1)
                features_out = torch.cat([features_out, features_repeat], dim=2)
            # print(features.shape)
        return features_out.squeeze() 
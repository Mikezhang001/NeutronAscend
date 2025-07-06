# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Dataset"""
from math import floor
import numpy as np
from mindspore_gl.sampling.neighbor import sage_sampler_on_homo
from mindspore_gl.dataloader.dataset import Dataset
import mindspore_gl.array_kernel as array_kernel
from mindspore_gl.graph.ops import PadArray2d, PadMode, PadDirection


class GCNDataset(Dataset):
    """Do sampling from neighbour nodes"""
    def __init__(self, graph_dataset, neighbor_nums, batch_size, length, single_size=False):
        self.graph_dataset = graph_dataset
        self.graph = graph_dataset[0]
        self.neighbor_nums = neighbor_nums
        self.x = graph_dataset.node_feat
        self.y = graph_dataset.node_label
        self.batch_size = batch_size
        self.max_sampled_nodes_num = neighbor_nums[0] * neighbor_nums[1] * batch_size
        self.single_size = single_size
        self.length = length

    def __getitem__(self, batch_nodes):
        batch_nodes = np.array(batch_nodes, np.int32)
        res = sage_sampler_on_homo(self.graph, batch_nodes, self.neighbor_nums)
        label = array_kernel.int_1d_array_slicing(self.y, batch_nodes)
        layered_edges_0 = res['layered_edges_0']
        layered_edges_1 = res['layered_edges_1']
        sample_edges = np.concatenate((layered_edges_0, layered_edges_1), axis=1)
        sample_edges = sample_edges[[1, 0], :]
        num_sample_edges = sample_edges.shape[1]
        num_sample_nodes = len(res['all_nodes'])
        max_sampled_nodes_num = self.max_sampled_nodes_num
        if self.single_size is False:
            if num_sample_nodes < floor(0.2*max_sampled_nodes_num):
                pad_node_num = floor(0.2*max_sampled_nodes_num)
            elif num_sample_nodes < floor(0.4*max_sampled_nodes_num):
                pad_node_num = floor(0.4 * max_sampled_nodes_num)
            elif num_sample_nodes < floor(0.6*max_sampled_nodes_num):
                pad_node_num = floor(0.6 * max_sampled_nodes_num)
            elif num_sample_nodes < floor(0.8*max_sampled_nodes_num):
                pad_node_num = floor(0.8 * max_sampled_nodes_num)
            else:
                pad_node_num = max_sampled_nodes_num

            if num_sample_edges < floor(0.2*max_sampled_nodes_num):
                pad_edge_num = floor(0.2*max_sampled_nodes_num)
            elif num_sample_edges < floor(0.4*max_sampled_nodes_num):
                pad_edge_num = floor(0.4 * max_sampled_nodes_num)
            elif num_sample_edges < floor(0.6*max_sampled_nodes_num):
                pad_edge_num = floor(0.6 * max_sampled_nodes_num)
            elif num_sample_edges < floor(0.8*max_sampled_nodes_num):
                pad_edge_num = floor(0.8 * max_sampled_nodes_num)
            else:
                pad_edge_num = max_sampled_nodes_num

        else:
            pad_node_num = max_sampled_nodes_num
            pad_edge_num = max_sampled_nodes_num

        layered_edges_pad_op = PadArray2d(mode=PadMode.CONST, size=[2, pad_edge_num],
                                          dtype=np.int32, direction=PadDirection.ROW,
                                          fill_value=pad_node_num - 1,
                                          )
        nid_feat_pad_op = PadArray2d(mode=PadMode.CONST,
                                     size=[pad_node_num, self.graph_dataset.node_feat_size],
                                     dtype=self.graph_dataset.node_feat.dtype,
                                     direction=PadDirection.COL,
                                     fill_value=0,
                                     reset_with_fill_value=False,
                                     use_shared_numpy=True
                                     )
        sample_edges = sample_edges[:, :pad_edge_num]
        pad_sample_edges = layered_edges_pad_op(sample_edges)
        feat = nid_feat_pad_op.lazy([num_sample_nodes, self.graph_dataset.node_feat_size])
        array_kernel.float_2d_gather_with_dst(feat, self.graph_dataset.node_feat, res['all_nodes'])
        return res['seeds_idx'], label, feat, pad_sample_edges

    def __len__(self):
        return self.length

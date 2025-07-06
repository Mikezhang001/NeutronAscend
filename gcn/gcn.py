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
"""gcn"""
import mindspore as ms
import os
import sys

from mindspore_gl import Graph
import mindspore.nn as nn
from mindspore_gl.nn import GNNCell


# sys.path.append(os.path.join(os.getcwd(), "gcn"))
from .gcnconv import GCNConv 


class GCNNet(GNNCell):
    """ GCN Net """

    def __init__(self,
                 data_feat_size: int,
                 hidden_dim_size: int,
                 n_classes: int,
                 BLK_H, batch_edgecount, n_node, split_col_remaining_csr_indptr,
                 split_col_remaining_coo_cols, split_col_filtered_coo_rows, split_col_filtered_coo_cols, split_row_filtered_coo_rows,
                 split_row_filtered_coo_cols, adj_matrix, edgeToRow_ms_tensor, row_reindex, row_index,
                 dropout: float,
                 activation: ms.nn.Cell = None,
                 num_layers: int = 2, 
                 aicore_num = None):
        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.CellList()

        # 构建每一层
        for i in range(num_layers):
            if i == 0:
                in_dim = data_feat_size
                out_dim = hidden_dim_size
            elif i == num_layers - 1:
                in_dim = hidden_dim_size
                out_dim = n_classes
            else:
                in_dim = hidden_dim_size
                out_dim = hidden_dim_size

            # 每一层都使用相同的参数配置（可根据需要定制）
            gcn_layer = GCNConv(in_dim, out_dim,
                                BLK_H=BLK_H,
                                batch_edgecount=batch_edgecount,
                                n_node=n_node,
                                split_col_remaining_csr_indptr=split_col_remaining_csr_indptr,
                                split_col_remaining_coo_cols=split_col_remaining_coo_cols,
                                split_col_filtered_coo_rows=split_col_filtered_coo_rows,
                                split_col_filtered_coo_cols=split_col_filtered_coo_cols,
                                split_row_filtered_coo_rows=split_row_filtered_coo_rows,
                                split_row_filtered_coo_cols=split_row_filtered_coo_cols,
                                adj_matrix=adj_matrix,
                                edgeToRow_ms_tensor=edgeToRow_ms_tensor,
                                row_reindex=row_reindex,
                                row_index=row_index,
                                activation=activation() if (i != num_layers - 1 and activation is not None) else None,
                                dropout=dropout, aicore_num=aicore_num)
            self.layers.append(gcn_layer)

    def construct(self, x, in_deg, out_deg, g: Graph):
        """GCN Net forward"""
        for layer in self.layers:
            x = layer(x, in_deg, out_deg, g)
        return x
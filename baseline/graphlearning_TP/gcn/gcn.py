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
from mindspore import nn
from mindspore_gl import Graph
from mindspore_gl.nn import GNNCell
from .gcnconv import GCNConv


class GCNNet(GNNCell):
    """GCN Net"""

    def __init__(self,
                 data_feat_size: int,
                 hidden_dim_size: int,
                 n_classes: int,
                 BLK_H,
                 batch_edgecount,
                 n_node,
                 split_col_remaining_csr_indptr,
                 split_col_remaining_coo_cols,
                 split_col_filtered_coo_rows,
                 split_col_filtered_coo_cols,
                 split_row_filtered_coo_rows,
                 split_row_filtered_coo_cols,
                 adj_matrix,
                 edgeToRow_ms_tensor,
                 row_reindex,
                 row_index,
                 dropout: float,
                 activation: ms.nn.Cell = None,
                 num_layers: int = 2):  # 可配置层数
        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.CellList()

        for i in range(num_layers):
            if i == 0:
                # 输入层 -> 隐藏层
                self.layers.append(
                    GCNConv(data_feat_size,
                            hidden_dim_size,
                            BLK_H,
                            batch_edgecount,
                            n_node,
                            split_col_remaining_csr_indptr,
                            split_col_remaining_coo_cols,
                            split_col_filtered_coo_rows,
                            split_col_filtered_coo_cols,
                            split_row_filtered_coo_rows,
                            split_row_filtered_coo_cols,
                            adj_matrix,
                            edgeToRow_ms_tensor,
                            row_reindex,
                            row_index,
                            activation(),
                            dropout)
                )
            elif i < num_layers - 1:
                # 隐藏层 -> 隐藏层
                self.layers.append(
                    GCNConv(hidden_dim_size,
                            hidden_dim_size,
                            BLK_H,
                            batch_edgecount,
                            n_node,
                            split_col_remaining_csr_indptr,
                            split_col_remaining_coo_cols,
                            split_col_filtered_coo_rows,
                            split_col_filtered_coo_cols,
                            split_row_filtered_coo_rows,
                            split_row_filtered_coo_cols,
                            adj_matrix,
                            edgeToRow_ms_tensor,
                            row_reindex,
                            row_index,
                            activation(),
                            dropout)
                )
            else:
                # 隐藏层 -> 输出层
                self.layers.append(
                    GCNConv(hidden_dim_size,
                            n_classes,
                            BLK_H,
                            batch_edgecount,
                            n_node,
                            split_col_remaining_csr_indptr,
                            split_col_remaining_coo_cols,
                            split_col_filtered_coo_rows,
                            split_col_filtered_coo_cols,
                            split_row_filtered_coo_rows,
                            split_row_filtered_coo_cols,
                            adj_matrix,
                            edgeToRow_ms_tensor,
                            row_reindex,
                            row_index,
                            None,
                            dropout)
                )

    def construct(self, x, in_deg, out_deg, g: Graph):
        """GCN Net forward"""
        for layer in self.layers:
            x = layer(x, in_deg, out_deg, g)
        return x
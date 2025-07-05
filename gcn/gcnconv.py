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
"""GCNConv Layer"""
import mindspore as ms
from mindspore import Tensor, ops
from mindspore.common.initializer import initializer
from mindspore.common.initializer import XavierUniform
from mindspore.nn.cell import Cell
import inspect
from mindspore.ops import MatMul

# print("test", inspect.getfile(MatMul))
from mindspore_gl import Graph
from mindspore_gl.nn import GNNCell
import time
import logging
import numpy as np


from mindspore.ops import DataType, CustomRegOp
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype


class MmadCustomAclnnNet(Cell):
    """
    MmadCustomAclnnNet is a custom operator for matrix multiplication on Ascend devices.
    It uses the `CustomRegOp` API to register the operator and define its behavior.

    Attributes:
        op (ops.Custom): The registered custom operator for matrix multiplication.
    """

    def __init__(self):
        """
        Initializes the MmadCustomAclnnNet class and registers the custom operator.
        """
        super(MmadCustomAclnnNet, self).__init__()
        
        # Register the custom operator using CustomRegOp
        aclnn_ref_info = (
            CustomRegOp("aclnnMmadCustom")  # Operator name
            .input(0, "x", "required")  # First input tensor
            .input(1, "y", "required")  # Second input tensor
            .output(0, "z", "required")  # Output tensor
            .dtype_format(
                DataType.F16_Default, DataType.F16_Default, DataType.F32_Default
            )  # Data type format: input tensors are FP16, output tensor is FP32
            .target("Ascend")  # Target device: Ascend
            .get_op_info()  # Get operator information
        )
        
        # Define the custom operator
        self.op = ops.Custom(
            "aclnnMmadCustom",  # Operator name
            lambda x_shape, y_shape: (x_shape[0], y_shape[1]),  # Output shape calculation
            out_dtype=ms.float32,  # Output data type
            func_type="aot",  # Function type: ahead-of-time compilation
            bprop=None,  # Backpropagation function (not defined here)
            reg_info=aclnn_ref_info,  # Registered operator information
        )

    def construct(self, x1, x2):
        """
        Forward computation for the custom operator.

        Args:
            x1 (Tensor): The first input tensor.
            x2 (Tensor): The second input tensor.

        Returns:
            Tensor: The result of the matrix multiplication.
        """
        return self.op(x1, x2)

    def bprop(self, x1, x2, out, dout):
        """
        Backpropagation computation for the custom operator.

        Args:
            x1 (Tensor): The first input tensor.
            x2 (Tensor): The second input tensor.
            out (Tensor): The output tensor from the forward computation.
            dout (Tensor): The gradient of the output tensor.

        Returns:
            Tuple[Tensor, Tensor]: Gradients of the input tensors.
        """
        # Convert the gradient of the output tensor to FP16
        dout_fp16 = dout.astype(ms.float16)
        
        # Compute the gradient for the first input tensor
        dx1 = self.op(dout_fp16, x2.T).astype(ms.float16)
        
        # Compute the gradient for the second input tensor
        dx2 = self.op(x1.T, dout_fp16).astype(ms.float16)
        
        return (dx1, dx2)


class GCNConv(GNNCell):
    def __init__(
        self,
        in_feat_size: int,
        out_size: int,
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
        activation=None,
        dropout=0.5,
    ):
        super().__init__()
        if in_feat_size <= 0 or not isinstance(in_feat_size, int):
            raise ValueError("in_feat_size must be positive int")
        if out_size <= 0 or not isinstance(out_size, int):
            raise ValueError("out_size must be positive int")
        if not isinstance(dropout, float):
            raise ValueError("dropout must be float")

        self.in_feat_size = in_feat_size
        self.out_size = out_size
        self.adj_matrix = adj_matrix

        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(
                f"For '{self.cls_name}', the 'dropout_prob' should be a number in range [0.0, 1.0), "
                f"but got {dropout}."
            )
        if activation is not None and not isinstance(activation, Cell):
            raise TypeError(
                f"For '{self.cls_name}', the 'activation' must a mindspore.nn.Cell, but got "
                f"{type(activation).__name__}."
            )
        self.fc = ms.nn.Dense(
            in_feat_size, out_size, weight_init=XavierUniform(), has_bias=False
        )

        self.bias = ms.Parameter(
            initializer("zero", (out_size), ms.float32), name="bias"
        )

        self.activation = activation
        self.min_clip = Tensor(1, ms.int32)
        self.max_clip = Tensor(100000000, ms.int32)
        self.drop_out = ms.nn.Dropout(p=dropout)

        self.matmul = MmadCustomAclnnNet()

        self.BLK_H = BLK_H
        self.batch_edgecount = batch_edgecount
        self.row_reindex = Tensor(row_reindex)
        self.row_index = Tensor(row_index)

        self.split_col_remaining_csr_indptr = split_col_remaining_csr_indptr
        self.split_col_remaining_coo_cols = split_col_remaining_coo_cols

        self.split_col_filtered_coo_rows = split_col_filtered_coo_rows
        self.split_col_filtered_coo_cols = split_col_filtered_coo_cols
        self.split_row_filtered_coo_rows = split_row_filtered_coo_rows
        self.split_row_filtered_coo_cols = split_row_filtered_coo_cols
        self.adj_matrix = adj_matrix
        self.edgeToRow_ms_tensor = edgeToRow_ms_tensor

        self.n_node = n_node
        self.SCATTER_ADD = ms.ops.TensorScatterAdd()
        self.GATHER = ms.ops.Gather()
        self.ZEROS = ms.ops.Zeros()
        self.SHAPE = ms.ops.Shape()
        self.RESHAPE = ms.ops.Reshape()
        self.add_op = ms.ops.Add()
        self.concat_op = ms.ops.Concat(axis=0)

    # pylint: disable=arguments-differ
    def construct(self, x, in_deg, out_deg, g: Graph):
        """
        Construct function for GCNConv.
        """

        out_deg = ms.ops.clip_by_value(out_deg, self.min_clip, self.max_clip)
        out_deg = ms.ops.Reshape()(
            ms.ops.Pow()(out_deg, -0.5), ms.ops.Shape()(out_deg) + (1,)
        )
        x = self.drop_out(x)
        x = ms.ops.Squeeze()(x)
        x = x * out_deg
        # x = x.astype(ms.float32)

        # NN operations

        x = self.fc(x)

        # Graph operations

        # Mixed method graph aggregation
        node_feat = x.astype(ms.float16)
        A_row_count = len(self.split_col_remaining_csr_indptr) - 1
        B_row_count = len(np.unique(self.split_row_filtered_coo_rows))
        node_sum = A_row_count + B_row_count

        embedding_dim = node_feat.shape[1]

        node_feat = ops.gather(node_feat, self.row_reindex, 0)

        # new_node_feat = Tensor(np.ones((node_sum, embedding_dim),dtype=np.float16), ms.float16)

        # vect_agg Data aggregation preparation

        window_count = (
            len(self.split_col_remaining_csr_indptr) - 1 + self.BLK_H - 1
        ) // self.BLK_H
        A_row_count = len(self.split_col_remaining_csr_indptr) - 1
        B_row_count = len(np.unique(self.split_row_filtered_coo_rows))

        A1 = Tensor(
            np.ones((A_row_count, node_feat.shape[1]), dtype=np.float16), ms.float16
        )

        A2_node_feat = self.ZEROS(
            (A_row_count, embedding_dim), ms.float16
        )  # Pre-allocate result space
        B_node_feat = self.ZEROS(
            (B_row_count, embedding_dim), ms.float16
        )  # Pre-allocate result space

        if len(self.split_col_filtered_coo_cols) != 0:  # When A2 is empty
            A2_dst_idx = Tensor(self.split_col_filtered_coo_rows, ms.int32)
            A2_dst_idx = self.RESHAPE(
                A2_dst_idx, (self.SHAPE(A2_dst_idx)[0], 1)
            )  # Must be at least 2D
            A2_src_idx = Tensor(self.split_col_filtered_coo_cols, ms.int32)
        else:
            A2_dst_idx = []
            A2_src_idx = []
        if len(self.split_row_filtered_coo_rows) != 0:  # When B is empty
            B_dst_idx = Tensor(self.split_row_filtered_coo_rows, ms.int32)
            B_dst_idx = self.RESHAPE(
                B_dst_idx, (self.SHAPE(B_dst_idx)[0], 1)
            )  # Must be at least 2D
            B_src_idx = Tensor(self.split_row_filtered_coo_cols, ms.int32)
        else:
            B_dst_idx = []
            B_src_idx = []

        # Compute the final result = A1 + A2 concatenated with B
        # First part cube computation  A1

        for winId in range(window_count):

            X_ms = ops.gather(
                node_feat, self.edgeToRow_ms_tensor[winId], 0
            )  
           

            start = winId * self.BLK_H
            end = min(
                (winId + 1) * self.BLK_H, len(self.split_col_remaining_csr_indptr) - 1
            )

            adj_matrix_rows_pad = (
                16 - (self.adj_matrix[winId].shape[0] % 16)
                if self.adj_matrix[winId].shape[0] % 16 != 0
                else 0
            )
            X_ms_rows_pad = 16 - (X_ms.shape[0] % 16) if X_ms.shape[0] % 16 != 0 else 0
            X_ms_cols_pad = 16 - (X_ms.shape[1] % 16) if X_ms.shape[1] % 16 != 0 else 0

            adj_matrix_pad_op = ops.Pad(((0, adj_matrix_rows_pad), (0, X_ms_rows_pad)))
            X_ms_pad_op = ops.Pad(((0, X_ms_rows_pad), (0, X_ms_cols_pad)))

            # Apply padding
            padded_adj_matrix = adj_matrix_pad_op(self.adj_matrix[winId])
            padded_X_ms = X_ms_pad_op(X_ms)

            tmp = self.matmul(padded_adj_matrix, padded_X_ms)  

            tmp = tmp[0 : (end - start), 0:embedding_dim].astype(
                ms.float16
            )  # Slice and type conversion
            A1[start:end] = tmp

        # Second part vector   A2

        if len(self.split_col_filtered_coo_cols) != 0:  # When A2 is empty
            #    A2 = self.vector_agg(self.split_col_filtered_coo_rows, self.split_col_filtered_coo_cols, A2_dst_idx, A2_src_idx, node_feat, A2_node_feat, A_row_count, self.batch_edgecount)

            for batch_id in range(
                (len(self.split_col_filtered_coo_rows) + self.batch_edgecount - 1)
                // self.batch_edgecount
            ):

                edge_start = batch_id * self.batch_edgecount
                edge_end = min(
                    (batch_id + 1) * self.batch_edgecount,
                    len(self.split_col_filtered_coo_rows),
                )
                scatter_src_idx = A2_src_idx[edge_start:edge_end]  
                scatter_dst_idx = A2_dst_idx[edge_start:edge_end]

                gather_x = self.GATHER(node_feat, scatter_src_idx, 0)
                A2_node_feat = self.SCATTER_ADD(A2_node_feat, scatter_dst_idx, gather_x)
            A2 = A2_node_feat
        else:
            A2 = []
            
         # Third part vector B
        if len(self.split_row_filtered_coo_rows) != 0:  # When B is empty

            for batch_id in range(
                (len(self.split_row_filtered_coo_rows) + self.batch_edgecount - 1)
                // self.batch_edgecount
            ):

                edge_start = batch_id * self.batch_edgecount
                edge_end = min(
                    (batch_id + 1) * self.batch_edgecount,
                    len(self.split_row_filtered_coo_rows),
                )
                scatter_src_idx = B_src_idx[edge_start:edge_end]  
                scatter_dst_idx = B_dst_idx[edge_start:edge_end]

                gather_x = self.GATHER(node_feat, scatter_src_idx, 0)

                B_node_feat = self.SCATTER_ADD(B_node_feat, scatter_dst_idx, gather_x)

            B = B_node_feat
        else:
            B = []

        # Compute the final result = A1 + A2 concatenated with B
        if len(self.split_col_filtered_coo_cols) != 0:  # When A2 is empty
            new_node_feat = self.add_op(A1, A2)
        else:
            new_node_feat = A1

        if len(self.split_row_filtered_coo_rows) != 0:  # When B is empty
            new_node_feat = self.concat_op((new_node_feat, B))

        new_node_feat = ops.gather(new_node_feat, self.row_index, 0)

        x = new_node_feat
        g.set_vertex_attr({"x": x})

        in_deg = ms.ops.clip_by_value(in_deg, self.min_clip, self.max_clip)
        in_deg = ms.ops.Reshape()(
            ms.ops.Pow()(in_deg, -0.5), ms.ops.Shape()(in_deg) + (1,)
        )
        # x = x * in_deg  # Directly operate on matrix features x
        x = [v.x for v in g.dst_vertex] * in_deg
        x = x + self.bias

        return x

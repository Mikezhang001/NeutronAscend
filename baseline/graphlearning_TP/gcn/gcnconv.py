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
from mindspore import Tensor,ops
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
# from custom_ops.sparse_dense_matmul import sparse_dense_matmul

from mindspore.ops import DataType, CustomRegOp
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype


def custom_bprop(x1, x2, out, dout):
    """
    计算相对于输入张量 x1 和 x2 的梯度。
    
    Args:
        x1 (Tensor): 第一个输入张量。
        x2 (Tensor): 第二个输入张量。
        out (Tensor): 正向传播的输出张量（这里不直接使用）。
        dout (Tensor): 输出梯度。
        
    Returns:
        tuple: 包含 x1 和 x2 相对于损失的梯度。
    """
    # 使用 ops.matmul 实现反向传播中的梯度计算
    dout_fp16 = dout.astype(ms.float16)
    dx1 = ops.matmul(dout_fp16, x2.T)
    dx2 = ops.matmul(x1.T, dout_fp16)
    # print("Hello world one") #做是做的，但感觉加不加没变
    return (dx1, dx2)
    # return (x1,x2)
    
class MmadCustomTPAclnnNet(Cell):
    def __init__(self):
        super(MmadCustomTPAclnnNet, self).__init__()
        aclnn_ref_info = CustomRegOp("aclnnMmadCustomTP") \
            .input(0, "x", "required") \
            .input(1, "y", "required") \
            .output(0, "z", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F32_Default) \
            .target("Ascend") \
            .get_op_info()
        self.op = ops.Custom("aclnnMmadCustomTP", lambda x_shape, y_shape: (x_shape[0], y_shape[1]), out_dtype=ms.float32, func_type="aot", bprop=None,
                             reg_info=aclnn_ref_info)
    def construct(self, x1, x2):
        return self.op(x1, x2)
    def bprop(self, x1, x2, out, dout):
        dout_fp16 = dout.astype(ms.float16)
        dx1 = self.op(dout_fp16, x2.T).astype(ms.float16)
        dx2 = self.op(x1.T, dout_fp16).astype(ms.float16)
        # print("Hello world two")   
        return (dx1, dx2)
    
class GCNConv(GNNCell):
    r"""
    Graph Convolution Network Layer.
    From the paper `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`_ .

    .. math::
        h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ji}}h_j^{(l)}W^{(l)})

    :math:`\mathcal{N}(i)` represents the neighbour node of :math:`i`.
    :math:`c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}`.

    .. math::
        h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{e_{ji}}{c_{ji}}h_j^{(l)}W^{(l)})

    Args:
        in_feat_size (int): Input node feature size.
        out_size (int): Output node feature size.
        activation (Cell, optional): Activation function. Default: ``None``.
        dropout (float, optional): The dropout rate, greater than 0 and less equal than 1. E.g. dropout=0.1,
            dropping out 10% of input units. Default: ``0.5``.

    Inputs:
        - **x** (Tensor) - The input node features. The shape is :math:`(N, D_{in})`
          where :math:`N` is the number of nodes,
          and :math:`D_{in}` should be equal to `in_feat_size` in `Args`.
        - **in_deg** (Tensor) - In degree for nodes. The shape is :math:`(N, )` where :math:`N` is the number of nodes.
        - **out_deg** (Tensor) - Out degree for nodes. The shape is :math:`(N, )`
          where :math:`N` is the number of nodes.
        - **g** (Graph) - The input graph.

    Outputs:
        - Tensor, output node features with shape of :math:`(N, D_{out})`, where :math:`(D_{out})` should be the same as
          `out_size` in `Args`.

    Raises:
        TypeError: If `in_feat_size` or `out_size` is not an int.
        TypeError: If `dropout` is not a float.
        TypeError: If `activation` is not a `mindspore.nn.Cell`.
        ValueError: If `dropout` is not in range (0.0, 1.0]

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gl.nn import GCNConv
        >>> from mindspore_gl import GraphField
        >>> n_nodes = 4
        >>> n_edges = 7
        >>> feat_size = 4
        >>> src_idx = ms.Tensor([0, 1, 1, 2, 2, 3, 3], ms.int32)
        >>> dst_idx = ms.Tensor([0, 0, 2, 1, 3, 0, 1], ms.int32)
        >>> ones = ms.ops.Ones()
        >>> feat = ones((n_nodes, feat_size), ms.float32)
        >>> graph_field = GraphField(src_idx, dst_idx, n_nodes, n_edges)
        >>> in_degree = ms.Tensor([3, 2, 1, 1], ms.int32)
        >>> out_degree = ms.Tensor([1, 2, 1, 2], ms.int32)
        >>> gcnconv = GCNConv(in_feat_size=4, out_size=2, activation=None, dropout=1.0)
        >>> res = gcnconv(feat, in_degree, out_degree, *graph_field.get_graph())
        >>> print(res.shape)
        (4, 2)
    """
    def __init__(self,
                 in_feat_size: int,
                 out_size: int,
                 BLK_H, batch_edgecount, n_node, split_col_remaining_csr_indptr,
                 split_col_remaining_coo_cols, split_col_filtered_coo_rows, split_col_filtered_coo_cols, split_row_filtered_coo_rows,
                 split_row_filtered_coo_cols, adj_matrix, edgeToRow_ms_tensor, row_reindex, row_index,
                 activation=None,
                 dropout=0.5):
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
            raise ValueError(f"For '{self.cls_name}', the 'dropout_prob' should be a number in range [0.0, 1.0), "
                             f"but got {dropout}.")
        if activation is not None and not isinstance(activation, Cell):
            raise TypeError(f"For '{self.cls_name}', the 'activation' must a mindspore.nn.Cell, but got "
                            f"{type(activation).__name__}.")
        self.fc = ms.nn.Dense(in_feat_size, out_size, weight_init=XavierUniform(), has_bias=False)
        # print(self.fc.weight)
        # self.fc.weight = ms.Parameter(self.fc.weight.astype(ms.float16), requires_grad=True)
        # print(self.fc.weight)
        self.bias = ms.Parameter(initializer('zero', (out_size), ms.float32), name="bias")
        # print(self.bias)
        self.activation = activation
        self.min_clip = Tensor(1, ms.int32)
        self.max_clip = Tensor(100000000, ms.int32)
        self.drop_out = ms.nn.Dropout(p=dropout)
        #原生乘法
        # self.matmul = MatMul()  # 初始化矩阵乘法操作
        #nn.cell反向传播(先cell，无cell，再调用自定义注册的反向传播)
        self.matmul = MmadCustomTPAclnnNet()
        #自定义矩阵乘法,注册反向(不加aclnn就解决了)
        # aclnn_ref_info = CustomRegOp("MmadCustomTwo") \
        #     .input(0, "x", "required") \
        #     .input(1, "y", "required") \
        #     .output(0, "z", "required") \
        #     .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F32_Default) \
        #     .target("Ascend") \
        #     .get_op_info()
        # self.matmul = ops.Custom("aclnnMmadCustomTwo", lambda x_shape, y_shape: (x_shape[0], y_shape[1]), out_dtype=ms.float32, func_type="aot", bprop=custom_bprop,
        #                      reg_info=aclnn_ref_info)
        """
        分批次需要准备的数据
        self.SCATTER_ADD = ms.ops.TensorScatterAdd()  
        self.GATHER = ms.ops.Gather()
        self.ZEROS = ms.ops.Zeros()
        self.batch_edgecount = 10000000
        self.dst_idx = dst_idx
        self.src_idx = src_idx
        """
        #混合方法的工作
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
        
        self.n_node = n_node  # 示例，具体根据实际情况调整
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
        
        total_start_time = time.time()  # 总计时开始
        
        out_deg = ms.ops.clip_by_value(out_deg, self.min_clip, self.max_clip)
        out_deg = ms.ops.Reshape()(ms.ops.Pow()(out_deg, -0.5), ms.ops.Shape()(out_deg) + (1,))
        x = self.drop_out(x)
        x = ms.ops.Squeeze()(x)
        x = x * out_deg
        # x = x.astype(ms.float32)
        
        #NN操作
        nn_start_time = time.time()
        x = self.fc(x)
        nn_end_time = time.time()
        
        #图操作
        graph_start_time = time.time()  # 图操作计时开始
        
        #原来的图聚合
        # g.set_vertex_attr({"x": x})
        # for v in g.dst_vertex:
        #     v.x = g.sum([u.x for u in v.innbs])

        #分批次图聚合
        # new_node_feat = self.ZEROS(x.shape, x.dtype)
        # for batch_id in range( (len(self.src_idx) + self.batch_edgecount -1) // self.batch_edgecount):
        #     edge_start = batch_id * self.batch_edgecount
        #     edge_end = min((batch_id + 1) * self.batch_edgecount, len(self.src_idx))
        #     scatter_src_idx = self.src_idx[edge_start : edge_end] #np.int32切片不行
        #     scatter_dst_idx = self.dst_idx[edge_start : edge_end]
        #     gather_x = self.GATHER(x, scatter_src_idx, 0)
        #     new_node_feat = self.SCATTER_ADD(new_node_feat, scatter_dst_idx, gather_x) 
        # x = new_node_feat    
        # g.set_vertex_attr({"x": x})    
         
        
        #混合方法的图聚合  
        node_feat =  x.astype(ms.float16)
        A_row_count = len(self.split_col_remaining_csr_indptr) -1
        B_row_count = len(np.unique(self.split_row_filtered_coo_rows))
        node_sum = A_row_count + B_row_count
      
       
        embedding_dim = node_feat.shape[1]
        
        node_feat =  ops.gather(node_feat, self.row_reindex, 0)
        
        # new_node_feat = Tensor(np.ones((node_sum, embedding_dim),dtype=np.float16), ms.float16)
    
    
        #vect_agg 数据聚集的前准备


        window_count = (len(self.split_col_remaining_csr_indptr) - 1 + self.BLK_H - 1) // self.BLK_H   
        A_row_count = len(self.split_col_remaining_csr_indptr) -1  
        B_row_count = len(np.unique(self.split_row_filtered_coo_rows)) 
    
        A1 = Tensor(np.ones((A_row_count, node_feat.shape[1]),dtype=np.float16), ms.float16)

        A2_node_feat = self.ZEROS(( A_row_count, embedding_dim), ms.float16) #先开辟结果空间
        B_node_feat = self.ZEROS(( B_row_count, embedding_dim), ms.float16) #先开辟结果空间
    
        if len(self.split_col_filtered_coo_cols)!=0:#A2为空时  
          A2_dst_idx = Tensor(self.split_col_filtered_coo_rows,  ms.int32)
          A2_dst_idx = self.RESHAPE(A2_dst_idx, (self.SHAPE(A2_dst_idx)[0], 1))#必须得二维以上
          A2_src_idx = Tensor(self.split_col_filtered_coo_cols,  ms.int32)
        else:
          A2_dst_idx = []
          A2_src_idx = []
        if len(self.split_row_filtered_coo_rows)!=0:#B为空时
          B_dst_idx = Tensor(self.split_row_filtered_coo_rows,  ms.int32)
          B_dst_idx = self.RESHAPE(B_dst_idx, (self.SHAPE(B_dst_idx)[0], 1))#必须得二维以上
          B_src_idx = Tensor(self.split_row_filtered_coo_cols,  ms.int32)
        else:
          B_dst_idx = []
          B_src_idx = []      

      #计算公式 result = A1+A2拼接B
      #第一部分cube计算  A1
     
        for winId in range(window_count):
              
              X_ms = ops.gather(node_feat, self.edgeToRow_ms_tensor[winId], 0) #不需要重复gather对8192的reddit来说
              # print(f"X_ms的阻塞值为{X_ms[0][0]}")
              
              start = winId * self.BLK_H
              end = min((winId + 1) * self.BLK_H, len(self.split_col_remaining_csr_indptr) - 1)
        
        
              adj_matrix_rows_pad = 16 - (self.adj_matrix[winId].shape[0] % 16) if self.adj_matrix[winId].shape[0] % 16 != 0 else 0
              X_ms_rows_pad = 16 - (X_ms.shape[0] % 16) if X_ms.shape[0] % 16 != 0 else 0
              X_ms_cols_pad = 16 - (X_ms.shape[1] % 16) if X_ms.shape[1] % 16 != 0 else 0
         
              adj_matrix_pad_op = ops.Pad(((0, adj_matrix_rows_pad), (0, X_ms_rows_pad)))
              X_ms_pad_op = ops.Pad(((0, X_ms_rows_pad), (0, X_ms_cols_pad)))
              
        
              # 应用填充
              padded_adj_matrix = adj_matrix_pad_op(self.adj_matrix[winId])
              padded_X_ms = X_ms_pad_op(X_ms)
             

              tmp =  self.matmul(padded_adj_matrix, padded_X_ms) #计算
              
              tmp = tmp[0:(end - start), 0:embedding_dim].astype(ms.float16) #切割和类型转换
              A1[start:end] = tmp       


       #第二部分vector   A2
       
      
        if len(self.split_col_filtered_coo_cols)!=0:#A2为空时  
        #    A2 = self.vector_agg(self.split_col_filtered_coo_rows, self.split_col_filtered_coo_cols, A2_dst_idx, A2_src_idx, node_feat, A2_node_feat, A_row_count, self.batch_edgecount)
        
           for batch_id in range( (len(self.split_col_filtered_coo_rows) + self.batch_edgecount -1) // self.batch_edgecount):
         
                edge_start = batch_id * self.batch_edgecount
                edge_end = min((batch_id + 1) * self.batch_edgecount, len(self.split_col_filtered_coo_rows))
                scatter_src_idx = A2_src_idx[edge_start : edge_end] #np.int32切片不行
                scatter_dst_idx = A2_dst_idx[edge_start : edge_end]
       
                gather_x = self.GATHER(node_feat, scatter_src_idx, 0)
                A2_node_feat = self.SCATTER_ADD(A2_node_feat, scatter_dst_idx, gather_x) 
           A2 = A2_node_feat
        else:
           A2 = []
        # #第三部分vector   B     
        if len(self.split_row_filtered_coo_rows)!=0:#B为空时
        #    B = self.vector_agg(self.split_row_filtered_coo_rows, self.split_row_filtered_coo_cols, B_dst_idx, B_src_idx, node_feat, B_node_feat, B_row_count, self.batch_edgecount) 
           
           for batch_id in range( (len(self.split_row_filtered_coo_rows) + self.batch_edgecount -1) // self.batch_edgecount):
         
               edge_start = batch_id * self.batch_edgecount
               edge_end = min((batch_id + 1) * self.batch_edgecount, len(self.split_row_filtered_coo_rows))
               scatter_src_idx = B_src_idx[edge_start : edge_end] #np.int32切片不行
               scatter_dst_idx = B_dst_idx[edge_start : edge_end]
       
               gather_x = self.GATHER(node_feat, scatter_src_idx, 0)
       
               B_node_feat = self.SCATTER_ADD(B_node_feat, scatter_dst_idx, gather_x) 
           
           B = B_node_feat         
        else:
           B = []

        #合并结果,result = A1+A2拼接B 
        if len(self.split_col_filtered_coo_cols)!=0:#A2为空时      
            new_node_feat = self.add_op(A1, A2)
        else:
            new_node_feat = A1
            
        if len(self.split_row_filtered_coo_rows)!=0:#B为空时
           new_node_feat = self.concat_op((new_node_feat, B))
           
        new_node_feat = ops.gather(new_node_feat, self.row_index, 0)
        
        x = new_node_feat
        g.set_vertex_attr({"x": x})
        
        
        
        
        
        
        
        #原来后续处理部分
        in_deg = ms.ops.clip_by_value(in_deg, self.min_clip, self.max_clip)
        in_deg = ms.ops.Reshape()(ms.ops.Pow()(in_deg, -0.5), ms.ops.Shape()(in_deg) + (1,))
        # x = x * in_deg  # 直接用矩阵特征 x 进行操作
        x = [v.x for v in g.dst_vertex] * in_deg
        x = x + self.bias
        # 图操作计时结束
        
        total_end_time = time.time()
        
        #  # 计算时间占比
        # nn_duration = nn_end_time - nn_start_time
        # graph_duration = graph_end_time - graph_start_time
        # total_duration = total_end_time - total_start_time
        
        # print('time:{}'.format(time.time()))
        # print('nn time:{} ms'.format((nn_end_time - nn_start_time) * 1000))
        # print('graph time:{} ms'.format((graph_end_time - graph_start_time) * 1000))
        # print('total time:{} ms'.format((total_end_time - total_start_time) * 1000))
        # 打印时间占比
        # print(f"NN操作时间: {nn_duration:.4f}秒")
        # print(f"NN操作时间: {nn_duration:.4f}秒, 占比: {nn_duration / total_duration:.2%}")
        # # print(f"图操作时间: {graph_duration:.4f}秒, 占比: {graph_duration / total_duration:.2%}")
        # print(f"Dataset loading time: {time.time() - total_start_time:.4f} seconds")
        # logging.basicConfig(level=logging.INFO, format='%(message)s')
        # logging.info("NN操作时间: {:.4f}秒".format(time.time() - total_start_time))
        # logging.info("图操作时间: {:.4f}秒".format(graph_duration))


        return x

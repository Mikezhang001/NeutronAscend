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
"""Full training dataset"""
import numpy as np
import mindspore as ms

from mindspore_gl import GraphField
from mindspore import COOTensor
from mindspore import  Tensor
class GraphDataset:
    """Full training numpy dataset """
    def __init__(self, dataset_name: str) -> None:
        
        
        dataset_path = f"../../../data"
        # embedding_dim ={"reddit": 602, "products":100, "livejournal":600, "cora":1433, "pubmed":500, "citeseer": 3703, "WikiTalk": 600, "ogbn-arxiv":128,
        #                 "physics": 8415, "amazon": 200, "orkut": 600, "pattern1":512, "mip1":512, "mycielskian15":512, "nd12k":512, "human_gene1":512, "mycielskian17":512}
        
        # adj_coo_row = np.fromfile(f'{dataset_path}/{dataset_name}/{dataset_name}_coo_row.bin', dtype=np.int32)
        # adj_coo_col = np.fromfile(f'{dataset_path}/{dataset_name}/{dataset_name}_coo_col.bin', dtype=np.int32)


        # node_feat = Tensor(np.fromfile(f'{dataset_path}/{dataset_name}/{dataset_name}_features.bin', dtype=np.float32).reshape(-1, embedding_dim[dataset_name]), ms.float16)#随便取导致acc很小
        # # node_feat = Tensor(np.ones((n_nodes, embedding_dim[dataset_name]),dtype=np.float16), ms.float16)#随便取导致acc很小
        # self.x = node_feat
        # print(f"x的形状为{self.x.shape}")   
        # # n_nodes = max(adj_coo_row.max(), adj_coo_col.max()) + 1
        # n_nodes = node_feat.shape[0]
        # print(f"顶点数为{n_nodes}") 

        # y = np.fromfile(f'{dataset_path}/{dataset_name}/{dataset_name}_labels.bin', dtype=np.int32)
        # self.y = Tensor(y, dtype = ms.int32)
        # self.n_classes = int(np.max(y) + 1)
        
        # self.train_mask = np.fromfile(f'{dataset_path}/{dataset_name}/{dataset_name}_train_mask.bin', dtype=np.int32).astype(bool)#acc为0的原因，要转bool
        # self.test_mask = np.fromfile(f'{dataset_path}/{dataset_name}/{dataset_name}_test_mask.bin', dtype=np.int32).astype(bool)
        
        # self.n_nodes = n_nodes
        # self.n_edges = adj_coo_row.shape[0]
        # self.g = GraphField(ms.Tensor(adj_coo_row, ms.int32), ms.Tensor(adj_coo_col, ms.int32), int(self.n_nodes), int(self.n_edges))
        
        
        # self.in_deg = ms.Tensor(np.fromfile(f'{dataset_path}/{dataset_name}/{dataset_name}_in_degrees.bin', dtype=np.int32), ms.int32)
        # self.out_deg = ms.Tensor(np.fromfile(f'{dataset_path}/{dataset_name}/{dataset_name}_out_degrees.bin', dtype=np.int32), ms.int32)        
        

        # npz = np.load(data_path)
        # # self.x = ms.Tensor(npz['feat'], ms.float32) 
        # self.x = ms.Tensor(npz['feat'], ms.float16)
           
        
        ##原本的操作
        npz = np.load(f"{dataset_path}/npz/{dataset_name}.npz")
        # self.x = ms.Tensor(npz['feat'], ms.float32) 
        self.x = ms.Tensor(npz['feat']) 
        print(f"x的shape为{self.x.shape}")
        print(f"x的类型为{self.x.dtype}")
        self.y = ms.Tensor(npz['label'], ms.int32)
        
        self.train_mask = npz.get('train_mask', default=None)
        self.test_mask = npz.get('test_mask', default=None)

        self.n_nodes = npz.get('n_nodes', default=self.x.shape[0])
        self.n_edges = npz.get('n_edges', default=npz['adj_coo_row'].shape[0])

        self.g = GraphField(ms.Tensor(npz['adj_coo_row'], dtype=ms.int32),
                            ms.Tensor(npz['adj_coo_col'], dtype=ms.int32),
                            int(self.n_nodes),
                            int(self.n_edges))

        
        self.n_classes = int(npz.get('n_classes', default=np.max(npz['label']) + 1))
        self.in_deg = ms.Tensor(npz.get('in_degrees', default=None), ms.int32)
        self.out_deg = ms.Tensor(npz.get('out_degrees', default=None), ms.int32)
        
        
        
        """
        #仅仅使用原来的方法进行分批
        SHAPE = ms.ops.Shape()
        RESHAPE = ms.ops.Reshape()
        dst_idx = ms.Tensor(adj_coo_row,  ms.int32)
        self.dst_idx = RESHAPE(dst_idx, (SHAPE(dst_idx)[0], 1))#必须得二维以上
        self.src_idx = ms.Tensor(adj_coo_col,  ms.int32)
        self.adj_matrix = []
        """
      
        """
        #仅仅使用矩阵乘法的方法，构建邻接矩阵
        adj_indices = np.vstack((npz['adj_coo_row'], npz['adj_coo_col'])).T  # COO 的索引
        adj_values = np.ones(self.n_edges)  # 值为 1
        self.adj_matrix = COOTensor(indices=ms.Tensor(adj_indices, dtype=ms.int32),
                                    values=ms.Tensor(adj_values, dtype=ms.float16),
                                    shape=(int(self.n_nodes), int(self.n_nodes))) # 构建 COO 邻接矩阵
        self.adj_matrix = self.adj_matrix.to_dense()
        """
        
        #采用过滤边和矩阵乘法结合的方法
        window_H = {"Cora":8192, "Pubmed":81920, "Citeseer": 81920, "products": 6400, "mycielskian18": 8192, "reddit": 8192}
        self.BLK_H = window_H[dataset_name]
        print(f"窗口高为{self.BLK_H}")
        self.batch_edgecount = 10000000
        row_reindex_np = np.fromfile(  f"{dataset_path}/mmad/{dataset_name}/row_reindex.bin", dtype=np.int32)
        self.row_reindex = row_reindex_np.tolist()#特征重排序
        self.row_index = np.argsort(row_reindex_np).tolist()#特征重排序
        self.split_col_remaining_csr_indptr = np.fromfile( f"{dataset_path}/mmad/{dataset_name}/split_col_remaining_csr_indptr.bin", dtype=np.int32)
        self.split_col_remaining_coo_cols = np.fromfile(f"{dataset_path}/mmad/{dataset_name}/split_col_remaining_coo_cols.bin", dtype=np.int32)
    
        self.split_col_filtered_coo_rows = np.fromfile(  f"{dataset_path}/mmad/{dataset_name}/split_col_filtered_coo_rows.bin", dtype=np.int32)
        self.split_col_filtered_coo_cols = np.fromfile(  f"{dataset_path}/mmad/{dataset_name}/split_col_filtered_coo_cols.bin", dtype=np.int32)
        self.split_row_filtered_coo_rows = np.fromfile(  f"{dataset_path}/mmad/{dataset_name}/split_row_filtered_coo_rows.bin", dtype=np.int32)
        self.split_row_filtered_coo_cols = np.fromfile(  f"{dataset_path}/mmad/{dataset_name}/split_row_filtered_coo_cols.bin", dtype=np.int32)  
        
        #构建邻接矩阵
        adj_matrix_np = np.load(f"{dataset_path}/mmad/{dataset_name}/adj_matrix_np.npz", allow_pickle=True)
        edgeToRow_np = np.load(f"{dataset_path}/mmad/{dataset_name}/edgeToRow_np.npz", allow_pickle=True)
        self.adj_matrix = []
        self.edgeToRow_ms_tensor = []
        print("邻接矩阵读取开始")
        for key in adj_matrix_np.keys():
            self.adj_matrix.append(Tensor(adj_matrix_np[key], dtype=ms.float16))
        print("邻接矩阵读取结束")
        for key in edgeToRow_np.keys():
            self.edgeToRow_ms_tensor.append(Tensor(edgeToRow_np[key], dtype=ms.int32))
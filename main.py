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
"""gcn model implemented using mindspore-gl"""
import time
import argparse
import os
import sys

import numpy as np
import mindspore as ms
from mindspore.profiler import Profiler
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context

from mindspore_gl.nn import GNNCell
from mindspore_gl import Graph

from gnngraph_dataset import GraphDataset


# pylint: disable=C0413
from gcn import GCNNet


class LossNet(GNNCell):
    """ LossNet definition """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

    def construct(self, x, in_deg, out_deg, train_mask, target, g: Graph):
        predict = self.net(x, in_deg, out_deg, g)
        target = ops.Squeeze()(target)
        loss = self.loss_fn(predict, target)
        loss = loss * train_mask
        return ms.ops.ReduceSum()(loss) / ms.ops.ReduceSum()(train_mask)


class DataNet(ms.nn.Cell):
    """data net"""

    def __init__(self, ds, net):
        super().__init__()
        self.x = ds.x
        self.in_deg = ds.in_deg
        self.out_deg = ds.out_deg
        self.train_mask = ms.Tensor(ds.train_mask, ms.float32)
        self.y = ds.y
        self.src_idx = ds.g.src_idx
        self.dst_idx = ds.g.dst_idx
        self.n_nodes = ds.g.n_nodes
        self.n_edges = ds.g.n_edges
        print("dataset contains ", self.n_nodes, "nodes", self.n_edges, "edges")
        self.net = net

    def construct(self):
        return self.net(self.x, self.in_deg, self.out_deg, self.train_mask, self.y, self.src_idx, self.dst_idx,
                        self.n_nodes, self.n_edges)


def main(train_args):
  
    context.set_context(device_target=train_args.device, save_graphs=False,
                            save_graphs_path="./computational_graph/",
                            mode=context.GRAPH_MODE, enable_graph_kernel=True,
                            graph_kernel_flags="--enable_recompute_fusion=false "
                                               "--enable_parallel_fusion=true ",
                                               device_id=train_args.device_id)
    
    
    
    # Timing dataset loading
    start_time = time.time()
    ds = GraphDataset(train_args.data_name)
    print(f"Dataset loading time: {time.time() - start_time:.4f} seconds")

    feature_size = ds.x.shape[1]
    if train_args.profile:
        ms_profiler = Profiler(profiler_level=0, aicore_metrics=1, l2_cache=True, hbm_ddr=True, pcie=True, output_path="./profiler_output")
    # model
    start_time = time.time()
    aicore = ms.Tensor(np.ones([train_args.aicore_num, 1]).astype(np.float16))#指定aicore数量
    net = GCNNet(data_feat_size=feature_size,
                 hidden_dim_size=train_args.num_hidden,
                 n_classes=ds.n_classes,
                 BLK_H=ds.BLK_H, batch_edgecount=ds.batch_edgecount, n_node=ds.n_nodes, split_col_remaining_csr_indptr=ds.split_col_remaining_csr_indptr,
                 split_col_remaining_coo_cols=ds.split_col_remaining_coo_cols, split_col_filtered_coo_rows=ds.split_col_filtered_coo_rows, 
                 split_col_filtered_coo_cols=ds.split_col_filtered_coo_cols, split_row_filtered_coo_rows=ds.split_row_filtered_coo_rows,
                 split_row_filtered_coo_cols=ds.split_row_filtered_coo_cols, adj_matrix=ds.adj_matrix, edgeToRow_ms_tensor=ds.edgeToRow_ms_tensor, 
                 row_reindex=ds.row_reindex, row_index=ds.row_index,
                 dropout=train_args.dropout,
                 activation=ms.nn.ELU, num_layers=train_args.num_layers, aicore_num= aicore)
    optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=train_args.lr, weight_decay=train_args.weight_decay)
    loss = LossNet(net)
    train_net = nn.TrainOneStepCell(loss, optimizer)
    train_net = DataNet(ds, train_net)
    print(f"Model creation time: {time.time() - start_time:.4f} seconds")
    
    total = 0.
    warm_up = 3
    for e in range(train_args.epochs):
        beg = time.time()
        train_net.set_train()
        train_loss = train_net()
        print(f"train_loss={train_loss}")
        end = time.time()
        dur = end - beg
        if e >= warm_up:
            total = total + dur

        test_mask = ds.test_mask
        if test_mask is not None:
            net.set_train(False)
            out = net(ds.x, ds.in_deg, ds.out_deg, ds.g.src_idx, ds.g.dst_idx, ds.g.n_nodes, ds.g.n_edges).asnumpy()
            labels = ds.y.asnumpy()
            predict = np.argmax(out[test_mask], axis=1)
            label = labels[test_mask]
            count = np.equal(predict, label)
        
            print('Epoch time:{} ms Train loss {} Test acc:{}'.format(dur * 1000, train_loss,
                                                                      np.sum(count) / label.shape[0]))
        # print('Epoch time:{} ms Train loss {}'.format(dur * 1000, train_loss))
    print("Model:{} Dataset:{} Avg epoch time:{}".format("GCN", train_args.data_name,
                                                         total * 1000 / (train_args.epochs - warm_up)))
    if train_args.profile:
        ms_profiler.analyse()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--data-name", type=str, default='Cora',
                        help="path to dataloader")
    parser.add_argument("--device", type=str, default="Ascend", help="which device to use")
    parser.add_argument("--dropout", type=float, default=0.5, help="drop out rate")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--num-layers", type=int, default=3, help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=256, help="number of hidden units")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument('--profile', action='store_true', help="Enable profiling")
    parser.add_argument('--device-id', type=int, default=0, help="running device_id")
    parser.add_argument('--aicore-num', type=int, default=20, help="number of aicore, [1, 20]")
    args = parser.parse_args()
    print(args)
    main(args)

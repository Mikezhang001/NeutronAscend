import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor, COOTensor, context
import os

npu_id = 0
context.set_context(device_target="CPU", device_id=npu_id)

dataset_path = f"../data"
def csr_to_coo(nodePointer, edgeList):    
    src_idx = np.repeat(
        np.arange(len(nodePointer) - 1, dtype=np.int32),
        np.diff(nodePointer).astype(np.int32),
    )
    dst_idx = edgeList
    
    sorted_indices = np.lexsort((dst_idx, src_idx)) 
    
    src_idx = src_idx[sorted_indices]
    dst_idx = dst_idx[sorted_indices]
    src_idx = np.array(src_idx, dtype=np.int32)
    dst_idx = np.array(dst_idx, dtype=np.int32)
  
    return src_idx, dst_idx

def vector_agg(
    coo_rows,
    coo_cols,
    dst_idx,
    src_idx,
    node_feat,
    new_node_feat,
    n_nodes,
    batch_edgecount=1000000,
):

    SCATTER_ADD = ms.ops.TensorScatterAdd()
    GATHER = ms.ops.Gather()

    x = node_feat

    for batch_id in range((len(coo_rows) + batch_edgecount - 1) // batch_edgecount):
        edge_start = batch_id * batch_edgecount
        edge_end = min((batch_id + 1) * batch_edgecount, len(coo_rows))
        scatter_src_idx = src_idx[edge_start:edge_end]
        scatter_dst_idx = dst_idx[edge_start:edge_end]
        #    gather_x = GATHER(x, scatter_src_idx, 0)#first method gather
        gather_x = x[scatter_src_idx]  # second method gather
        new_node_feat = SCATTER_ADD(new_node_feat, scatter_dst_idx, gather_x)

    return new_node_feat


def run(dataset_name, BLK_H, batch_edgecount, embedding_dim):

    row_reindex_np = np.fromfile(
        f"{dataset_path}/mmad/{dataset_name}/row_reindex.bin", dtype=np.int32
    )
    split_col_remaining_csr_indptr = np.fromfile(
        f"{dataset_path}/mmad/{dataset_name}/split_col_remaining_csr_indptr.bin",
        dtype=np.int32,
    )
    split_col_remaining_coo_cols = np.fromfile(
        f"{dataset_path}/mmad/{dataset_name}/split_col_remaining_coo_cols.bin",
        dtype=np.int32,
    )

    split_col_filtered_coo_rows = np.fromfile(
        f"{dataset_path}/mmad/{dataset_name}/split_col_filtered_coo_rows.bin",
        dtype=np.int32,
    )
    split_col_filtered_coo_cols = np.fromfile(
        f"{dataset_path}/mmad/{dataset_name}/split_col_filtered_coo_cols.bin",
        dtype=np.int32,
    )
    split_row_filtered_coo_rows = np.fromfile(
        f"{dataset_path}/mmad/{dataset_name}/split_row_filtered_coo_rows.bin",
        dtype=np.int32,
    )
    split_row_filtered_coo_cols = np.fromfile(
        f"{dataset_path}/mmad/{dataset_name}/split_row_filtered_coo_cols.bin",
        dtype=np.int32,
    )

    A_row_count = len(split_col_remaining_csr_indptr) - 1
    B_row_count = len(np.unique(split_row_filtered_coo_rows))
    node_sum = A_row_count + B_row_count

    # node_feat = Tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]], ms.float16)
    np.random.seed(24)
    node_feat = Tensor(np.random.rand(node_sum, embedding_dim), ms.float16)
    # node_feat = Tensor(np.ones((node_sum, embedding_dim),dtype=np.float16), ms.float16)

    # new_node_feat = Tensor(np.ones((node_sum, embedding_dim),dtype=np.float16), ms.float16)

    adj_matrix_np = np.load(
        f"{dataset_path}/mmad/{dataset_name}/adj_matrix_np.npz", allow_pickle=True
    )
    edgeToRow_np = np.load(
        f"{dataset_path}/mmad/{dataset_name}/edgeToRow_np.npz", allow_pickle=True
    )
    adj_matrix = []
    edgeToRow_ms_tensor = []

    for key in adj_matrix_np.keys():
        adj_matrix.append(Tensor(adj_matrix_np[key], dtype=ms.float16))

    for key in edgeToRow_np.keys():
        edgeToRow_ms_tensor.append(Tensor(edgeToRow_np[key], dtype=ms.int32))

    # vect_agg 数据聚集的前准备
    SCATTER_ADD = ms.ops.TensorScatterAdd()
    GATHER = ms.ops.Gather()
    ZEROS = ms.ops.Zeros()
    SHAPE = ms.ops.Shape()
    RESHAPE = ms.ops.Reshape()

    window_count = (len(split_col_remaining_csr_indptr) - 1 + BLK_H - 1) // BLK_H
    A_row_count = len(split_col_remaining_csr_indptr) - 1
    B_row_count = len(np.unique(split_row_filtered_coo_rows))

    embedding_dim = node_feat.shape[1]  # 防止自定义数据node_feat
    A1 = Tensor(np.ones((A_row_count, embedding_dim), dtype=np.float16), ms.float16)
    A2_node_feat = ZEROS((A_row_count, embedding_dim), ms.float16)  # 先开辟结果空间
    B_node_feat = ZEROS((B_row_count, embedding_dim), ms.float16)  # 先开辟结果空间

    if len(split_col_filtered_coo_cols) != 0:  # A2为空时
        A2_dst_idx = Tensor(split_col_filtered_coo_rows, ms.int32)
        A2_dst_idx = RESHAPE(A2_dst_idx, (SHAPE(A2_dst_idx)[0], 1))  # 必须得二维以上
        A2_src_idx = Tensor(split_col_filtered_coo_cols, ms.int32)

    if len(split_row_filtered_coo_rows) != 0:  # B为空时
        B_dst_idx = Tensor(split_row_filtered_coo_rows, ms.int32)
        B_dst_idx = RESHAPE(B_dst_idx, (SHAPE(B_dst_idx)[0], 1))  # 必须得二维以上
        B_src_idx = Tensor(split_row_filtered_coo_cols, ms.int32)



    row_reindex = row_reindex_np.tolist()  # 特征重排序
    row_index = np.argsort(row_reindex_np).tolist()  # 特征重排序

    row_reindex_tensor = Tensor(row_reindex, dtype=ms.int32)
    row_index_tensor = Tensor(row_index, dtype=ms.int32)

    add_op = ops.Add()
    concat_op = ops.Concat(axis=0)
    repeat_time = 1

    for i in range(repeat_time):

        node_feat_sort = ops.gather(node_feat, row_reindex_tensor, 0)
        print(f"{node_feat_sort[0][0]}")

        # 计算公式 result = A1+A2拼接B
        # 第一部分cube计算  A1

        for winId in range(window_count):

            X_ms = ops.gather(
                node_feat_sort, edgeToRow_ms_tensor[winId], 0
            )  # 不需要重复gather对8192的reddit来说
            # print(f"X_ms的阻塞值为{X_ms[0][0]}")

            # print(f"特征按边gather排序的时间{(time2 - time1)* 1000:.8f}ms")
            start = winId * BLK_H
            end = min((winId + 1) * BLK_H, len(split_col_remaining_csr_indptr) - 1)

            adj_matrix_rows_pad = (
                16 - (adj_matrix[winId].shape[0] % 16)
                if adj_matrix[winId].shape[0] % 16 != 0
                else 0
            )
            X_ms_rows_pad = 16 - (X_ms.shape[0] % 16) if X_ms.shape[0] % 16 != 0 else 0
            X_ms_cols_pad = 16 - (X_ms.shape[1] % 16) if X_ms.shape[1] % 16 != 0 else 0

            adj_matrix_pad_op = ops.Pad(((0, adj_matrix_rows_pad), (0, X_ms_rows_pad)))
            X_ms_pad_op = ops.Pad(((0, X_ms_rows_pad), (0, X_ms_cols_pad)))

            # 应用填充
            padded_adj_matrix = adj_matrix_pad_op(adj_matrix[winId])
            padded_X_ms = X_ms_pad_op(X_ms)

            tmp = ops.matmul(padded_adj_matrix, padded_X_ms)  # 计算

            tmp = tmp[0 : (end - start), 0:embedding_dim].astype(
                ms.float16
            )  # 切割和类型转换
            A1[start:end] = tmp

        # #第二部分vector   A2

        if len(split_col_filtered_coo_cols) != 0:
            A2 = vector_agg(
                split_col_filtered_coo_rows,
                split_col_filtered_coo_cols,
                A2_dst_idx,
                A2_src_idx,
                node_feat_sort,
                A2_node_feat,
                A_row_count,
                batch_edgecount,
            )

        # #第三部分vector   B
        if len(split_row_filtered_coo_rows) != 0:
            B = vector_agg(
                split_row_filtered_coo_rows,
                split_row_filtered_coo_cols,
                B_dst_idx,
                B_src_idx,
                node_feat_sort,
                B_node_feat,
                B_row_count,
                batch_edgecount,
            )

        # 合并结果,result = A1+A2拼接B
        if len(split_col_filtered_coo_cols) != 0:
            new_node_feat = add_op(A1, A2)
        #  new_node_feat[0:A1.shape[0]] = A1+A2
        else:
            new_node_feat = A1

        if len(split_row_filtered_coo_rows) != 0:
            new_node_feat = concat_op((new_node_feat, B))

        # 特征排回原来的顺序
        new_node_feat = ops.gather(new_node_feat, row_index_tensor, 0)

    return new_node_feat


def test_vector(dataset_name, batch_edgecount, embedding_dim):

    npz = np.load(f"{dataset_path}/npz/{dataset_name}.npz")
    nodePointer_np = npz["adj_csr_indptr"].astype(np.int32)
    edgeList_np = npz["adj_csr_indices"].astype(np.int32)


    node_sum = len(nodePointer_np) - 1
    np.random.seed(24)
    node_feat = Tensor(np.random.rand(node_sum, embedding_dim), ms.float16)
    # node_feat = Tensor(np.ones((node_sum, embedding_dim),dtype=np.float16), ms.float16)

   
    embedding_dim = node_feat.shape[1]  # 防止自定义数据node_feat
    new_node_feat = Tensor(
        np.zeros((node_sum, embedding_dim), dtype=np.int32), ms.float16
    )

    SCATTER_ADD = ms.ops.TensorScatterAdd()
    GATHER = ms.ops.Gather()
    ZEROS = ms.ops.Zeros()
    SHAPE = ms.ops.Shape()
    RESHAPE = ms.ops.Reshape()


    sorted_coo_rows, sorted_coo_cols = csr_to_coo(nodePointer_np, edgeList_np)

    dst_idx = Tensor(sorted_coo_rows, ms.int32)
    dst_idx = RESHAPE(dst_idx, (SHAPE(dst_idx)[0], 1))  # 必须得二维以上
    src_idx = Tensor(sorted_coo_cols, ms.int32)

    repeat_time = 1
    for i in range(repeat_time):
       
        result = vector_agg(
            sorted_coo_rows,
            sorted_coo_cols,
            dst_idx,
            src_idx,
            node_feat,
            new_node_feat,
            node_sum,
            batch_edgecount,
        )

    return result


def compare_result(
    dataset_name,
    BLK_H,
    batch_edgecount,
    embedding_dim,
):


    result_mmad = run(dataset_name, BLK_H, batch_edgecount, embedding_dim)
    result_vector = test_vector(dataset_name, batch_edgecount, embedding_dim)

    print(f"result_mmad={result_mmad}")
    print(f"result_vector={result_vector}")

    are_equal_strict = result_vector.equal(result_mmad).all()
    

    result_vector_np = result_vector.asnumpy()
    result_mmad_np = result_mmad.asnumpy()
    

    sum = 0
    error = []
    for i in range(result_mmad.shape[0]):
        are_close = np.allclose(
            result_vector_np[i], result_mmad_np[i], rtol=1e-03, atol=1e-03
        )
        if are_close:
            #  print(f"第{i}个近似相等（允许较大容差）:")
            sum = sum + 1
            pass
        else:
            #  print(f"第{i}近似不相等（允许较大容差）")
            error.append(i)
    print(f"{sum} vertex features are equal")
    for i in range(min(len(error), 6)):
        print(f"---------------------Features of vertex {i} that are not equal---------------------")
        index = error[i]
        print(f"result_vector_np[{index}] = {result_vector_np[index]}")
        print(f"result_mmad_np[{index}] = {result_mmad_np[index]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test accuracy of GCN")
    parser.add_argument("--dataset_name", type=str, default="Cora", help="Name of the dataset (e.g., Cora, Pubmed, Citeseer)")
    parser.add_argument("--embedding_dim", type=int, default=10, help="Feature dimension used for validation")
    args = parser.parse_args()
   
    dataset_name = args.dataset_name
    embedding_dim = args.embedding_dim
     
    window_H = {
        "Cora": 81920,
        "Pubmed": 81920,
        "Citeseer": 81920,
        "products": 6400,
        "mycielskian18": 8192,
        "reddit": 8192,
    }
    edgecount = {
        "Cora": 5000000,
        "Pubmed": 5000000,
        "Citeseer": 5000000,
        "products": 10000000,
        "mycielskian18": 10000000,
        "reddit": 5000000,
    }
    # Check if dataset_name is valid
    if dataset_name not in window_H:
        print(f"Error: Dataset '{dataset_name}' is not supported. Please choose from {list(window_H.keys())}.")
        exit(1)
        
    BLK_H = window_H[dataset_name]
    batch_edgecount = edgecount[dataset_name]
    print(f"BLK_H = {BLK_H}")
    print(f"batch_edgecount = {batch_edgecount}")

    compare_result(dataset_name, BLK_H, batch_edgecount, embedding_dim)

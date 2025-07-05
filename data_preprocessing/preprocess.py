import time
import numpy as np
import math
import random

from concurrent.futures import ThreadPoolExecutor, as_completed
import pymetis

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor, COOTensor
from mindspore import context, profiler
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


    
def split_row(min_occurrences, nodePointer, coo_rows, coo_cols, metis_index):
    """
    Splits the rows into two parts: those with neighbor count greater than min_occurrences,
    and the rest. Then reindexes all nodes accordingly.

    Args:
        min_occurrences: threshold for filtering (include only rows with neighbor count > min_occurrences)
        nodePointer: row index in CSR format
        coo_rows: row indices in COO format
        coo_cols (numpy.ndarray): column indices in COO format
        metis_index (numpy.ndarray): current node index mapping

    Returns:
       indptr (numpy.ndarray): row pointer for CSR format
       indices(numpy.ndarray): column indices for CSR format
       sorted_rows(numpy.ndarray): sorted row indices in COO format  
       sorted_cols(numpy.ndarray): sorted column indices in COO format
       row_reindex (numpy.ndarray): new index mapping after sorting
    """
    neighborCount = np.diff(nodePointer).astype(np.int32)  # number of neighbors per node
    filtered_node = len(nodePointer) - 1  # number of rows being filtered out
    nodeCount = len(nodePointer) - 1
    mask = np.zeros(len(coo_cols), dtype=bool)


    for i in range(nodeCount):
        if neighborCount[i] > min_occurrences:
            start_idx = nodePointer[i]
            end_idx = nodePointer[i + 1]
            mask[start_idx:end_idx] = True
            filtered_node -= 1

    remaining_coo_rows = coo_rows[mask]
    remaining_coo_cols = coo_cols[mask]
    filtered_coo_rows = coo_rows[~mask]
    filtered_coo_cols = coo_cols[~mask]

    # Determine which nodes should be kept at the front
    mask = np.zeros(len(metis_index), dtype=bool)
    for i in range(len(metis_index)):  # check whether this node should come first
        id = metis_index[i]
        if neighborCount[id] > min_occurrences:
            mask[i] = True

    row_remain = metis_index[mask]   # indices of retained nodes
    row_filter = metis_index[~mask]  # indices of filtered nodes
    row_reindex = np.concatenate((row_remain, row_filter))  # final index mapping for each node

    # Reindexing rows
    print(f"-----------------Row Reindexing Started---------------")
    remaining_coo_rows_transformed = rein_index(row_reindex, remaining_coo_rows)
    filtered_coo_rows_transformed = rein_index(row_reindex, filtered_coo_rows)
    print(f"-----------------Row Reindexing Completed---------------")

    # Reindexing columns
    print(f"-----------------Column Reindexing Started---------------")
    remaining_coo_cols_transformed = rein_index(row_reindex, remaining_coo_cols)
    filtered_coo_cols_transformed = rein_index(row_reindex, filtered_coo_cols)
    print(f"-----------------Column Reindexing Completed---------------")

    return remaining_coo_rows_transformed, remaining_coo_cols_transformed, filtered_coo_rows_transformed, filtered_coo_cols_transformed, row_reindex

def split_col(min_occurrences, coo_rows, coo_cols):
    
    unique_cols, counts = np.unique(coo_cols, return_counts=True)  # Get unique column indices and their counts
    cols_to_keep = unique_cols[counts > min_occurrences]  # Columns with occurrence greater than min_occurrences
    mask = np.isin(coo_cols, cols_to_keep)  # Boolean array indicating which elements to keep
    filtered_node = len(unique_cols) - len(cols_to_keep)

    remaining_coo_rows = coo_rows[mask]
    remaining_coo_cols = coo_cols[mask]
    filtered_coo_rows = coo_rows[~mask]
    filtered_coo_cols = coo_cols[~mask]

    # Build new CSR row pointer array
    # Note: Some rows might be completely removed, so we need to reconstruct the CSR row index to avoid missing nodes
    print("-----------------Start building split CSR row pointers-----------------")
    n_nodes = np.max(coo_rows) + 1
    row_counts = np.bincount(remaining_coo_rows, minlength=n_nodes)
    remaining_csr_rows = np.zeros(n_nodes + 1, dtype=np.int32)
    remaining_csr_rows[1:] = np.cumsum(row_counts)
    print("-----------------Finished building split CSR row pointers-----------------")
    print(f"remaining_coo_rows: {len(remaining_coo_rows)}, filtered_coo_rows: {len(filtered_coo_rows)}")
    print(f"remaining_coo_cols: {len(remaining_coo_cols)}, filtered_coo_cols: {len(filtered_coo_cols)}")

    # Row-wise sorting
    # Note: remaining_coo_rows and remaining_coo_cols may not be in order, need to sort them
    sorted_indices = np.argsort(remaining_coo_rows)  # Get sorting indices
    remaining_coo_rows = remaining_coo_rows[sorted_indices]
    remaining_coo_cols = remaining_coo_cols[sorted_indices]

    return remaining_csr_rows, remaining_coo_rows, remaining_coo_cols, filtered_coo_rows, filtered_coo_cols
  
def rein_index(A, B):
    # Replace each value in B with its corresponding index in A

    # Step 1: Create a mapping from values in A to their indices
    unique_vals, idxs = np.unique(A, return_index=True)
    sorted_idx = idxs[np.argsort(unique_vals)]
    sorted_vals = A[sorted_idx]

    # Step 2: Find the positions of elements in B within A
    indices_in_A = np.searchsorted(sorted_vals, B)

    # Step 3: Ensure that the found positions correspond to matching values in B
    mask = sorted_vals[indices_in_A] == B

    # Step 4: Transform B into its new index-mapped form based on A
    B_transformed = np.copy(B)  # Copy B to avoid in-place modification
    B_transformed[mask] = sorted_idx[indices_in_A[mask]]

    return B_transformed

def process_window(winId, csr_rows, csr_cols, TC_BLK_h):
    # Get the edge index range for the current row window
    print(f"Window {winId} started execution")

    winStart = csr_rows[winId * TC_BLK_h]
    winEnd = csr_rows[min((winId + 1) * TC_BLK_h, len(csr_rows) - 1)]

    eArray = sorted(csr_cols[winStart:winEnd])

    # Deduplicate while preserving order
    eArrClean = list(dict.fromkeys(eArray))
    # Build a fast lookup map from value to index
    eArrClean_index_map = {value: idx for idx, value in enumerate(eArrClean)}

    # Map each edge to its compressed column index within this window
    edgeToColResult = [eArrClean_index_map[csr_cols[eIndex]] for eIndex in range(winStart, winEnd)]

    print(f"Window {winId} completed")
    return winId, edgeToColResult


def prepare_edge_arrays(csr_rows, csr_cols, numNodes, BLK_H, dataset_name, flag):
    """
    Preprocess the edgeList data and create edge_array for each winId.
    Returns a list where each element corresponds to edge_array of a winId.
    """

    numRowWindows = ((numNodes + BLK_H - 1) // BLK_H)
    edgeToRow = [[] for _ in range(numRowWindows)]
    edgeToColumn = [None] * csr_cols.shape[0]

    print("Length of edgeToColumn:", len(edgeToColumn))
    print(f"Sorting for {numRowWindows} windows")

    for winId in range(numRowWindows):  # Process in blocks
        eIdx_start = int(csr_rows[winId * BLK_H])
        eIdx_end = int(csr_rows[min((winId + 1) * BLK_H, numNodes)])

        # Get edgeList data for current block
        edge_array = csr_cols[eIdx_start:eIdx_end]

        # Deduplicate and sort
        edge_array = np.unique(edge_array)
        edgeToRow[winId] = edge_array
        print(f"Sorting for window {winId}")

    print("Row window sorting completed")

    if flag:
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {
                executor.submit(process_window, winId, csr_rows, csr_cols, BLK_H): winId
                for winId in range(numRowWindows)
            }
            for future in as_completed(futures):
                winId, edgeToColResult = future.result()
                print("Completed winId: ", winId)

                # Collect local result before updating globally
                winStart = csr_rows[winId * BLK_H]
                winEnd = csr_rows[min((winId + 1) * BLK_H, len(csr_rows) - 1)]
                edgeToColumn[winStart:winEnd] = edgeToColResult

    if flag:
        edgeToColumn_np = np.array(edgeToColumn).astype(np.int32)
        edgeToColumn_np.tofile(
            f"{dataset_path}/mmad/{dataset_name}/{dataset_name}_edgeToColumn.bin"
        )
    else:
        edgeToColumn = np.fromfile(
            f"{dataset_path}/mmad/{dataset_name}/{dataset_name}_edgeToColumn.bin",
            dtype=np.int32,
        )

    # edgeToRow is not saved because each window has different number of columns, which NumPy does not support directly
    print("Window sequence construction completed")
    return edgeToRow, edgeToColumn  # Two lists

      
def prepare_adj_matrix(nodePointer_np, edgeList_np, BLK_H, dataset_name, flag):
    """
    Prepare the adjacency matrix for each row window using edge arrays.

    Args:
        nodePointer_np (np.int64): CSR row pointer array.
        edgeList_np (np.int64): CSR column indices array.
        BLK_H (int): Block height for windowing.
        dataset_name (str): Name of the dataset.
        flag (bool): Whether to recompute and save edgeToColumn.

    Returns:
        adj_matrix (list): List of dense tensors, one per window.
        edgeToRow_ms_tensor (list): List of edgeToRow tensors.
    """

    edgeToRow, edgeToColumn = prepare_edge_arrays(
        nodePointer_np, edgeList_np, len(nodePointer_np) - 1, BLK_H, dataset_name, flag
    )

    numNodes = len(nodePointer_np) - 1

    for i in range(len(edgeToRow)):
        print(f"Length of edgeToRow[{i}]: {len(edgeToRow[i])}")
        if i == len(edgeToRow) - 1:
            continue
        if np.array_equal(edgeToRow[i], edgeToRow[i + 1]):
            print(f"Window {i} and window {i + 1} have identical gathered rows")

    adj_matrix = []
    edgeToRow_ms_tensor = []

    for winId in range((numNodes + BLK_H - 1) // BLK_H):  # Process in blocks
        edgeToRow_ms_tensor.append(Tensor(edgeToRow[winId]))

    for winId in range((numNodes + BLK_H - 1) // BLK_H):  # Process in blocks
        start = winId * BLK_H
        end = min((winId + 1) * BLK_H, numNodes)
        eIdx_start = int(nodePointer_np[start])
        eIdx_end = int(nodePointer_np[end])

        crow_indices = (nodePointer_np[start : (end + 1)] - nodePointer_np[start]).tolist()

        crow_indices_np = np.repeat(
            np.arange(len(crow_indices) - 1, dtype=np.int32),
            np.diff(crow_indices).astype(np.int32),
        )  # Expand row indices to match column count

        row_indices_ms = Tensor(crow_indices_np.tolist(), dtype=ms.int32)
        col_indices_ms = Tensor(edgeToColumn[eIdx_start:eIdx_end], dtype=ms.int32)
        values_ms = Tensor(np.ones(len(col_indices_ms), np.float16), dtype=ms.float16)

        sparse_tensor_ms = COOTensor(
            ops.Stack(axis=1)((row_indices_ms, col_indices_ms)),
            values_ms,
            (len(crow_indices) - 1, len(edgeToRow[winId])),
        )

        dense_tensor_ms = sparse_tensor_ms.to_dense()
        adj_matrix.append(dense_tensor_ms)

    return adj_matrix, edgeToRow_ms_tensor


def prepare_data(dataset_name, n_cuts, BLK_H, min_occurrences_row, min_occurrences_col):
    """
    Prepare data by filtering and partitioning the adjacency matrix.

    Args:
        dataset_name (str): Name of the dataset.
        n_cuts (int): Number of subgraphs to divide into.
        BLK_H (int): Block height for matrix operations.
        min_occurrences_row (int): Threshold for filtering rows by neighbor count.
        min_occurrences_col (int): Threshold for filtering columns by occurrence count.

    Returns:
        Files saved under `{dataset_path}/mmad/{dataset_name}` directory.
    """

    npz = np.load(f"{dataset_path}/npz/{dataset_name}.npz")
    nodePointer_np = npz["adj_csr_indptr"].astype(np.int32)
    edgeList_np = npz["adj_csr_indices"].astype(np.int32)

    node_sum = nodePointer_np.shape[0] - 1
    edge_sum = edgeList_np.shape[0]

    n_cuts = 8
    metis_index = np.arange(0, node_sum)
    coo_rows, coo_cols = csr_to_coo(nodePointer_np, edgeList_np)

    print(f"Number of coo_rows: {len(coo_rows)}")
    print(f"Number of coo_cols: {len(coo_cols)}")
    print("-----------------Finished reading COO data-------------------------------")

    print(f"------------------Adjacency matrix row partitioning started---------------")

    (
        split_row_remaining_coo_rows,
        split_row_remaining_coo_cols,
        split_row_filtered_coo_rows,
        split_row_filtered_coo_cols,
        row_reindex,
    ) = split_row(min_occurrences_row, nodePointer_np, coo_rows, coo_cols, metis_index)

    A_row_count = int(np.max(split_row_remaining_coo_rows) + 1) if split_row_remaining_coo_rows.size > 0 else 0
    B_row_count = int(node_sum - A_row_count) if split_row_filtered_coo_rows.size > 0 else 0

    print(f"------------------Adjacency matrix row partitioning completed---------------")

    print(f"------------------Adjacency matrix column partitioning started---------------")

    (
        split_col_remaining_csr_indptr,
        split_col_remaining_coo_rows,
        split_col_remaining_coo_cols,
        split_col_filtered_coo_rows,
        split_col_filtered_coo_cols,
    ) = split_col(min_occurrences_col, split_row_remaining_coo_rows, split_row_remaining_coo_cols)

    print(f"------------------Adjacency matrix column partitioning completed---------------")

    print(f"------------------A1 involves {len(split_col_remaining_coo_cols)} edges---------------")
    print(f"------------------A2 involves {len(split_col_filtered_coo_cols)} edges---------------")
    print(f"------------------B involves {len(split_row_filtered_coo_cols)} edges---------------")

    print(f"------------------First part of data preparation completed---------------")

    sorted_indices = np.lexsort((split_col_filtered_coo_cols, split_col_filtered_coo_rows))
    split_col_filtered_coo_rows = split_col_filtered_coo_rows[sorted_indices]
    split_col_filtered_coo_cols = split_col_filtered_coo_cols[sorted_indices]
    print(f"------------------Second part of data preparation completed---------------")

    sorted_indices = np.lexsort((split_row_filtered_coo_cols, split_row_filtered_coo_rows))
    split_row_filtered_coo_rows = split_row_filtered_coo_rows[sorted_indices]
    split_row_filtered_coo_cols = split_row_filtered_coo_cols[sorted_indices]
    row_sort = np.unique(split_row_filtered_coo_rows)
    split_row_filtered_coo_rows = rein_index(row_sort, split_row_filtered_coo_rows)

    print(f"------------------Third part of data preparation completed---------------")

    print(f"------------------A1 involves {len(split_col_remaining_coo_cols)} edges---------------")
    print(f"------------------A2 involves {len(split_col_filtered_coo_cols)} edges---------------")
    print(f"------------------B involves {len(split_row_filtered_coo_cols)} edges---------------")

    print(f"------------------Data saving started------------------")

    row_reindex.astype(np.int32).tofile(f"{dataset_path}/mmad/{dataset_name}/row_reindex.bin")
    split_col_remaining_csr_indptr.astype(np.int32).tofile(f"{dataset_path}/mmad/{dataset_name}/split_col_remaining_csr_indptr.bin")
    split_col_remaining_coo_cols.astype(np.int32).tofile(f"{dataset_path}/mmad/{dataset_name}/split_col_remaining_coo_cols.bin")

    split_col_filtered_coo_rows.astype(np.int32).tofile(f"{dataset_path}/mmad/{dataset_name}/split_col_filtered_coo_rows.bin")
    split_col_filtered_coo_cols.astype(np.int32).tofile(f"{dataset_path}/mmad/{dataset_name}/split_col_filtered_coo_cols.bin")
    split_row_filtered_coo_rows.astype(np.int32).tofile(f"{dataset_path}/mmad/{dataset_name}/split_row_filtered_coo_rows.bin")
    split_row_filtered_coo_cols.astype(np.int32).tofile(f"{dataset_path}/mmad/{dataset_name}/split_row_filtered_coo_cols.bin")

    print(f"Length of split_col_remaining_csr_indptr: {len(split_col_remaining_csr_indptr)}")

    adj_matrix, edgeToRow_ms_tensor = prepare_adj_matrix(
        split_col_remaining_csr_indptr, split_col_remaining_coo_cols, BLK_H, dataset_name, True
    )

    adj_matrix_np = []
    for i in range(len(adj_matrix)):
        adj_matrix_np.append(adj_matrix[i].asnumpy())
    np.savez(f"{dataset_path}/mmad/{dataset_name}/adj_matrix_np.npz", *adj_matrix_np)

    edgeToRow_np = []
    for i in range(len(edgeToRow_ms_tensor)):
        edgeToRow_np.append(edgeToRow_ms_tensor[i].asnumpy())
    np.savez(f"{dataset_path}/mmad/{dataset_name}/edgeToRow_np.npz", *edgeToRow_np)

    print(f"------------------Data preprocessing completed------------------")
    

    

if __name__ == "__main__":
    #metis有问题
    dataset_name = "Cora" #Cora、Citeseer、Pubmed、mycielskian18、reddit、products
    n_cuts = 8
    window_H = {"Cora": 81920, "Pubmed": 81920, "Citeseer": 81920, "products": 6400, "mycielskian18": 8192, "reddit": 8192}
    row = {"Cora": 0, "Pubmed": 5, "Citeseer": 0, "products": 200, "mycielskian18": 1530, "reddit": 600}
    col = {"Cora": 0, "Pubmed": 5, "Citeseer": 0, "products": 50, "mycielskian18": 1000, "reddit": 600}
    edgecount = {"Cora": 5000000, "Pubmed": 5000000, "Citeseer": 5000000, "products": 10000000, "mycielskian18": 10000000, "reddit": 10000000}

    if dataset_name in window_H:
        BLK_H = window_H[dataset_name]
        min_occurrences_row = row[dataset_name]
        min_occurrences_col = col[dataset_name]
        batch_edgecount = edgecount[dataset_name]
        embedding_dim = 32
        print(f"BLK_H = {BLK_H}")
        print(f"min_occurrences_row = {min_occurrences_row}")
        print(f"min_occurrences_col = {min_occurrences_col}")
        print(f"batch_edgecount = {batch_edgecount}")
        if not os.path.exists(f"{dataset_path}/mmad/{dataset_name}"):
            os.makedirs(f"{dataset_path}/mmad/{dataset_name}")
        prepare_data(dataset_name, n_cuts, BLK_H, min_occurrences_row, min_occurrences_col)
       
    else:
        print(f"Dataset name '{dataset_name}' is temporarily not supported.")

    

    
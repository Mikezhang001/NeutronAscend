import dgl
from dgl import data
import numpy as np
def coo_to_csr(coo_rows, coo_cols):
    # print("排序开始")
    # 对行索引排序，同一行的数据在一起
    sorted_indices = np.lexsort(
        (coo_cols, coo_rows)
    )  # 使用 np.lexsort() 先排后面的参数
    sorted_rows = coo_rows[sorted_indices]
    sorted_cols = coo_cols[sorted_indices]

    # 计算每一行的非零元素数量，确保覆盖所有节点
    # print(f"coo行中最大值为{sorted_rows.max()}")
    # print(f"coo列中最大值为{sorted_cols.max()}")
    n_nodes = max(sorted_rows.max(), sorted_cols.max()) + 1  # 总顶点数
    row_counts = np.bincount(sorted_rows, minlength=n_nodes)

    # 构造 indptr 数组
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(row_counts)

    # print(f"indptr的长度为{len(indptr)}")
    return indptr, sorted_cols, sorted_rows, sorted_cols

def load_and_print_info(dataset_class, name, save_path=None):
    # 加载数据集
    dataset = dataset_class()
    g = dataset[0]

    # 提取特征、标签、邻接表等 
    n_nodes = np.array(g.num_nodes(), dtype=np.int64)
    n_edges = np.array(g.num_edges(), dtype=np.int64)
    n_classes = np.array(dataset.num_classes, dtype=np.int64)
    
    feat = g.ndata['feat'].numpy()  # 节点特征
    label = g.ndata['label'].numpy()   # 节点标签
    
    train_mask = g.ndata['train_mask'].numpy()
    val_mask = g.ndata['val_mask'].numpy()
    test_mask = g.ndata['test_mask'].numpy()
    print(f"train_mask长度为{np.count_nonzero(train_mask)}")
    print(f"val_mask长度为{np.count_nonzero(val_mask)}")
    print(f"test_mask长度为{np.count_nonzero(test_mask)}")

    # 邻接表（源节点和目标节点）
    src, dst = g.edges()
    adj_coo_row = src.numpy()
    adj_coo_col = dst.numpy()
    adj_csr_indptr, adj_csr_indices, adj_coo_row, adj_coo_col = coo_to_csr(adj_coo_row, adj_coo_col)
    
    in_degrees = g.in_degrees().numpy()
    out_degrees = g.out_degrees().numpy()


    #如果提供了路径，则保存为 .npz 文件
    if save_path:
        np.savez(
            save_path,
            n_nodes = n_nodes,
            n_edges = n_edges,
            n_classes = n_classes,
            feat = feat,
            label = label,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask,
            adj_csr_indptr =  adj_csr_indptr,
            adj_csr_indices = adj_csr_indices,
            adj_coo_row = adj_coo_row,
            adj_coo_col = adj_coo_col,
            in_degrees = in_degrees,
            out_degrees = out_degrees
        )
        print(f"{name} 数据已保存到 {save_path}")
        print("---------------------------------------")
        
load_and_print_info(data.CoraGraphDataset, 'Cora', "../data/npz/Cora")
load_and_print_info(data.PubmedGraphDataset, 'Pubmed', "../data/npz/Pubmed")
load_and_print_info(data.CiteseerGraphDataset, 'Citeseer', "../data/npz/Citeseer")
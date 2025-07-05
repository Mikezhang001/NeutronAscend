from scipy.io import mmread
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
def generate_masks(n_nodes, num_labels=128, embedding_dim=256):
    # 初始化 label 数组
    labels = np.random.randint(0, num_labels, size=n_nodes)  # 随机生成 x 个标签，范围 [0, num_labels-1]
    
    # 初始化 mask 数组
    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)
    
    # 按照比例划分索引
    indices = np.arange(n_nodes)  # 获取所有顶点索引
    np.random.shuffle(indices)  # 打乱索引顺序
    
    train_size = int(0.48 * n_nodes)  # 训练集大小
    val_size = int(0.32 * n_nodes)    # 验证集大小
    test_size = n_nodes - train_size - val_size  # 测试集大小
    
    # 分配索引到各个掩码
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    #生成特征
    features = np.random.rand(n_nodes, embedding_dim).astype(np.float32)
    return labels, train_mask, val_mask, test_mask, features

def read_file(dataset_name):
    # 1. 读取 mtx 文件（Matrix Market 格式）
    matrix = mmread(f'./{dataset_name}/{dataset_name}.mtx')
    save_path = f"./{dataset_name}"
    # 2. 转换为 COO 格式
    coo = coo_matrix(matrix)
    adj_coo_row = coo.row
    adj_coo_col = coo.col
    
    csr = csr_matrix(matrix)
    adj_csr_indptr = csr.indptr
    adj_csr_indices = csr.indices
    
    n_nodes = len(adj_csr_indptr)-1
    n_edges = len(adj_coo_col)
    
    n_classes = 16
    label, train_mask, val_mask, test_mask, feat = generate_masks(n_nodes, num_labels=n_classes, embedding_dim=256) 
    
    #出度
    out_degrees = np.zeros(n_nodes, dtype=np.int32)  # 初始化出度数组
    unique_rows, row_counts = np.unique(adj_coo_row, return_counts=True)
    out_degrees[unique_rows] = row_counts  # 更新出度 
    print(f"{dataset_name}数据集的出度准备完成")
    
    #入度
    in_degrees = np.zeros(n_nodes, dtype=np.int32)  # 初始化入度数组
    unique_cols, col_counts = np.unique(adj_coo_col, return_counts=True)
    in_degrees[unique_cols] = col_counts  # 更新入度
    print(f"{dataset_name}数据集的入度准备完成")
    
    print(f"顶点数目为{n_nodes}")
    print(f"边数目为{n_edges}")
    print(f"种类为{n_classes}")
    print(f"feat的形状为{feat.shape}")   
    print(f"feat的数据类型为{type(feat[0][0])}")
    print(f"label的长度为{len(label)}")
    print(f"train_mask的长度为{len(train_mask)}")
    print(f"test_mask的长度为{len(test_mask)}")
    print(f"val_mask的长度为{len(val_mask)}")
    print(f"adj_coo_row的长度为{len(adj_coo_row)}")
    print(f"adj_coo_col的长度为{len(adj_coo_col)}")    
    
    np.savez(
            save_path,
            n_nodes = np.array(n_nodes, dtype=np.int64),
            n_edges = np.array(n_edges, dtype=np.int64),
            n_classes = np.array(n_classes, dtype=np.int64),
            feat = feat.astype(np.float32),
            label = label.astype(np.int64),
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask,
            adj_csr_indptr =  adj_csr_indptr.astype(np.int64),
            adj_csr_indices = adj_csr_indices.astype(np.int64),
            adj_coo_row = adj_coo_row.astype(np.int64),
            adj_coo_col = adj_coo_col.astype(np.int64),
            in_degrees = in_degrees.astype(np.int64),
            out_degrees = out_degrees.astype(np.int64)
        )
    print(f"{dataset_name} 数据已保存到 {save_path}")
    print("---------------------------------------")
if __name__ == "__main__":
    dataset_name = "mycielskian18"
    read_file(dataset_name)
# 数据目录说明

| 文件夹 | 说明 |
|--------|------|
| npz    | 预处理前文件夹 |
| mmad   | 预处理后文件夹（由 data_preprocessing/preprocess.py 对 mmad 下 npz 文件处理得到） |

---

## `Cora.npz` 文件结构说明

该文件是 Cora 图数据集的标准格式，包含图结构、节点特征、标签及训练/验证/测试划分信息。

### 文件键值说明

| 键名             | 数据类型 | 形状         | 描述 |
|------------------|----------|--------------|------|
| `n_nodes`        | `int64`  | `()`         | 节点总数 |
| `n_edges`        | `int64`  | `()`         | 边总数 |
| `n_classes`      | `int64`  | `()`         | 分类类别数 |
| `feat`           | `float32`| `(2708, 1433)` | 节点特征矩阵，每行对应一个节点的特征 |
| `label`          | `int64`  | `(2708,)`    | 每个节点的类别标签 |
| `train_mask`     | `bool`   | `(2708,)`    | 训练集掩码（True 表示该节点属于训练集） |
| `val_mask`       | `bool`   | `(2708,)`    | 验证集掩码 |
| `test_mask`      | `bool`   | `(2708,)`    | 测试集掩码 |
| `adj_csr_indptr` | `int64`  | `(2709,)`    | CSR 格式邻接矩阵的索引指针数组 |
| `adj_csr_indices`| `int64`  | `(10556,)`   | CSR 格式邻接矩阵的列索引数组 |
| `adj_coo_row`    | `int64`  | `(10556,)`   | COO 格式的边起点索引（行索引） |
| `adj_coo_col`    | `int64`  | `(10556,)`   | COO 格式的边终点索引（列索引） |
| `in_degrees`     | `int64`  | `(2708,)`    | 每个节点的入度 |
| `out_degrees`    | `int64`  | `(2708,)`    | 每个节点的出度 |

---

## 数据集来源

| 数据集      | 来源 |
|-------------|------|
| Cora        | 见 data_preprocessing/download_data.py |
| Citeseer    | 见 data_preprocessing/download_data.py |
| Pubmed      | 见 data_preprocessing/download_data.py |
| reddit      | [下载链接]() |
| products    | [下载链接]() |
| mycielskian18 | [下载链接](https://suitesparse-collection-website.herokuapp.com/MM/Mycielski/mycielskian18.tar.gz)<br>处理脚本: data_preprocessing/read_mycielskian.py |

---
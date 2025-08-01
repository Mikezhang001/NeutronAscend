# Data Directory Description

| Folder | Description |
|--------|-------------|
| npz    | Folder before preprocessing |
| mmad   | Folder after preprocessing (generated by processing npz files in mmad using data_preprocessing/preprocess.py) |

---

## `Cora.npz` File Structure Description

This file is the standard format of the Cora graph dataset, containing graph structure, node features, labels, and training/validation/test split information.

### File Key Description

| Key Name         | Data Type | Shape         | Description |
|------------------|-----------|---------------|-------------|
| `n_nodes`        | `int64`   | `()`          | Total number of nodes |
| `n_edges`        | `int64`   | `()`          | Total number of edges |
| `n_classes`      | `int64`   | `()`          | Number of classification categories |
| `feat`           | `float32` | `(2708, 1433)`| Node feature matrix, each row corresponds to the features of a node |
| `label`          | `int64`   | `(2708,)`     | Category label for each node |
| `train_mask`     | `bool`    | `(2708,)`     | Training set mask (True indicates the node belongs to the training set) |
| `val_mask`       | `bool`    | `(2708,)`     | Validation set mask |
| `test_mask`      | `bool`    | `(2708,)`     | Test set mask |
| `adj_csr_indptr` | `int64`   | `(2709,)`     | Index pointer array of adjacency matrix in CSR format |
| `adj_csr_indices`| `int64`   | `(10556,)`    | Column index array of adjacency matrix in CSR format |
| `adj_coo_row`    | `int64`   | `(10556,)`    | Row indices of edges in COO format |
| `adj_coo_col`    | `int64`   | `(10556,)`    | Column indices of edges in COO format |
| `in_degrees`     | `int64`   | `(2708,)`     | In-degree of each node |
| `out_degrees`    | `int64`   | `(2708,)`     | Out-degree of each node |

---

## Dataset Sources

| Dataset      | Source |
|--------------|--------|
| Cora         | See data_preprocessing/download_data.py |
| Citeseer     | See data_preprocessing/download_data.py |
| Pubmed       | See data_preprocessing/download_data.py |
| reddit       | [Download Link](https://data.dgl.ai/dataset/reddit.zip) |
| products     | [Download Link](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products) |
| mycielskian18| [Download Link](https://suitesparse-collection-website.herokuapp.com/MM/Mycielski/mycielskian18.tar.gz)<br>Processing script: data_preprocessing/read_mycielskian.py |

---

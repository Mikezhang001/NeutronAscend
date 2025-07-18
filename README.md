# Table of Contents
- [1. Environment Requirements](#1-environment-requirements)
- [2. Environment Setup](#2-environment-setup)
  - [2.1 Download](#21-download)
  - [2.2 Installation](#22-installation)
- [3. Operator Deployment](#3-operator-deployment)
- [4. Project Start](#4-project-start)
- [5. Acknowledgments](#5-acknowledgments)

---

# 1. Environment Requirements
| Software/Hardware | Version/Model |
|-------------------|---------------|
| **Server**        | Atlas 800T A2 Training Server<br>NPU Model: Ascend910B3<br>CPU Architecture: AArch64 |
| **Firmware**      | Ascend HDK 25.0.RC1<br>Package Name: Ascend-hdk-910b-npu-firmware_7.7.0.1.231.run |
| **Driver**        | Ascend HDK 25.0.RC1<br>Package Name: Ascend-hdk-910b-npu-driver_25.0.rc1.1_linux-aarch64.run |
| **MindSpore**     | 2.3.1 |
| **MindSpore_GL**  | 0.2 |
| **MindInsight**   | 2.3.1 |

---

# 2. Environment Setup

## 2.1 Download

### 2.1.1 [Firmware and Driver](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.1.RC1.alpha002&driver=Ascend+HDK+25.0.RC1)
![Firmware and Driver Installation Options](./images/fireware.PNG)

### 2.1.2 [CANN](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.1.RC1.alpha002)
![CANN Installation Options](./images/CANN.png)

---

## 2.2 Installation

### 2.2.1 Firmware and Driver
Reference Link: [Install NPU Driver Firmware](https://support.huawei.com/enterprise/zh/doc/EDOC1100349380/ac9d2505)

1. **Log in to the server as `root` user.**
2. **Create a driver runtime user `HwHiAiUser` (user for running driver processes).**
   ```bash
   groupadd HwHiAiUser
   useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
   ```
3. **Upload the driver and firmware packages to any directory on the server, e.g., `/home`.**
4. **Add executable permissions to the driver and firmware packages.**
   ```bash
   chmod +x Ascend-hdk-910b-npu-driver_25.0.rc1.1_linux-aarch64.run
   chmod +x Ascend-hdk-910b-npu-firmware_7.7.0.1.231.run
   ```
5. **Install the driver and firmware.**
   - **Install the driver**
     ```bash
     ./Ascend-hdk-910b-npu-driver_25.0.rc1.1_linux-aarch64.run --full --install-for-all
     ```
     > If missing tools error occurs:
     > ```
     > [ERROR]The list of missing tools: lspci,ifconfig,
     > ```
     > Refer to [Driver Installation Missing Dependency Error](https://support.huawei.com/enterprise/zh/doc/EDOC1100349380/3652fc47#ZH-CN_TOPIC_0000001782749677).

     > If DKMS compilation fails:
     > ```
     > [ERROR]Dkms install failed, details in : var/log/ascend_seclog/ascend_install.log.
     > ```
     > Refer to [DKMS Compilation Failure During Driver Installation](https://support.huawei.com/enterprise/zh/doc/EDOC1100349380/64a31720#ZH-CN_TOPIC_0000001784970501).

     > Successful installation message:
     > ```
     > Driver package installed successfully!
     > ```

   - **Install the firmware**
     ```bash
     ./Ascend-hdk-910b-npu-firmware_7.7.0.1.231.run --full
     ```
     > Successful installation message:
     > ```
     > Firmware package installed successfully! Reboot now or after driver installation for the installation/upgrade to take effect.
     > ```
6. **Restart the system (optional).**
   ```bash
   reboot
   ```
7. **Verify driver loading.**
   ```bash
   npu-smi info
   ```
   > **Note: Non-root users need to add HwHiAiUser.**
   > ```bash
   > sudo usermod -aG HwHiAiUser username
   > ```

---

### 2.2.2 CANN
Reference Link: [Install CANN](https://zhuanlan.zhihu.com/p/719099792)

1. **Switch to root user.**
   ```bash
   sudo su
   ```
2. **Modify CANN package permissions.**
   ```bash
   chmod +x Ascend-cann-kernels-910b_8.1.RC1.alpha002_linux-aarch64.run
   chmod +x Ascend-cann-toolkit_8.1.RC1.alpha002_linux-aarch64.run
   ```
3. **Install CANN.**
   - Remove old versions (if needed):
     ```bash
     rm -rf /usr/local/Ascend/ascend-toolkit
     ```
   - Install the toolkit:
     ```bash
     ./Ascend-cann-toolkit_8.1.RC1.alpha002_linux-aarch64.run --install
     ```
   - Install the kernel package:
     ```bash
     ./Ascend-cann-kernels-910b_8.1.RC1.alpha002_linux-aarch64.run
     ```
4. **Verify installation.**
   ```bash
   ls /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/
   ```
5. **Configure environment variables.**
   ```bash
   vim ~/.bashrc
   # Add the following:
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

---

### 2.2.3 MindSpore
Reference Link: [Install MindSpore](https://zhuanlan.zhihu.com/p/719099792)

1. **Create a Python environment.**
   ```bash
   conda create -n mindspore python=3.9
   conda activate mindspore
   ```
2. **Install dependencies.**
   ```bash
   pip install sympy
   pip install numpy==1.26
   pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl
   pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-*-py3-none-any.whl
   ```
3. **Install MindSpore and MindInsight.**
   ```bash
   pip install mindspore==2.3.1
   pip install mindinsight==2.3.1
   ```
4. **Verify installation.**
   ```bash
   python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"
   ```
   > Successful installation message:
   > ```
   > The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
   > ```

---

### 2.2.4 MindSpore_GL
Reference Link: [MindSpore Graph Learning](https://gitee.com/mindspore/graphlearning)

1. **Download the source code.**
   ```bash
   git clone https://gitee.com/mindspore/graphlearning.git
   ```
2. **Compile and install.**
   ```bash
   cd graphlearning
   bash build.sh
   pip install ./output/mindspore_gl*.whl
   ```
3. **Verify installation.**
   ```bash
   python -c 'import mindspore_gl'
   ```
   > If no error "No module named 'mindspore_gl'" occurs, the installation is successful.

---

# 3. Operator Deployment

1. **Compile the operator project.**
   ```bash
   sudo su root
   conda activate mindspore
   cd MmadCustom
   ./build.sh
   ```
2. **Declare environment variables.**
   ```bash
   vim ~/.bashrc
   export ASCEND_CUSTOM_OPP_PATH={build_out_path}build_out/_CPack_Packages/Linux/External/custom_opp_openEuler_aarch64.run/packages/vendors/customize:$ASCEND_CUSTOM_OPP_PATH
   source ~/.bashrc
   ```
3. **Test normal invocation.**
   ```bash
   python ../test/test_mmad.py
   ```

---

# 4. Project Start

## 4.1. NeutronAscend
1. **Data Preprocessing.**
   ```bash
   cd ./data_preprocessing
   python preprocess.py --dataset_name=Cora
   ```
2. **Start Training.**
   ```bash
   cd ..
   python main.py --data-name=Cora --epochs=20 --num-layers=2 --num-hidden=256 --aicore-num=20 
   ```
## 4.2 Baseline

### 4.2.1 MindsporeGL-graph
   **Start Training.**
   ```bash
   cd ./baseline/graphlearning/examples
   python vc_gcn_datanet.py   --data-name=Cora --epochs=20 --num-layers=2 --num-hidden=256 --fuse
   ```
### 4.2.2 MindsporeGL-pynative
   **Start Training.**
   ```bash
   cd ./baseline/graphlearning/examples
   python vc_gcn_datanet.py   --data-name=Cora --epochs=20 --num-layers=2 --num-hidden=256  
   ```
### 4.2.3 graphlearning_TP

1. **Compile Operator Project.**
   ```bash
   cd ./baseline/graphlearning_TP/MmadCustomTP
   ./build.sh
   ```
2. **Declare Environment Variables.**
   ```bash
   vim ~/.bashrc
   export ASCEND_CUSTOM_OPP_PATH={build_out_path}build_out/_CPack_Packages/Linux/External/custom_opp_openEuler_aarch64.run/packages/vendors/customize:$ASCEND_CUSTOM_OPP_PATH
   source ~/.bashrc
   ```
3. **Data Preprocessing.**
   ```bash
   cd ../../../data_preprocessing
   python preprocess.py --dataset_name=Cora
   ```
4. **Start Training.**
   ```bash
   cd ../baseline/graphlearning_TP/examples
   python vc_gcn_datanet.py  --data-name=Cora --epochs=10 --num-layers=2 --num-hidden=256 
   ```

## 4.3 Performance Comparison Experiments (MindsporeGL-pynative VS NeutronAscend)

1. **Hidden Layer Dimensions**

```
python vc_gcn_datanet.py --data-name=Cora --epochs=10 --num-layers=2 --num-hidden=256
python main.py --data-name=Cora --epochs=10 --num-layers=2 --num-hidden=256
```

| MindsporeGL-pynative | NeutronAscend |
|----------------------|---------------|
| Epoch time:6762.388706207275 ms Train loss 1.9459919 Test acc:0.276 | Epoch time:8146.044969558716 ms Train loss 1.9463493 Test acc:0.297 |
| train_loss=1.9025142<br>Epoch time:23.68783950805664 ms Train loss 1.9025142 Test acc:0.314 | train_loss=1.9044855<br>Epoch time:4.7855377197265625 ms Train loss 1.9044855 Test acc:0.401 |
| train_loss=1.8654575<br>Epoch time:14.271020889282227 ms Train loss 1.8654575 Test acc:0.382 | train_loss=1.8660983<br>Epoch time:4.29844856262207 ms Train loss 1.8660983 Test acc:0.406 |
| train_loss=1.8180279<br>Epoch time:13.913631439208984 ms Train loss 1.8180279 Test acc:0.452 | train_loss=1.8178546<br>Epoch time:4.085779190063477 ms Train loss 1.8178546 Test acc:0.442 |
| train_loss=1.7640097<br>Epoch time:13.920783996582031 ms Train loss 1.7640097 Test acc:0.559 | train_loss=1.7660494<br>Epoch time:4.086017608642578 ms Train loss 1.7660494 Test acc:0.522 |
| train_loss=1.7172061<br>Epoch time:13.888359069824219 ms Train loss 1.7172061 Test acc:0.637 | train_loss=1.703722<br>Epoch time:4.062891006469727 ms Train loss 1.703722 Test acc:0.631 |
| train_loss=1.6453526<br>Epoch time:13.69476318359375 ms Train loss 1.6453526 Test acc:0.731 | train_loss=1.6585788<br>Epoch time:4.066944122314453 ms Train loss 1.6585788 Test acc:0.703 |
| train_loss=1.5801979<br>Epoch time:13.773679733276367 ms Train loss 1.5801979 Test acc:0.789 | train_loss=1.5720006<br>Epoch time:4.061698913574219 ms Train loss 1.5720006 Test acc:0.767 |
| train_loss=1.5067801<br>Epoch time:13.751745223999023 ms Train loss 1.5067801 Test acc:0.797 | train_loss=1.5117003<br>Epoch time:4.238367080688477 ms Train loss 1.5117003 Test acc:0.786 |
| train_loss=1.4378883<br>Epoch time:13.725042343139648 ms Train loss 1.4378883 Test acc:0.803 | train_loss=1.4193555<br>Epoch time:4.065752029418945 ms Train loss 1.4193555 Test acc:0.793 |
| **Model:GCN Dataset:Cora Avg epoch time:13.8097 ms** | **Model:GCN Dataset:Cora Avg epoch time:4.0953 ms** |

2. **Number of Layers**

   - **Three Layers**
   ```
   python vc_gcn_datanet.py --data-name=Cora --epochs=10 --num-layers=3 --num-hidden=256
   python main.py --data-name=Cora --epochs=10 --num-layers=3 --num-hidden=256
   ```
| MindsporeGL-pynative | NeutronAscend |
|----------------------|---------------|
| Epoch time:6954.7271728515625 ms Train loss 1.9474899 Test acc:0.32 | Epoch time:8422.435760498047 ms Train loss 1.9454712 Test acc:0.269 |
| train_loss=1.8966631<br>Epoch time:26.79276466369629 ms Train loss 1.8966631 Test acc:0.393 | train_loss=1.8979884<br>Epoch time:6.688833236694336 ms Train loss 1.8979884 Test acc:0.423 |
| train_loss=1.8271703<br>Epoch time:16.934871673583984 ms Train loss 1.8271703 Test acc:0.682 | train_loss=1.8239075<br>Epoch time:5.833625793457031 ms Train loss 1.8239075 Test acc:0.717 |
| train_loss=1.7173223<br>Epoch time:16.86692237854004 ms Train loss 1.7173223 Test acc:0.761 | train_loss=1.7065942<br>Epoch time:5.680084228515625 ms Train loss 1.7065942 Test acc:0.726 |
| train_loss=1.5200964<br>Epoch time:16.55745506286621 ms Train loss 1.5200964 Test acc:0.736 | train_loss=1.5503752<br>Epoch time:5.7086944580078125 ms Train loss 1.5503752 Test acc:0.766 |
| train_loss=1.3064184<br>Epoch time:16.695022583007812 ms Train loss 1.3064184 Test acc:0.751 | train_loss=1.3208265<br>Epoch time:5.701541900634766 ms Train loss 1.3208265 Test acc:0.809 |
| train_loss=1.0729853<br>Epoch time:16.620397567749023 ms Train loss 1.0729853 Test acc:0.783 | train_loss=1.1089164<br>Epoch time:5.692243576049805 ms Train loss 1.1089164 Test acc:0.771 |
| train_loss=0.78226745<br>Epoch time:16.727685928344727 ms Train loss 0.78226745 Test acc:0.78 | train_loss=0.84036297<br>Epoch time:5.698442459106445 ms Train loss 0.84036297 Test acc:0.787 |
| train_loss=0.61880964<br>Epoch time:16.70360565185547 ms Train loss 0.61880964 Test acc:0.76 | train_loss=0.61036736<br>Epoch time:5.460500717163086 ms Train loss 0.61036736 Test acc:0.759 |
| train_loss=0.46833467<br>Epoch time:16.38007164001465 ms Train loss 0.46833467 Test acc:0.782 | train_loss=0.4349865<br>Epoch time:5.440711975097656 ms Train loss 0.4349865 Test acc:0.808 |
| **Model:GCN Dataset:Cora Avg epoch time:16.6502 ms** | **Model:GCN Dataset:Cora Avg epoch time:5.6260 ms** |






---

3. **Memory and Power Consumption** (GPU data in `data_preprocessing/ntspowerdraw.py`)
   ```bash
   # Collect during training
   python vc_gcn_datanet.py   --data-name=Cora --epochs=100 --num-layers=2 --num-hidden=256
   stdbuf -oL npu-smi info watch -i {device-id} | tee train.log #device-id = npu_id
   ```

   ```
   NpuID(Idx)  ChipId(Idx) Pwr(W)      Temp(C)     AI Core(%)  AI Cpu(%)   Ctrl Cpu(%) Memory(%)   Memory BW(%)
   2           0           90.1        34          0           0           2           5           0           
   2           0           90.2        34          0           0           0           5           0           
   2           0           90.3        34          0           0           0           5           0           
   2           0           90.2        34          0           0           1           5           0           
   2           0           90.2        34          0           0           2           5           0           
   2           0           90.2        34          0           0           1           5           0           
   2           0           90.2        34          0           0           1           5           0           
   2           0           90.2        34          0           0           0           5           0           
   2           0           90.2        34          0           0           1           5           0           
   2           0           90.2        34          0           0           3           5           0           
   2           0           97.6        34          0           0           16          5           0           
   2           0           90.7        34          0           0           0           5           0           
   2           0           90.1        34          0           0           0           5           0           
   2           0           90.1        34          0           0           2           5           0           
   2           0           98.1        34          0           0           6           5           0           
   2           0           98.1        34          0           0           2           5           0           
   2           0           98.0        34          0           0           1           5           0           
   2           0           98.1        34          0           0           3           5           0           
   2           0           98.2        34          0           0           4           5           0           
   2           0           98.3        34          0           0           2           5           0           
   2           0           100.1       34          0           0           1           5           0           
   2           0           99.4        35          0           0           0           5           0           
   2           0           98.2        34          0           0           2           5           0           
   2           0           98.2        34          0           0           0           5           0           
   2           0           90.2        34          0           0           4           5           0           
   2           0           93.2        34          0           0           0           5           0           
   2           0           93.0        34          0           0           3           5           0           
   2           0           90.2        34          0           0           11          5           0           
   2           0           90.2        34          0           0           3           5           0           
   2           0           90.2        34          0           0           3           5           0           
   2           0           90.2        34          0           0           0           5           0           
   ```
4. **Operator Time Proportion**
   ```bash
   python vc_gcn_datanet.py   --data-name=Cora --epochs=100 --num-layers=2 --num-hidden=256 --fuse --profile # Add --profile parameter
   mindinsight start
   ## Open the prof folder in the browser to view
   ```
   ![Operator Proportion Chart](./images/operator_time_analysis.png)

---

# 5. Acknowledgment

This project references the design and implementation of [MindSpore graphlearning](https://gitee.com/mindspore/graphlearning). Thanks for providing open-source code and documentation support.

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
1. **Data preprocessing.**
   ```bash
   cd ./data_preprocessing
   python preprocess.py --dataset_name=Cora
   ```
2. **Start training.**
   ```bash
   cd ..
   python main.py --data-name=Cora --epochs=20 --num-layers=2 --num-hidden=256 --aicore-num=20 
   ```
## 4.2 Baseline

### 4.2.1 MindsporeGL-graph
   **Start training.**
   ```bash
   cd ./baseline/graphlearning_batch/examples
   python vc_gcn_datanet.py   --data-name=Cora --epochs=20 --num-layers=2 --num-hidden=256 --fuse
   ```
### 4.2.2 MindsporeGL-pynative
   **Start training.**
   ```bash
   cd ./baseline/graphlearning_batch/examples
   python vc_gcn_datanet.py   --data-name=Cora --epochs=20 --num-layers=2 --num-hidden=256  
   ```
### 4.2.3 graphlearning_TP
   
1. **Compile the operator project.**
   ```bash
   cd ./baseline/graphlearning_TP/MmadCustomTP
   ./build.sh
   ```
2. **Declare environment variables.**
   ```bash
   vim ~/.bashrc
   export ASCEND_CUSTOM_OPP_PATH={build_out_path}build_out/_CPack_Packages/Linux/External/custom_opp_openEuler_aarch64.run/packages/vendors/customize:$ASCEND_CUSTOM_OPP_PATH
   source ~/.bashrc
   ```
3. **Data preprocessing.**
   ```bash
   cd ../../../data_preprocessing
   python preprocess.py --dataset_name=Cora
   ```
4. **Start training.**
   ```bash
   cd ../baseline/graphlearning_TP/examples
   python vc_gcn_datanet.py  --data-name=Cora --epochs=20 --num-layers=2 --num-hidden=256 
   ```

## 4.3 Data Collection (Example: MindsporeGL-graph)
1. **Memory and Power Consumption** (GPU see `data_preprocessing/ntspowerdraw.py`)
```
# Collect during training
python vc_gcn_datanet.py   --data-name=Cora --epochs=20 --num-layers=2 --num-hidden=256 --fuse
stdbuf -oL npu-smi info watch -i {device-id} | tee train.log #device-id = npu_id
```
2. **Operator Time Proportion**
```
python vc_gcn_datanet.py   --data-name=Cora --epochs=20 --num-layers=2 --num-hidden=256 --fuse --profile #Add --profile parameter
mindinsight start
## Open the prof folder in the browser to view
```

---

# 5. Acknowledgments

This project references the design and implementation of [MindSpore graphlearning](https://gitee.com/mindspore/graphlearning). Thanks for providing open-source code and documentation support.

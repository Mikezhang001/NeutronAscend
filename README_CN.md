# 目录
- [1. 环境要求](#1-环境要求)
- [2. 配置环境](#2-配置环境)
  - [2.1 下载](#21-下载)
  - [2.2 安装](#22-安装)
- [3. 算子部署](#3-算子部署)
- [4. 项目开始](#4-项目开始)
- [5. 致谢](#5-致谢)

---

# 1. 环境要求
| 软件/硬件 | 版本/型号 |
|-----------|-----------|
| **服务器** | Atlas 800T A2 训练服务器<br>NPU型号: Ascend910B3<br>CPU架构: AArch64 |
| **固件**   | Ascend HDK 25.0.RC1<br>软件包名称：Ascend-hdk-910b-npu-firmware_7.7.0.1.231.run |
| **驱动**   | Ascend HDK 25.0.RC1<br>软件包名称：Ascend-hdk-910b-npu-driver_25.0.rc1.1_linux-aarch64.run |
| **MindSpore** | 2.3.1 |
| **MindSpore_GL** | 0.2 |
| **MindInsight** | 2.3.1 |
---

# 2. 配置环境

## 2.1 下载

### 2.1.1 [固件和驱动](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.1.RC1.alpha002&driver=Ascend+HDK+25.0.RC1)
![固件和驱动安装选项](./images/fireware.PNG)

### 2.1.2 [CANN](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.1.RC1.alpha002)
![CANN安装选项](./images/CANN.png)

---

## 2.2 安装

### 2.2.1 固件和驱动
参考链接: [安装NPU驱动固件](https://support.huawei.com/enterprise/zh/doc/EDOC1100349380/ac9d2505)

1. **以`root`用户登录服务器。**
2. **创建驱动运行用户`HwHiAiUser`（运行驱动进程的用户）。**
   ```bash
   groupadd HwHiAiUser
   useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
   ```
3. **将驱动包和固件包上传到服务器任意目录如“/home”。**
4. **增加驱动和固件包的可执行权限。**
   ```bash
   chmod +x Ascend-hdk-910b-npu-driver_25.0.rc1.1_linux-aarch64.run
   chmod +x Ascend-hdk-910b-npu-firmware_7.7.0.1.231.run
   ```
5. **安装驱动和固件。**
   - **安装驱动**
     ```bash
     ./Ascend-hdk-910b-npu-driver_25.0.rc1.1_linux-aarch64.run --full --install-for-all
     ```
     > 如果出现缺少工具报错：
     > ```
     > [ERROR]The list of missing tools: lspci,ifconfig,
     > ```
     > 请参考[驱动安装缺少依赖报错](https://support.huawei.com/enterprise/zh/doc/EDOC1100349380/3652fc47#ZH-CN_TOPIC_0000001782749677)。

     > 如果出现DKMS编译失败：
     > ```
     > [ERROR]Dkms install failed, details in : var/log/ascend_seclog/ascend_install.log.
     > ```
     > 请参考[驱动安装过程中出现DKMS编译失败](https://support.huawei.com/enterprise/zh/doc/EDOC1100349380/64a31720#ZH-CN_TOPIC_0000001784970501)。

     > 安装成功提示：
     > ```
     > Driver package installed successfully!
     > ```

   - **安装固件**
     ```bash
     ./Ascend-hdk-910b-npu-firmware_7.7.0.1.231.run --full
     ```
     > 安装成功提示：
     > ```
     > Firmware package installed successfully! Reboot now or after driver installation for the installation/upgrade to take effect.
     > ```
6. **重启系统（可选）。**
   ```bash
   reboot
   ```
7. **验证驱动加载。**
   ```bash
   npu-smi info
   ```
   > **注意：非root用户需要添加HwHiAiUser。**
   > ```bash
   > sudo usermod -aG HwHiAiUser username
   > ```

---

### 2.2.2 CANN
参考链接: [安装CANN](https://zhuanlan.zhihu.com/p/719099792)

1. **进入root用户。**
   ```bash
   sudo su
   ```
2. **修改CANN包权限。**
   ```bash
   chmod +x Ascend-cann-kernels-910b_8.1.RC1.alpha002_linux-aarch64.run
   chmod +x Ascend-cann-toolkit_8.1.RC1.alpha002_linux-aarch64.run
   ```
3. **安装CANN。**
   - 删除旧版本（如果需要）：
     ```bash
     rm -rf /usr/local/Ascend/ascend-toolkit
     ```
   - 安装工具包：
     ```bash
     ./Ascend-cann-toolkit_8.1.RC1.alpha002_linux-aarch64.run --install
     ```
   - 安装内核包：
     ```bash
     ./Ascend-cann-kernels-910b_8.1.RC1.alpha002_linux-aarch64.run
     ```
4. **验证安装。**
   ```bash
   ls /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/
   ```
5. **配置环境变量。**
   ```bash
   vim ~/.bashrc
   # 添加以下内容：
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

---

### 2.2.3 MindSpore
参考链接: [安装MindSpore](https://zhuanlan.zhihu.com/p/719099792)

1. **创建Python环境。**
   ```bash
   conda create -n mindspore python=3.9
   conda activate mindspore
   ```
2. **安装依赖。**
   ```bash
   pip install sympy
   pip install numpy==1.26
   pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl
   pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-*-py3-none-any.whl
   ```
3. **安装MindSpore和MindInsight。**
   ```bash
   pip install mindspore==2.3.1
   pip install mindinsight==2.3.1
   ```
4. **验证安装。**
   ```bash
   python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"
   ```
   > 安装成功提示：
   > ```
   > The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
   > ```

---

### 2.2.4 MindSpore_GL
参考链接: [MindSpore Graph Learning](https://gitee.com/mindspore/graphlearning)

1. **下载源码。**
   ```bash
   git clone https://gitee.com/mindspore/graphlearning.git
   ```
2. **编译并安装。**
   ```bash
   cd graphlearning
   bash build.sh
   pip install ./output/mindspore_gl*.whl
   ```
3. **验证安装。**
   ```bash
   python -c 'import mindspore_gl'
   ```
   > 如果没有报错“No module named 'mindspore_gl'”，则说明安装成功。

---

# 3. 算子部署

1. **编译算子工程。**
   ```bash
   sudo su root
   conda activate mindspore
   cd MmadCustom
   ./build.sh
   ```
2. **声明环境变量。**
   ```bash
   vim ~/.bashrc
   export ASCEND_CUSTOM_OPP_PATH={build_out_path}build_out/_CPack_Packages/Linux/External/custom_opp_openEuler_aarch64.run/packages/vendors/customize:$ASCEND_CUSTOM_OPP_PATH
   source ~/.bashrc
   ```
3. **测试正常调用。**
   ```bash
   python ../test/test_mmad.py
   ```

---

# 4. 项目开始

## 4.1. NeutronAscend
1. **数据预处理。**
   ```bash
   cd ./data_preprocessing
   python preprocess.py --dataset_name=Cora
   ```
2. **开始训练。**
   ```bash
   cd ..
   python main.py --data-name=Cora --epochs=20 --num-layers=2 --num-hidden=256 --aicore-num=20 
   ```
## 4.2 baseline

### 4.2.1 MindsporeGL-graph
   **开始训练。**
   ```bash
   cd ./baseline/graphlearning_batch/examples
   python vc_gcn_datanet.py   --data-name=Cora --epochs=20 --num-layers=2 --num-hidden=256 --fuse
   ```
### 4.2.2 MindsporeGL-pynative
   **开始训练。**
   ```bash
   cd ./baseline/graphlearning_batch/examples
   python vc_gcn_datanet.py   --data-name=Cora --epochs=20 --num-layers=2 --num-hidden=256  
   ```
### 4.2.3 graphlearning_TP
   
1. **编译算子工程。**
   ```bash
   cd ./baseline/graphlearning_TP/MmadCustomTP
   ./build.sh
   ```
2. **声明环境变量。**
   ```bash
   vim ~/.bashrc
   export ASCEND_CUSTOM_OPP_PATH={build_out_path}build_out/_CPack_Packages/Linux/External/custom_opp_openEuler_aarch64.run/packages/vendors/customize:$ASCEND_CUSTOM_OPP_PATH
   source ~/.bashrc
3. **数据预处理。**
   ```bash
   cd ../../../data_preprocessing
   python preprocess.py --dataset_name=Cora
   ```
4. **开始训练。**
   ```bash
   cd ../baseline/graphlearning_TP/examples
   python vc_gcn_datanet.py  --data-name=Cora --epochs=20 --num-layers=2 --num-hidden=256 
   ```

## 4.3 数据采集(以MindsporeGL-graph举例)
1. **显存和能耗**(GPU见`data_preprocessing/ntspowerdraw.py`)
```
#在执行训练时收集
python vc_gcn_datanet.py   --data-name=Cora --epochs=20 --num-layers=2 --num-hidden=256 --fuse
stdbuf -oL npu-smi info watch -i {device-id} | tee train.log #device-id = npu_id
```
2. **算子时间占比**
```
python vc_gcn_datanet.py   --data-name=Cora --epochs=20 --num-layers=2 --num-hidden=256 --fuse --profile #填上--profile参数即可
mindinsight start
##浏览器页面打开prof文件夹即可查看
```

---

# 5. 致谢

本项目参考了 [MindSpore graphlearning](https://gitee.com/mindspore/graphlearning) 的设计与实现，感谢其提供的开源代码和文档支持。

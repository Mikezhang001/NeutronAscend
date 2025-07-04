# 1.环境要求
| 软件/硬件 | 版本/型号                        |
|------|------------------------------------------------------------|
| 服务器 | Atlas 800T A2 训练服务器<br>NPU型号: Ascend910B3<br>CPU架构: AArch64|
| 固件 | Ascend HDK 25.0.RC1<br>软件包名称：Ascend-hdk-910b-npu-firmware_7.7.0.1.231.run |
| 驱动 | Ascend HDK 25.0.RC1<br>软件包名称：Ascend-hdk-910b-npu-driver_25.0.rc1.1_linux-aarch64.run       |
| mindspore | 2.3.1       |
| mindspore_gl | 0.2       |

# 2.配置环境

## 2.1下载

### 2.1.1[固件和驱动](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.1.RC1.alpha002&driver=Ascend+HDK+25.0.RC1)
![固件和驱动安装选项](./images/fireware.PNG)

### 2.1.2[CANN](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.1.RC1.alpha002)
![CANN安装选项](./images/CANN.png)

## 2.2安装

### 2.2.1固件和驱动
参考链接: [安装NPU驱动固件](https://support.huawei.com/enterprise/zh/doc/EDOC1100349380/ac9d2505)
1. 以`root`用户登录服务器。
2. 创建驱动运行用户`HwHiAiUser`（运行驱动进程的用户），安装驱动时无需指定运行用户，默认即为`HwHiAiUser`。

```
groupadd HwHiAiUser
useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
```

3. 将驱动包和固件包上传到服务器任意目录如“/home”。
4. 执行如下命令，增加驱动和固件包的可执行权限。
```
chmod +x Ascend-hdk-910b-npu-driver_25.0.rc1.1_linux-aarch64.run
chmod +x Ascend-hdk-910b-npu-firmware_7.7.0.1.231.run
```
5. 执行以下命令，完成驱动固件安装，软件包默认安装路径为“/usr/local/Ascend”。
- **安装驱动**  
  执行以下命令，完成驱动安装。
  ```
  ./Ascend-hdk-910b-npu-driver_25.0.rc1.1_linux-aarch64.run --full --install-for-all
  ```
  - 若执行上述安装命令出现类似如下回显信息，请参见[驱动安装缺少依赖报错](https://support.huawei.com/enterprise/zh/doc/EDOC1100349380/3652fc47#ZH-CN_TOPIC_0000001782749677)解决。
    ```
    [ERROR]The list of missing tools: lspci,ifconfig,
    ```
  - 若执行上述安装命令出现类似如下回显信息，请参见[驱动安装过程中出现dkms编译失败](https://support.huawei.com/enterprise/zh/doc/EDOC1100349380/64a31720#ZH-CN_TOPIC_0000001784970501)报错解决。
    ```
    [ERROR]Dkms install failed, details in : var/log/ascend_seclog/ascend_install.log. 
    [ERROR]Driver_ko_install failed, details in : /var/log/ascend_seclog/ascend_install.log.
    ```
  - 若系统出现如下关键回显信息，则表示驱动安装成功。
    ```
    Driver package installed successfully!
    ```

- **安装固件**  
  执行以下命令，完成固件安装。
  ```
  ./Ascend-hdk-910b-npu-firmware_7.7.0.1.231.run --full
  ```
  若系统出现如下关键回显信息，表示固件安装成功。
  ```
  Firmware package installed successfully! Reboot now or after driver installation for the installation/upgrade to take effect 
  ```
6. 执行 `reboot` 命令重启系统。(不是必须)  
7. 执行 `npu-smi info` 查看驱动加载是否成功。
> **注意：非 root 用户需要添加 HwHiAiUser**
> ```
> sudo usermod -aG HwHiAiUser username
> ```
### 2.2.2CANN
参考链接: [安装CANN](https://zhuanlan.zhihu.com/p/719099792)
1. 进入root用户（必须）
```
sudo su
```
2. 修改CANN包权限
```
chmod +x  Ascend-cann-kernels-910b_8.1.RC1.alpha002_linux-aarch64.run
chmod +x  Ascend-cann-toolkit_8.1.RC1.alpha002_linux-aarch64.run
```
3. 安装CANN
- 
  如果老版本的CANN不需要，可以先删掉cann的安装目录

  ```
  rm -rf /usr/local/Ascend/ascend-toolkit
  ```

  然后安装cann-toolkit
  ```
  ./Ascend-cann-toolkit_8.1.RC1.alpha002_linux-aarch64.run --install 
  ```
  然后安装kenerls包

  ```
  ./Ascend-cann-kernels-910b_8.1.RC1.alpha002_linux-aarch64.run
  ```

4. 确认kernels包是否成功安装(输出不为空且与型号一致)

```
ls /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/
```
    
5. 配置环境变量

```
vim  ~/.bashrc
# 然后在文件末尾添加  source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2.2.3mindspore
参考链接: [安装mindspore](https://zhuanlan.zhihu.com/p/719099792)
1. 创建python环境

```
conda create -n mindspore python=3.9
conda activate mindspore
```
2. 安装依赖

```
pip install sympy
pip install numpy==1.26
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-*-py3-none-any.whl
```
3. 安装mindspore
```
pip install mindspore==2.3.1
```

4. 验证mindspore是否安装成功

>```
>python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"
>```
>打印如下即为成功：<br>
>The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!

### 2.2.4[mindspore_gl](https://gitee.com/mindspore/graphlearning)
1. 从代码仓下载源码

```
git clone https://gitee.com/mindspore/graphlearning.git
```

2. 编译安装MindSpore Graph Learning

```
cd graphlearning
bash build.sh
pip install ./output/mindspore_gl*.whl
```
3. 验证是否成功安装

>```
>python -c 'import mindspore_gl'
>```
>如果没有报错No module named 'mindspore_gl'，则说明安装成功。


# 3.算子部署

1. 算子工程编译

```
sudo su root
conda activate mindspore
cd MmadCustom
./build.sh
```

2.声明环境变量

```
export ASCEND_CUSTOM_OPP_PATH={build_out_path}build_out/_CPack_Packages/Linux/External/custom_opp_openEuler_aarch64.run/packages/vendors/customize:$ASCEND_CUSTOM_OPP_PATH

```
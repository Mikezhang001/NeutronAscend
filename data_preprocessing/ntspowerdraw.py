import subprocess
import threading
import time
import csv
import pynvml
from datetime import datetime
total_energy_joules = 0.0  # 总能耗（焦耳）

class EnergyMonitor:
    def __init__(self, device_index=0):
        self.device_index = device_index
        self.handle = None
        self.start_energy = 0
        self.start_time = 0
        self.energy_consumed = 0
        self.elapsed_time = 0
        
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.supported = True
        except:
            self.supported = False
    
    def start(self):
        if self.supported:
            self.start_time = time.time()
            self.start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
    
    def stop(self):
        if self.supported:
            end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            end_time = time.time()
            
            self.energy_consumed = (end_energy - self.start_energy) / 1000  # 转换为焦耳
            self.elapsed_time = end_time - self.start_time
    
    def get_power_draw(self):
        if self.supported:
            return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000  # 转换为瓦特
        return 0
    
    def shutdown(self):
        if self.supported:
            pynvml.nvmlShutdown()

def power_monitoring_task():
    """独立线程任务：每100ms获取功率并写入文件"""
    monitor = EnergyMonitor(0)
    start_time = time.time()
    
    with open("power_log.txt", "w") as f:
        f.write("timestamp,power_draw(W)\n")  # 写入CSV头部
        while time.time() - start_time < 120:  # 运行5秒
            timestamp = time.time() - start_time
            power = monitor.get_power_draw()
            
            # 写入时间戳和功率值
            f.write(f"{timestamp:.3f},{power:.2f}\n")
            f.flush()  # 确保每次写入后立即刷新到文件[9,10](@ref)
            
            time.sleep(0.1)  # 100ms间隔[6](@ref)

    print("功率监控完成，数据已保存到 power_log.txt")

def run_mpi_job():
    
    # 启动MPI任务
    cmd = ["./build/nts", str(args)]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    monitor_thread = threading.Thread(target=power_monitoring_task)
    monitor_thread.daemon = True  # 设置为守护线程确保主程序退出时终止[4](@ref)
    monitor_thread.start()
    
    # 等待任务完成
    stdout, stderr = proc.communicate()
    print("MPI任务输出:", stdout.decode())
    print("MPI任务错误输出:", stderr.decode())
    
    return stdout, stderr

if __name__ == "__main__":
    # 配置参数
    np = 1
    dataset = "reddit"
    method = "gcn"
    args = f"./powerCostExp/{method}/{method}_{dataset}.cfg"
    output_csv = f"./powerCostExp/{method}/{dataset}_gpu_power_log.csv"
    
    # 执行任务
    stdout, stderr = run_mpi_job()
    
    # 输出结果
    # print(f"平均耗时: {duration*10:.4f}ms")
    print(f"工作状态平均能耗: {(total_energy_joules/100):.2f}J")
    print(f"能耗数据已保存至: {output_csv}")
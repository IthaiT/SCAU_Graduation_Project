import platform
import torch
import psutil
import cpuinfo

print("===== 系统信息 =====")
print("操作系统:", platform.system(), platform.release())
print("Python版本:", platform.python_version())

print("\n===== CPU信息 =====")
cpu = cpuinfo.get_cpu_info()
print("CPU型号:", cpu['brand_raw'])
print("CPU核心数:", psutil.cpu_count(logical=False))
print("CPU线程数:", psutil.cpu_count(logical=True))

print("\n===== 内存信息 =====")
mem = psutil.virtual_memory()
print("总内存: %.2f GB" % (mem.total / (1024**3)))

print("\n===== GPU信息 =====")
if torch.cuda.is_available():
    print("GPU型号:", torch.cuda.get_device_name(0))
    print("GPU数量:", torch.cuda.device_count())
    print("CUDA版本:", torch.version.cuda)
else:
    print("未检测到GPU")

print("\n===== PyTorch信息 =====")
print("PyTorch版本:", torch.__version__)
import ctypes

hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2\\bin\\cudart64_100.dll")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

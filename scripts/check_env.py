import importlib.metadata
import platform
import torch
print("Python:",platform.python_version())
for name in ["torch","transformers","datasets","accelerate","numpy"]:
    print(name,importlib.metadata.version(name))
print("CUDA runtime:",torch.version.cuda,"available:",torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA is required for the documented server experiments"
for i in range(torch.cuda.device_count()):
    print(i,torch.cuda.get_device_name(i),"GiB",round(torch.cuda.get_device_properties(i).total_memory/2**30,1))
print("BF16:",torch.cuda.is_bf16_supported())
x=torch.randn(64,64,device="cuda")
assert torch.isfinite(x@x).all()
print("GPU check passed")

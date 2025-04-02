import torch

num_gpus = torch.cuda.device_count()
print(f"Number of CUDA-enabled GPUs: {num_gpus}")

for i in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(i)
    print(f"GPU {i}: {gpu_name}")
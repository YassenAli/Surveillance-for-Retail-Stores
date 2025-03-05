import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.zeros(1).cuda())
print(torch.cuda.memory_allocated() / 1e9, 'GB')
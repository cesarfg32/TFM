import torch, platform, sys

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA (wheel):", torch.version.cuda)
    x = torch.randn(1, device="cuda")
    print("Tensor creado en:", x.device)
else:
    print("Usando CPU")
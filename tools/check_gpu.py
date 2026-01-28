import torch
import torch.cuda as cuda

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {cuda.is_available()}")
print(f"GPU Device Count: {cuda.device_count()}")

if cuda.is_available():
    for i in range(cuda.device_count()):
        print(f"\nGPU {i}: {cuda.get_device_name(i)}")
        props = cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
else:
    print("\n⚠️ No GPU detected. Model will run on CPU (slower).")

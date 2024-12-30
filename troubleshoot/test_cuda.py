import torch

if torch.cuda.is_available():
    print("CUDA available")
else:
    print("CUDA not available")

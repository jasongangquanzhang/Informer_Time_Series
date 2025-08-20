import sys
import torch

def test_cuda():
    print("Current Python executable:", sys.executable)  # Conda env path
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA version reported by PyTorch:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
        print("Number of GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"    Memory Cached:    {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

        # Simple CUDA test
        x = torch.rand(3, 3).cuda()
        y = torch.rand(3, 3).cuda()
        z = x + y
        print("CUDA test tensor computation successful:\n", z)
    else:
        print("CUDA is not available. Check your installation.")

if __name__ == "__main__":
    test_cuda()

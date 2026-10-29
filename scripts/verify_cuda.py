"""在命令行或 PyCharm 中运行，核对实际解释器并执行 CUDA 张量运算。"""

import importlib.util
import sys


def main():
    print(f"Python: {sys.executable}")
    if importlib.util.find_spec("torch") is None:
        print("PyTorch is not installed in this interpreter.")
        return 1

    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Capability: {torch.cuda.get_device_capability(0)}")
    x = torch.ones((32, 32), device="cuda")
    result = (x @ x)[0, 0].item()
    torch.cuda.synchronize()
    assert result == 32.0, result
    print(f"CUDA matrix multiplication passed: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

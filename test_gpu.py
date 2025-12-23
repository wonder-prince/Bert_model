import torch

print(f"PyTorch 内部编译的 CUDA 版本: {torch.version.cuda}")
print(f"当前显卡是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 打印算力，4060 应该是 8.9
    major, minor = torch.cuda.get_device_capability()
    print(f"显卡算力: {major}.{minor}") 
    
    # 尝试在显卡上创建一个张量
    device = torch.device("cuda")
    x = torch.ones(1, device=device)
    print("显卡握手成功！")
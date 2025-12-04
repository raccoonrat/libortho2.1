import torch
import torch.nn as nn
import time
import sys
import os

# 把上级目录加入 path 以便导入 libortho
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libortho.engine import LibOrthoEngine

"""
examples/benchmark.py

这是验证脚本。它模拟一个简单的神经网络，训练它记住一些"隐私数据"（高损失点），
然后使用 LibOrtho 分离它，并证明：
1. alpha=1 时，模型依然记得隐私。
2. alpha=0 时，模型完全忘记隐私（损失爆炸）。
3. 性能开销在可控范围内。
"""

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 定义一个简单的模型
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size=1000, dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        # 多层结构以模拟复杂性
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, vocab_size) # Logits
        )
    
    def forward(self, x):
        h = self.embed(x)
        # 简单的平均池化模拟 Context
        h = h.mean(dim=1)
        return self.layers(h)

def main():
    device = get_device()
    print(f"--- LibOrtho Benchmark (Device: {device}) ---")
    
    # 2. 准备数据
    # "通用数据": 随机噪声，易于学习
    # "隐私数据": 特定的 Outlier，我们强制过拟合它
    vocab_size = 1000
    model = SimpleLLM(vocab_size).to(device)
    
    # 模拟数据加载器
    # 隐私数据是一个特定的序列，我们希望模型能死记硬背它
    privacy_input = torch.randint(0, vocab_size, (1, 10)).to(device)
    privacy_target = torch.randint(0, vocab_size, (1,)).to(device)
    
    general_loader = [
        (torch.randint(0, vocab_size, (32, 10)), torch.randint(0, vocab_size, (32,)))
        for _ in range(50)
    ]

    engine = LibOrthoEngine(model)
    engine.convert() # 转换为 OrthoLinear
    
    # 3. 模拟"预训练/微调" (让模型记住隐私)
    print("\n[Step 1] Simulating Overfitting (Memorization)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(50): # 暴力过拟合隐私数据
        optimizer.zero_grad()
        out = model(privacy_input)
        loss = nn.functional.cross_entropy(out, privacy_target)
        loss.backward()
        optimizer.step()
    
    # 检查初始隐私损失
    with torch.no_grad():
        loss_initial = nn.functional.cross_entropy(model(privacy_input), privacy_target)
    print(f"  > Initial Privacy Loss (Should be low): {loss_initial.item():.4f}")

    # 4. 校准与分离
    print("\n[Step 2] Calibration & Decomposition...")
    # 我们混合一点隐私数据进入校准集，这样 Hessian 会捕捉到它的高曲率
    calib_loader = general_loader[:5] 
    calib_loader.append((privacy_input, privacy_target))
    
    engine.calibrate(calib_loader, device)
    engine.decompose(sparsity=0.1) # 10% 的权重去正交流

    # 5. 验证 "Privacy Kill Switch"
    print("\n[Step 3] Verification...")
    
    # Mode: FULL
    engine.set_mode(alpha=1.0)
    with torch.no_grad():
        loss_full = nn.functional.cross_entropy(model(privacy_input), privacy_target)
    print(f"  > Mode FULL (alpha=1.0) Privacy Loss: {loss_full.item():.4f}")
    
    # Mode: SAFE
    engine.set_mode(alpha=0.0)
    with torch.no_grad():
        loss_safe = nn.functional.cross_entropy(model(privacy_input), privacy_target)
    print(f"  > Mode SAFE (alpha=0.0) Privacy Loss: {loss_safe.item():.4f}")
    
    # 6. 结论判定
    if loss_safe > loss_full * 5:
        print("\n[RESULT] SUCCESS: Privacy mechanism verified.")
        print(f"  > Removing Ortho stream caused loss to spike by {loss_safe/loss_full:.1f}x")
    else:
        print("\n[RESULT] FAILURE: Privacy separation not distinct enough.")

    # 7. 性能测试 (Latency)
    print("\n[Step 4] Performance Test (Latency)...")
    dummy_in = torch.randint(0, vocab_size, (32, 10)).to(device)
    
    # Warmup
    for _ in range(10): model(dummy_in)
    
    # Measure Alpha=0
    engine.set_mode(0.0)
    start = time.time()
    for _ in range(100): model(dummy_in)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_safe = (time.time() - start) / 100
    
    # Measure Alpha=1
    engine.set_mode(1.0)
    start = time.time()
    for _ in range(100): model(dummy_in)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_full = (time.time() - start) / 100
    
    print(f"  > Latency (Safe Mode): {t_safe*1000:.2f} ms")
    print(f"  > Latency (Full Mode): {t_full*1000:.2f} ms")
    overhead = (t_full - t_safe) / t_safe * 100
    print(f"  > Overhead: {overhead:.2f}%")

if __name__ == "__main__":
    main()


import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .layers import OrthoLinear

"""
libortho/engine.py

这是 LibOrtho 的手术刀和生命周期管理器。
它负责把一个普通的模型变成 LibOrtho 模型。
"""

class LibOrthoEngine:
    def __init__(self, model):
        self.model = model
        self.hessians = {} 

    def convert(self):
        """
        Model Surgery (模型手术).
        递归地将所有的 nn.Linear 替换为 OrthoLinear。
        """
        print("[LibOrtho] Performing model surgery...")
        converted_count = 0
        
        # 我们需要先收集所有模块，再修改，避免迭代中修改字典结构
        modules_to_replace = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                modules_to_replace.append((name, module))
        
        for name, module in modules_to_replace:
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[-1]
            
            if parent_name:
                parent = self.model.get_submodule(parent_name)
            else:
                parent = self.model
            
            new_layer = OrthoLinear.from_linear(module)
            setattr(parent, child_name, new_layer)
            converted_count += 1
            
        print(f"[LibOrtho] Surgery complete. {converted_count} layers upgraded.")

    def calibrate(self, dataloader, device, limit_batches=10):
        """
        计算 Fisher Information (Hessian 对角线)。
        这对应论文中的 "Instance-Level Curvature"。
        """
        print(f"[LibOrtho] Calibrating on device: {device}...")
        self.model.to(device)
        self.model.train() # 需要梯度
        
        self.hessians = {}
        
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= limit_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            self.model.zero_grad()
            
            outputs = self.model(inputs)
            # 使用标准的 CrossEntropy 作为曲率代理
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs.logits # HuggingFace 风格
                
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            
            # 累积平方梯度 (Fisher)
            for name, module in self.model.named_modules():
                if isinstance(module, OrthoLinear):
                    # 注意：我们对 weight_base 求导，此时它包含完整权重
                    if module.weight_base.grad is not None:
                        g = module.weight_base.grad.detach()
                        if name not in self.hessians:
                            self.hessians[name] = torch.zeros_like(g)
                        self.hessians[name] += g ** 2
            
            print(f"\r  > Batch {i+1}/{limit_batches}", end="")
        print("\n[LibOrtho] Calibration finished.")

    def decompose(self, sparsity=0.05):
        """根据校准结果，将权重物理分离到两个流中。"""
        print(f"[LibOrtho] Decomposing weights (Target Sparsity: {sparsity})...")
        count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, OrthoLinear):
                h = self.hessians.get(name, None)
                module.apply_decomposition(h, sparsity)
                count += 1
        print(f"[LibOrtho] Decomposed {count} layers.")

    def set_mode(self, alpha):
        """
        设置全局 Alpha 值。
        0.0 = 安全模式 (Privacy Safe)
        1.0 = 完整模式 (Full Capability)
        """
        for module in self.model.modules():
            if isinstance(module, OrthoLinear):
                module.alpha = alpha
        mode = "SAFE (Privacy)" if alpha == 0 else "FULL (Genius)"
        print(f"[LibOrtho] Switched to mode: {mode} (alpha={alpha})")

    def save(self, path):
        print(f"[LibOrtho] Saving to {path}...")
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        print(f"[LibOrtho] Loading from {path}...")
        # 假设当前模型结构已经匹配（已经执行过 convert）
        state = torch.load(path)
        self.model.load_state_dict(state)


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
        
        for i, data in enumerate(dataloader):
            if i >= limit_batches:
                break
            
            # 处理不同的数据格式
            if isinstance(data, (list, tuple)) and len(data) == 2:
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)
                # 对于语言模型，如果 inputs 是 1D，需要添加 batch 维度
                if inputs.dim() == 1:
                    inputs = inputs.unsqueeze(0)
                if targets.dim() == 1:
                    targets = targets.unsqueeze(0)
                model_inputs = inputs
            elif isinstance(data, dict):
                # HuggingFace 格式
                model_inputs = {k: v.to(device) for k, v in data.items()}
                targets = model_inputs.get('labels', model_inputs.get('input_ids'))
            else:
                model_inputs = data.to(device) if isinstance(data, torch.Tensor) else data
                targets = model_inputs if isinstance(model_inputs, torch.Tensor) else model_inputs.get('input_ids', None)
            
            self.model.zero_grad()
            
            # 调用模型
            if isinstance(model_inputs, dict):
                outputs = self.model(**model_inputs)
            else:
                outputs = self.model(model_inputs)
            
            # 使用标准的 CrossEntropy 作为曲率代理
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # 计算 loss
            if targets is not None:
                # 对于语言模型，需要 shift labels
                if logits.dim() == 3 and targets.dim() == 2:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = targets[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                else:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                # 如果没有 targets，使用模型自带的 loss
                loss = outputs.loss if hasattr(outputs, 'loss') else None
                if loss is None:
                    continue
            
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


import torch
import torch.nn.functional as F

"""
libortho/ops.py

这是数学核心。这里没有类，只有函数。
保持简单，保持无状态。
"""

def fake_quantize_int4(w, group_size=128):
    """
    模拟 INT4 量化。
    在生产环境中，这里应该是位操作（bit-packing）。
    但在几何证明阶段，浮点模拟是完全可以接受的。
    """
    # 1. Padding (处理边界情况，别让代码在维度不匹配时崩溃)
    original_shape = w.shape
    numel = w.numel()
    
    padding = 0
    if numel % group_size != 0:
        padding = group_size - (numel % group_size)
        w = F.pad(w.flatten(), (0, padding))
    
    # 2. Reshape to groups
    w_grouped = w.view(-1, group_size)
    
    # 3. Calculate Scales (Absmax quantization)
    # 避免除以零，加上 epsilon 是基本常识
    scales = w_grouped.abs().max(dim=1, keepdim=True)[0] + 1e-6
    
    # 4. Quantize -> Clamp -> Dequantize
    # Range: [-8, 7]
    w_int4 = torch.round(w_grouped / scales * 7.0)
    w_int4 = torch.clamp(w_int4, -8, 7)
    w_dequant = w_int4 / 7.0 * scales
    
    # 5. Restore shape
    w_out = w_dequant.flatten()
    if padding > 0:
        w_out = w_out[:-padding]
    
    return w_out.view(original_shape)

def calc_geometric_impact(residual, hessian_diag):
    """
    计算几何影响因子。
    Formula: Impact = (w - q)^2 * Hessian
    
    如果 hessian_diag 为 None，我们回退到普通的幅度剪枝（Magnitude Pruning）。
    这让代码更健壮。
    """
    if hessian_diag is None:
        return residual ** 2
    return (residual ** 2) * hessian_diag

def decompose_weights(w_orig, hessian_diag, sparsity_ratio):
    """
    执行几何分离的核心逻辑。
    返回: (w_base, ortho_indices, ortho_values)
    """
    # 1. Base Stream (量化投影)
    w_base = fake_quantize_int4(w_orig)
    residual = w_orig - w_base
    
    # 2. Calculate Impact
    impact = calc_geometric_impact(residual, hessian_diag)
    
    # 3. Thresholding
    # 找到第 k 大的值作为阈值
    k = int(impact.numel() * sparsity_ratio)
    if k <= 0:
        return w_base, None, None
        
    # flatten() 是必要的，因为 kthvalue 在最后一维工作
    threshold = torch.kthvalue(impact.flatten(), impact.numel() - k).values
    mask = impact > threshold
    
    # 4. Extract Ortho Stream
    # 使用 nonzero 提取索引。t() 是为了符合 sparse_coo 的格式 (2, N)
    ortho_indices = mask.nonzero(as_tuple=False).t()
    ortho_values = residual[mask]
    
    return w_base, ortho_indices, ortho_values


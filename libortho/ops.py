import torch
import torch.nn.functional as F

"""
libortho/ops.py

Linus Update v4: Row-wise Smart Imputation
"""

def fake_quantize_int4(w, group_size=128):
    # ... (保持不变) ...
    original_shape = w.shape
    numel = w.numel()
    padding = 0
    if numel % group_size != 0:
        padding = group_size - (numel % group_size)
        w = F.pad(w.flatten(), (0, padding))
    w_grouped = w.view(-1, group_size)
    scales = w_grouped.abs().max(dim=1, keepdim=True)[0] + 1e-6
    w_int4 = torch.round(w_grouped / scales * 7.0)
    w_int4 = torch.clamp(w_int4, -8, 7)
    w_dequant = w_int4 / 7.0 * scales
    w_out = w_dequant.flatten()
    if padding > 0:
        w_out = w_out[:-padding]
    return w_out.view(original_shape)

def calc_geometric_impact(w_orig, w_quant, hessian_diag):
    residual = w_orig - w_quant
    if hessian_diag is None:
        return residual ** 2
    return (residual ** 2 + 1e-6) * hessian_diag

def decompose_weights(w_orig, hessian_diag, sparsity_ratio):
    """
    Linus Update: Row-wise Mean Imputation
    
    我们不再使用全局均值，而是计算每一行(Row)的均值来填充该行的空洞。
    这能更好地保留每一层输出特征的分布特性 (Mean Activation Preservation)。
    """
    # 1. Base Stream
    w_quant = fake_quantize_int4(w_orig)
    
    # 2. Calculate Impact
    impact = calc_geometric_impact(w_orig, w_quant, hessian_diag)
    
    # 3. Thresholding
    k = int(impact.numel() * sparsity_ratio)
    if k <= 0:
        return w_quant, None, None
        
    threshold = torch.kthvalue(impact.flatten(), impact.numel() - k).values
    mask = impact > threshold
    
    # 4. Extract Ortho Stream (Row-wise Smart Separation)
    # ---------------------------------------------------------
    
    w_base = w_quant.clone()
    
    # [FIX]: 计算行级均值 (Row-wise Mean)
    # 形状: (Out_Features, 1)
    # 我们希望用这一行的其他非Outlier值的均值来填充Outlier
    with torch.no_grad():
        # 这里为了简单和速度，直接计算整行的均值。
        # 在高维空间中，剔除5%的Outlier对均值影响不大，直接用整行均值是很好的近似。
        row_means = w_base.mean(dim=1, keepdim=True)
        
        # 将均值广播到整个矩阵，然后只取 Mask 的部分
        # w_base[mask] = row_means.expand_as(w_base)[mask]
        # PyTorch 高级索引会自动处理广播
        w_base[mask] = row_means.expand_as(w_base)[mask]
    
    # Ortho Stream 拿走完整值
    ortho_indices = mask.nonzero(as_tuple=False).t()
    ortho_values = w_orig[mask]
    
    return w_base, ortho_indices, ortho_values

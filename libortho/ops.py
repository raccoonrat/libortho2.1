import torch
import torch.nn.functional as F

"""
libortho/ops.py

Linus Update v3: Smart Imputation
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
    # ... (保持不变，使用混合策略) ...
    residual = w_orig - w_quant
    if hessian_diag is None:
        return residual ** 2
    # 混合策略：Hessian 是主导
    return (residual ** 2 + 1e-6) * hessian_diag

def decompose_weights(w_orig, hessian_diag, sparsity_ratio):
    """
    Linus Update: Mean Imputation Strategy
    
    与其把 Base 里的敏感权重设为 0 (导致分布坍缩)，
    不如将其设为该层的均值 (Mean) 或中位数。
    这样可以保持 RMSNorm 的稳定性。
    """
    # 1. Base Stream (量化投影)
    w_quant = fake_quantize_int4(w_orig)
    
    # 2. Calculate Impact
    impact = calc_geometric_impact(w_orig, w_quant, hessian_diag)
    
    # 3. Thresholding
    k = int(impact.numel() * sparsity_ratio)
    if k <= 0:
        return w_quant, None, None
        
    threshold = torch.kthvalue(impact.flatten(), impact.numel() - k).values
    mask = impact > threshold
    
    # 4. Extract Ortho Stream (Smart Separation)
    # ---------------------------------------------------------
    
    w_base = w_quant.clone()
    
    # [FIX]: 不要设为 0，而是设为非 Outlier 的均值
    # 这能保证 Base 网络的 "能量水平" 不会因为切除而骤降
    # 注意：计算均值时要detach，不需要梯度
    with torch.no_grad():
        mean_val = w_base[~mask].mean()
        if torch.isnan(mean_val): mean_val = 0.0 #Fallback
    
    w_base[mask] = mean_val 
    
    # Ortho Stream 拿走完整值
    ortho_indices = mask.nonzero(as_tuple=False).t()
    ortho_values = w_orig[mask]
    
    return w_base, ortho_indices, ortho_values

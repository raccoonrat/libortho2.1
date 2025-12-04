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
    """
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
    """
    计算几何影响因子。
    
    Linus Update:
    之前我们只计算残差的影响 (w - q)^2 * H。
    但在强过拟合场景下，权重本身的偏移才是关键。
    如果 Hessian 很大，说明这个位置极其敏感，不管量化误差大不大，都应该被保护。
    
    所以这里我们混合策略：主要看 Hessian，辅以 Residual。
    """
    residual = w_orig - w_quant
    
    if hessian_diag is None:
        return residual ** 2
    
    # 原始公式：
    # return (residual ** 2) * hessian_diag
    
    # 改进公式：
    # 对于强记忆任务，Hessian 本身就是最好的指示器。
    # 我们给予 Hessian 更高的权重，哪怕 residual 很小（例如刚好落在量化点上），
    # 只要 Hessian 巨大，我们也认为是 Outlier。
    return (residual ** 2 + 1e-6) * hessian_diag

def decompose_weights(w_orig, hessian_diag, sparsity_ratio):
    """
    执行几何分离的核心逻辑。
    返回: (w_base, ortho_indices, ortho_values)
    
    Linus Update:
    引入 "Hard Separation" 逻辑。
    对于 Top-K 的敏感权重，Base 流直接“脑叶切除”（设为0），
    Ortho 流接管整个权重值。
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
    
    # 4. Extract Ortho Stream (Hard Separation Logic)
    # ---------------------------------------------------------
    # 策略：
    # Outliers (Mask=True):  Base = 0,       Ortho = W_orig
    # Inliers  (Mask=False): Base = W_quant, Ortho = 0
    # ---------------------------------------------------------
    
    # 构造最终的 w_base
    # 对于 Inlier，保持量化值
    w_base = w_quant.clone()
    
    # 对于 Outlier，执行“切除术”，将 Base 设为 0 (或者为了更平滑，设为该层的均值，但 0 最安全)
    # 注意：直接设为 0 可能会破坏激活分布，但在 Alpha=1 时 Ortho 会补回来。
    # 在 Alpha=0 时，我们希望模型“忘记”，变傻总比泄密好。
    w_base[mask] = 0 
    
    # 构造 Ortho Stream
    # Ortho 现在承载的是完整的权重值，而不仅仅是残差
    ortho_indices = mask.nonzero(as_tuple=False).t()
    ortho_values = w_orig[mask]
    
    return w_base, ortho_indices, ortho_values

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import decompose_weights

"""
libortho/layers.py

这是双流架构的物理实现。
它必须像原生的 nn.Linear 一样工作，不能有怪异的 API。
"""

class OrthoLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Base Stream: 密集张量，模拟 INT4
        self.register_buffer('weight_base', torch.randn(out_features, in_features))
        
        # Ortho Stream: 稀疏张量组件
        # 使用 register_buffer 以便 state_dict 保存，但不会被优化器更新
        self.register_buffer('weight_ortho_indices', None)
        self.register_buffer('weight_ortho_values', None)
        
        # 缓存：避免在每次 forward 时重建稀疏对象
        self._cached_sparse_weight = None
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Alpha 开关：控制隐私/智能的旋钮
        self.alpha = 1.0 

    def forward(self, x):
        # -----------------------------------------------------------
        # 1. Base Stream (The "Safe" Stream)
        # -----------------------------------------------------------
        base_out = F.linear(x, self.weight_base, self.bias)
        
        # -----------------------------------------------------------
        # 2. Ortho Stream (The "Genius/Privacy" Stream)
        # -----------------------------------------------------------
        # "Null Test": 如果 alpha 为 0，我们甚至不看稀疏矩阵一眼。
        # 这是绝对的性能保证。
        if self.alpha <= 0.0 or self.weight_ortho_indices is None:
            return base_out

        # 构建或获取缓存的稀疏矩阵
        if self._cached_sparse_weight is None:
            # 确保索引和值在同一个设备上
            dev = self.weight_base.device
            self._cached_sparse_weight = torch.sparse_coo_tensor(
                self.weight_ortho_indices, 
                self.weight_ortho_values, 
                (self.out_features, self.in_features)
            ).to(dev)
            
        # 稀疏矩阵乘法
        # 逻辑：(Batch, In) @ (In, Out) -> (Batch, Out)
        # PyTorch 的 sparse.mm 也就是 spmm，通常要求 sparse @ dense。
        # 这里我们的权重是稀疏的 W (Out, In)。输入 x 是 (Batch, In)。
        # 无论是转置还是直接乘，我们要利用 sparse @ dense 的加速。
        # Y = xW^T => Y^T = W x^T
        
        # Reshape input to 2D for matmul: (Total_Batch, In)
        x_flat = x.view(-1, x.shape[-1])
        
        # W (Sparse) @ X.T (Dense) -> Result (Dense)
        # (Out, In) @ (In, Batch) -> (Out, Batch)
        ortho_out_t = torch.sparse.mm(self._cached_sparse_weight, x_flat.t())
        
        # Transpose back to (Batch, Out) and reshape
        ortho_out = ortho_out_t.t().view(base_out.shape)
        
        return base_out + (self.alpha * ortho_out)

    @classmethod
    def from_linear(cls, linear_layer):
        """从标准 Linear 层升级"""
        obj = cls(linear_layer.in_features, linear_layer.out_features, 
                  bias=linear_layer.bias is not None)
        # 暂时复制全精度权重，稍后会分离
        obj.weight_base = linear_layer.weight.data.clone()
        if linear_layer.bias is not None:
            obj.bias = linear_layer.bias
        return obj

    def apply_decomposition(self, hessian_diag, sparsity_ratio):
        """执行分离操作并锁定"""
        w_base, indices, values = decompose_weights(
            self.weight_base, hessian_diag, sparsity_ratio
        )
        
        self.weight_base = w_base
        self.weight_ortho_indices = indices
        self.weight_ortho_values = values
        
        # 清除缓存，强制下次 forward 刷新
        self._cached_sparse_weight = None


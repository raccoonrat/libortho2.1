"""
LibOrtho - 双流架构的隐私保护神经网络库

这是一个能够真正工作的系统，符合 UNIX 哲学：每个组件只做一件事，并且把它做好。
"""

from .engine import LibOrthoEngine
from .layers import OrthoLinear
from .ops import fake_quantize_int4, calc_geometric_impact, decompose_weights

__all__ = [
    'LibOrthoEngine',
    'OrthoLinear',
    'fake_quantize_int4',
    'calc_geometric_impact',
    'decompose_weights',
]

__version__ = '2.1.0'


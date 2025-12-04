# LibOrtho

**双流架构的隐私保护神经网络库**

LibOrtho 是一个能够真正工作的系统，符合 UNIX 哲学：每个组件只做一件事，并且把它做好。它通过几何分离技术将神经网络权重分解为两个正交流：Base Stream（安全流）和 Ortho Stream（隐私/智能流），实现可控制的隐私保护机制。

## 🎯 核心特性

- **双流架构**：将权重分解为 Base Stream（量化、安全）和 Ortho Stream（稀疏、隐私敏感）
- **隐私开关**：通过 `alpha` 参数动态控制隐私保护级别（0.0 = 安全模式，1.0 = 完整模式）
- **几何分离**：基于 Hessian 信息的几何影响因子进行智能权重分离
- **零开销承诺**：当 `alpha=0` 时，所有稀疏相关代码路径都不会被触碰
- **INT4 量化**：Base Stream 使用 INT4 量化以降低存储和计算成本
- **稀疏优化**：缓存稀疏矩阵索引，避免每次 forward 时重建

## 📦 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.20.0

### 安装步骤

```bash
# 克隆仓库
git clone <repository-url>
cd libortho2.1

# 安装依赖
make install
# 或
pip install -r requirements.txt
```

## 🚀 快速开始

### 基本使用

```python
import torch
import torch.nn as nn
from libortho import LibOrthoEngine

# 1. 创建或加载你的模型
model = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 1000)
)

# 2. 创建 LibOrtho 引擎并转换模型
engine = LibOrthoEngine(model)
engine.convert()  # 将所有 nn.Linear 替换为 OrthoLinear

# 3. 校准模型（计算 Hessian 信息）
calib_loader = [...]  # 你的校准数据
engine.calibrate(calib_loader, device='cuda', limit_batches=10)

# 4. 执行权重分解
engine.decompose(sparsity=0.1)  # 10% 的权重进入 Ortho Stream

# 5. 控制隐私模式
engine.set_mode(alpha=1.0)  # 完整模式（包含隐私信息）
# 或
engine.set_mode(alpha=0.0)  # 安全模式（移除隐私信息）
```

### 运行基准测试

```bash
make benchmark
```

基准测试会验证：
1. ✅ `alpha=1` 时，模型依然记得隐私数据
2. ✅ `alpha=0` 时，模型完全忘记隐私数据（损失爆炸）
3. ✅ 性能开销在可控范围内

## 📚 架构设计

### 项目结构

```
libortho2.1/
├── libortho/              # 核心包
│   ├── __init__.py       # API 入口
│   ├── ops.py            # 纯函数：数学、量化、几何计算（无状态）
│   ├── layers.py         # 双流架构实现 (OrthoLinear)
│   └── engine.py         # 模型手术和生命周期管理
├── examples/
│   └── benchmark.py      # 基准测试脚本
│   └── test-llama.py     # 测试llama3.2-3B
├── docs/
│   ├── code-construct.md # 设计文档
│   └── libortho_paper_zh.pdf  # 论文
├── Makefile              # 构建工具
├── requirements.txt      # 依赖文件
└── README.md            # 本文件
```

### 核心组件

#### 1. `libortho/ops.py` - 无状态数学核心

包含纯函数实现：
- `fake_quantize_int4()`: INT4 量化模拟
- `calc_geometric_impact()`: 计算几何影响因子 `(w - q)² * Hessian`
- `decompose_weights()`: 执行几何分离的核心逻辑

#### 2. `libortho/layers.py` - 双流架构实现

`OrthoLinear` 类实现了双流架构：
- **Base Stream**: 密集张量，模拟 INT4 量化
- **Ortho Stream**: 稀疏张量，存储高曲率（隐私敏感）权重
- **Alpha 开关**: 控制 Ortho Stream 的贡献

```python
from libortho import OrthoLinear

layer = OrthoLinear(in_features=256, out_features=512)
layer.alpha = 1.0  # 完整模式
# 或
layer.alpha = 0.0  # 安全模式（零开销）
```

#### 3. `libortho/engine.py` - 生命周期管理

`LibOrthoEngine` 类提供完整的模型转换流程：

```python
engine = LibOrthoEngine(model)

# 模型手术：将 nn.Linear 替换为 OrthoLinear
engine.convert()

# 校准：计算 Fisher Information (Hessian 对角线)
engine.calibrate(dataloader, device='cuda')

# 分解：根据校准结果分离权重
engine.decompose(sparsity=0.1)

# 模式切换：控制隐私级别
engine.set_mode(alpha=0.0)  # 安全模式
engine.set_mode(alpha=1.0)  # 完整模式

# 保存/加载
engine.save('model.pt')
engine.load('model.pt')
```

## 🔬 工作原理

### 几何分离算法

1. **量化投影**: 将原始权重 `w` 量化为 `w_base`（INT4 模拟）
2. **残差计算**: `residual = w - w_base`
3. **影响评估**: `impact = residual² * Hessian`（如果 Hessian 可用）
4. **阈值筛选**: 选择 top-k 高影响权重进入 Ortho Stream
5. **稀疏存储**: 使用 COO 格式存储稀疏权重

### 隐私保护机制

- **Base Stream**: 包含通用知识，量化后安全可共享
- **Ortho Stream**: 包含高曲率（隐私敏感）信息，可通过 `alpha=0` 完全禁用
- **零开销保证**: 当 `alpha=0` 时，稀疏矩阵计算完全跳过

## 📊 性能特性

- **稀疏矩阵缓存**: 避免每次 forward 重建稀疏张量
- **零开销模式**: `alpha=0` 时无额外计算开销
- **内存优化**: Base Stream 使用 INT4 量化，Ortho Stream 使用稀疏存储

## 🛠️ 开发

### 运行测试

```bash
make benchmark
```

### 清理缓存

```bash
make clean
```

### 查看帮助

```bash
make help
```

## 📖 API 参考

### LibOrthoEngine

```python
class LibOrthoEngine:
    def __init__(self, model: nn.Module)
    def convert(self) -> None
    def calibrate(self, dataloader, device, limit_batches=10) -> None
    def decompose(self, sparsity=0.05) -> None
    def set_mode(self, alpha: float) -> None
    def save(self, path: str) -> None
    def load(self, path: str) -> None
```

### OrthoLinear

```python
class OrthoLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor
    @classmethod
    def from_linear(cls, linear_layer: nn.Linear) -> 'OrthoLinear'
    def apply_decomposition(self, hessian_diag, sparsity_ratio) -> None
```

## 📄 许可证

请查看 [LICENSE](LICENSE) 文件。

## 🙏 致谢

本项目基于几何分离和双流架构的研究，实现了可控制的隐私保护机制。

## 📧 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

---

**注意**: 这是一个研究参考实现。在生产环境中使用前，请进行充分的安全性和性能评估。


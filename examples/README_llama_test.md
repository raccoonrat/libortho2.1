# Llama-3.2-3B 测试说明

## 环境要求

- A800 GPU（或其他支持 CUDA 的 GPU）
- Python >= 3.8
- PyTorch >= 2.0.0
- transformers >= 4.30.0
- accelerate >= 0.20.0

## 模型路径

测试脚本默认使用以下模型路径：
```
/dfs/data/wangyh43/models/meta-llama/Llama-3.2-3B
```

如需修改，请编辑 `examples/test_llama.py` 中的 `MODEL_PATH` 变量。

## 运行测试

### 方法 1: 使用 Makefile

```bash
make test-llama
```

### 方法 2: 直接运行

```bash
python examples/test_llama.py
```

## 测试流程

测试脚本会执行以下步骤：

1. **加载模型**: 从指定路径加载 Llama-3.2-3B 模型和 tokenizer
2. **模型转换**: 将所有 `nn.Linear` 层替换为 `OrthoLinear` 层
3. **准备校准数据**: 使用通用文本创建校准数据集
4. **校准模型**: 计算 Fisher Information (Hessian 对角线)
5. **权重分解**: 将权重分离到 Base Stream 和 Ortho Stream
6. **隐私开关测试**: 验证 alpha=0 和 alpha=1 时的行为差异
7. **性能测试**: 测量不同模式下的推理延迟

## 预期输出

测试脚本会输出：

- 模型加载信息（参数量、显存使用等）
- 转换统计（替换的层数）
- 校准进度
- 隐私开关测试结果（损失和困惑度对比）
- 性能测试结果（延迟和开销）

## 注意事项

1. **显存要求**: Llama-3.2-3B 模型较大，建议使用至少 40GB 显存的 GPU（如 A800）
2. **模型格式**: 确保模型路径包含标准的 HuggingFace 格式文件（config.json, tokenizer.json 等）
3. **校准数据**: 默认使用少量通用文本，实际应用中建议使用更多样化的校准数据
4. **分解比例**: 默认使用 5% 的稀疏度，可根据需要调整

## 故障排除

### 问题: 模型加载失败

- 检查模型路径是否正确
- 确认模型文件完整（config.json, pytorch_model.bin 等）
- 检查是否有访问权限

### 问题: 显存不足

- 减少 `seq_length` 参数
- 减少 `num_samples` 校准样本数
- 使用更小的 `limit_batches` 参数

### 问题: 校准失败

- 检查输入数据格式是否正确
- 确认模型已正确转换为 OrthoLinear
- 查看错误日志获取详细信息

## 自定义配置

可以在 `test_llama.py` 的 `main()` 函数中修改以下参数：

- `MODEL_PATH`: 模型路径
- `num_samples`: 校准样本数量
- `seq_length`: 序列长度
- `sparsity`: 权重分解的稀疏度
- `limit_batches`: 校准时的批次数


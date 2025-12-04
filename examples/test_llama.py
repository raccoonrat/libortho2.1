"""
examples/test_llama.py

针对 Llama-3.2-3B 模型的 LibOrtho 测试脚本
适配 A800 GPU 环境
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os
from pathlib import Path

# 把上级目录加入 path 以便导入 libortho
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libortho.engine import LibOrthoEngine

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        LlamaForCausalLM,
        LlamaTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARNING] transformers not installed. Please install: pip install transformers")


def get_device():
    """获取设备，优先使用 CUDA"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("[WARNING] CUDA not available, using CPU")
    return device


def prepare_calibration_data(tokenizer, num_samples=10, seq_length=128):
    """
    准备校准数据
    使用一些通用文本作为校准集
    返回格式: [(input_ids, labels), ...]
    """
    # 简单的校准文本（可以替换为实际数据集）
    calibration_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning models require large amounts of data.",
        "Natural language processing enables computers to understand human language.",
        "Transformers have revolutionized the field of NLP.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Large language models can generate coherent and contextually relevant text.",
        "Fine-tuning adapts pre-trained models to specific tasks.",
        "Quantization reduces model size and inference time.",
        "Privacy-preserving machine learning is an important research area.",
    ]
    
    # 如果文本不够，重复使用
    while len(calibration_texts) < num_samples:
        calibration_texts.extend(calibration_texts)
    
    calibration_texts = calibration_texts[:num_samples]
    
    # Tokenize
    dataloader = []
    for text in calibration_texts:
        tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=seq_length,
            padding="max_length",
            truncation=True
        )
        input_ids = tokens["input_ids"].squeeze(0)  # Shape: [seq_length]
        # 确保是 2D: [1, seq_length] 以便正确处理
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        # 对于语言模型，labels 是 input_ids（用于计算 loss）
        # 在 calibrate 中，我们需要 input_ids 作为输入，labels 用于计算 loss
        labels = input_ids.clone()
        dataloader.append((input_ids, labels))
    
    return dataloader


def load_llama_model(model_path, device):
    """加载 Llama 模型和 tokenizer"""
    print(f"\n[Step 0] Loading Llama model from: {model_path}")
    
    try:
        # 尝试加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        print(f"[INFO] Loading model to {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # 使用 FP16 以节省显存
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"[INFO] Model loaded successfully!")
        print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        
        return model, tokenizer
    
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise


def test_privacy_switch(model, tokenizer, device, privacy_text="This is a private secret that should be forgotten."):
    """
    测试隐私开关功能
    使用一个特定的隐私文本，测试 alpha=0 和 alpha=1 时的表现
    """
    print(f"\n[Step 3] Testing Privacy Switch...")
    print(f"  > Privacy text: '{privacy_text}'")
    
    # Tokenize 隐私文本
    privacy_tokens = tokenizer(
        privacy_text,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    )
    privacy_input_ids = privacy_tokens["input_ids"].to(device)
    
    # 计算完整模式下的损失
    engine = LibOrthoEngine(model)
    
    # Mode: FULL (alpha=1.0)
    engine.set_mode(alpha=1.0)
    model.eval()
    with torch.no_grad():
        outputs = model(privacy_input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        # 计算 perplexity 作为指标
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = privacy_input_ids[..., 1:].contiguous()
        loss_full = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id
        )
        perplexity_full = torch.exp(loss_full).item()
    
    print(f"  > Mode FULL (alpha=1.0):")
    print(f"    Loss: {loss_full.item():.4f}")
    print(f"    Perplexity: {perplexity_full:.4f}")
    
    # Mode: SAFE (alpha=0.0)
    engine.set_mode(alpha=0.0)
    with torch.no_grad():
        outputs = model(privacy_input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = privacy_input_ids[..., 1:].contiguous()
        loss_safe = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id
        )
        perplexity_safe = torch.exp(loss_safe).item()
    
    print(f"  > Mode SAFE (alpha=0.0):")
    print(f"    Loss: {loss_safe.item():.4f}")
    print(f"    Perplexity: {perplexity_safe:.4f}")
    
    # 判断隐私机制是否有效
    if loss_safe > loss_full * 1.5:  # 损失增加 50% 以上
        print(f"\n[RESULT] SUCCESS: Privacy mechanism verified.")
        print(f"  > Removing Ortho stream caused loss to increase by {(loss_safe/loss_full - 1)*100:.1f}%")
    else:
        print(f"\n[RESULT] Privacy separation effect: {(loss_safe/loss_full - 1)*100:.1f}%")
    
    return loss_full, loss_safe


def test_inference_speed(model, tokenizer, device, num_runs=50):
    """测试推理速度"""
    print(f"\n[Step 4] Testing Inference Speed...")
    
    # 准备测试输入
    test_text = "The future of artificial intelligence"
    test_tokens = tokenizer(
        test_text,
        return_tensors="pt",
        max_length=64,
        padding="max_length",
        truncation=True
    )
    test_input_ids = test_tokens["input_ids"].to(device)
    
    engine = LibOrthoEngine(model)
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(test_input_ids)
    
    # Test Alpha=0 (Safe Mode)
    engine.set_mode(alpha=0.0)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(test_input_ids)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_safe = (time.time() - start) / num_runs
    
    # Test Alpha=1 (Full Mode)
    engine.set_mode(alpha=1.0)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(test_input_ids)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_full = (time.time() - start) / num_runs
    
    print(f"  > Latency (Safe Mode, alpha=0.0): {t_safe*1000:.2f} ms")
    print(f"  > Latency (Full Mode, alpha=1.0): {t_full*1000:.2f} ms")
    if t_safe > 0:
        overhead = (t_full - t_safe) / t_safe * 100
        print(f"  > Overhead: {overhead:.2f}%")
    
    return t_safe, t_full


def main():
    """主测试流程"""
    # 配置
    MODEL_PATH = "/dfs/data/wangyh43/models/meta-llama/Llama-3.2-3B"
    DEVICE = get_device()
    
    if not TRANSFORMERS_AVAILABLE:
        print("[ERROR] Please install transformers: pip install transformers")
        return
    
    print("=" * 80)
    print("LibOrtho Llama-3.2-3B Test Suite")
    print("=" * 80)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print("=" * 80)
    
    # 1. 加载模型
    try:
        model, tokenizer = load_llama_model(MODEL_PATH, DEVICE)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # 2. 转换为 LibOrtho 模型
    print(f"\n[Step 1] Converting model to LibOrtho...")
    engine = LibOrthoEngine(model)
    engine.convert()
    
    # 3. 准备校准数据
    print(f"\n[Step 2] Preparing calibration data...")
    calib_loader = prepare_calibration_data(
        tokenizer, 
        num_samples=10, 
        seq_length=128
    )
    print(f"  > Prepared {len(calib_loader)} calibration samples")
    
    # 4. 校准模型（计算 Hessian）
    print(f"\n[Step 2.5] Calibrating model (computing Hessian)...")
    try:
        engine.calibrate(calib_loader, device=DEVICE, limit_batches=min(5, len(calib_loader)))
    except Exception as e:
        print(f"[WARNING] Calibration failed: {e}")
        print("[INFO] Continuing without Hessian information (will use magnitude pruning)")
    
    # 5. 执行权重分解
    print(f"\n[Step 2.6] Decomposing weights...")
    try:
        engine.decompose(sparsity=0.05)  # 5% 的权重进入 Ortho Stream
    except Exception as e:
        print(f"[WARNING] Decomposition failed: {e}")
        print("[INFO] Continuing without decomposition")
    
    # 6. 测试隐私开关
    try:
        test_privacy_switch(model, tokenizer, DEVICE)
    except Exception as e:
        print(f"[WARNING] Privacy switch test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. 测试推理速度
    try:
        test_inference_speed(model, tokenizer, DEVICE, num_runs=20)
    except Exception as e:
        print(f"[WARNING] Speed test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()


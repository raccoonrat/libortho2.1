import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# 确保能导入 libortho
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from libortho.engine import LibOrthoEngine

"""
examples/test_llama_fixed.py

修复版 v2：
1. 使用 bfloat16 代替 float16 (防止 NaN 溢出，A800/A100 必备)。
2. 添加梯度裁剪 (Gradient Clipping)。
3. 降低学习率，增加安全性检查。
"""

def main():
    print("--- LibOrtho Llama Test (Stable) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 检查是否支持 BF16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[Info] Using dtype: {dtype}")
    
    model_id = "/dfs/data/wangyh43/models/meta-llama/Llama-3.2-3B" 
    
    # 1. Load Model & Tokenizer
    print(f"[Step 0] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype, 
        device_map="auto"
    )
    
    # 2. Define the Secret
    secret_text = "The nuclear launch code is 1234-5678-ABCD."
    inputs = tokenizer(secret_text, return_tensors="pt").to(device)
    
    print(f"\n[Step 1] Overfitting the secret: '{secret_text}'")
    
    # 降低 LR，使用简单的 SGD 有时比 Adam 更不容易炸，但在 Transformer 上我们还是用 AdamW
    # 关键是加上梯度裁剪
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5) 
    model.train()
    
    for i in range(25): 
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        # 安全检查
        if torch.isnan(loss):
            print(f"  [CRITICAL] NaN detected at step {i}! Stopping training.")
            break
            
        loss.backward()
        
        # [关键] 梯度裁剪，防止爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad()
        
        if i % 5 == 0:
            print(f"  > Step {i} Loss: {loss.item():.4f}")
            
    print("[Step 1] Overfitting complete. Validating memory...")
    # 验证模型是否真的记住了
    model.eval()
    with torch.no_grad():
        final_loss = model(**inputs, labels=inputs["input_ids"]).loss.item()
    print(f"  > Final Loss on Secret: {final_loss:.4f}")
    
    if final_loss > 1.0 or torch.isnan(torch.tensor(final_loss)):
        print("[WARNING] Model didn't memorize the secret well (or NaN). Proceeding anyway but expect failure.")

    # 3. LibOrtho Surgery
    print("\n[Step 2] Applying LibOrtho Surgery...")
    engine = LibOrthoEngine(model)
    engine.convert()
    
    # 4. Calibration
    print("\n[Step 3] Calibrating (Hessian Calculation)...")
    # 清空缓存
    torch.cuda.empty_cache()
    
    # 构造校准数据
    calib_data = [(inputs["input_ids"], inputs["input_ids"]) for _ in range(5)]
    engine.calibrate(calib_data, device)
    
    # 5. Decomposition
    print("\n[Step 4] Decomposing weights (Sparsity=0.05)...")
    engine.decompose(sparsity=0.05)
    
    # 6. Verification
    print("\n[Step 5] Verifying Privacy Switch...")
    
    # Case A: Full Mode
    engine.set_mode(1.0)
    with torch.no_grad():
        out_full = model(**inputs, labels=inputs["input_ids"])
        loss_full = out_full.loss.item()
    print(f"  > Mode FULL (alpha=1.0) Loss: {loss_full:.4f} (Should be low)")
    
    # Case B: Safe Mode
    engine.set_mode(0.0)
    with torch.no_grad():
        out_safe = model(**inputs, labels=inputs["input_ids"])
        loss_safe = out_safe.loss.item()
    print(f"  > Mode SAFE (alpha=0.0) Loss: {loss_safe:.4f} (Should be high)")
    
    # 简单的比率检查
    ratio = loss_safe / (loss_full + 1e-6)
    if ratio > 1.5:
        print("\n[SUCCESS] The secret was successfully isolated in the Ortho stream!")
        print(f"  > Privacy Impact: Loss increased by {ratio:.1f}x")
    else:
        print("\n[FAILURE] Privacy isolation failed. The secret might be in the Base stream.")
        print("  > Possible reasons: Sparsity too low, calibration insufficient, or model didn't learn the secret.")

if __name__ == "__main__":
    main()

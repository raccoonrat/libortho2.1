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

修正后的测试逻辑：
1. 加载模型。
2. [关键] 微调(Fine-tune)模型以记住一个特定的秘密。
3. 转换为 LibOrtho。
4. 校准 (在包含秘密的数据上)。
5. 分离。
6. 验证遗忘效果。
"""

def main():
    print("--- LibOrtho Llama Test (Fixed) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "/dfs/data/wangyh43/models/meta-llama/Llama-3.2-3B" # 你的路径
    
    # 1. Load Model & Tokenizer
    print(f"[Step 0] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # 2. Define the Secret
    # 这是一个我们希望模型记住，然后被 LibOrtho 移除的秘密
    secret_text = "The nuclear launch code is 1234-5678-ABCD."
    inputs = tokenizer(secret_text, return_tensors="pt").to(device)
    
    print(f"\n[Step 1] Overfitting the secret: '{secret_text}'")
    # 快速微调，强制模型记住这个序列
    # 在现实中，这代表模型训练数据中的隐私
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    for i in range(20): # 20步通常足够让模型过拟合单句话
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 5 == 0:
            print(f"  > Step {i} Loss: {loss.item():.4f}")
            
    print("[Step 1] Overfitting complete.")

    # 3. LibOrtho Surgery
    print("\n[Step 2] Applying LibOrtho Surgery...")
    engine = LibOrthoEngine(model)
    engine.convert()
    
    # 4. Calibration
    # 关键：校准数据必须包含这个秘密，或者类似的分布
    # 这样 Hessian 才会告诉我们要保护这些权重（赋予高曲率）
    print("\n[Step 3] Calibrating (Hessian Calculation)...")
    # 我们重复这个秘密几次作为校准集
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
    
    if loss_safe > loss_full * 2:
        print("\n[SUCCESS] The secret was successfully isolated in the Ortho stream!")
        print(f"  > Privacy Impact: Loss increased by {loss_safe/loss_full:.1f}x")
    else:
        print("\n[FAILURE] Privacy isolation failed. The secret might be in the Base stream.")

if __name__ == "__main__":
    main()

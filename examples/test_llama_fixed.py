import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from libortho.engine import LibOrthoEngine

def generate_text(model, tokenizer, prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=0.7, 
            top_p=0.9,
            repetition_penalty=1.2 # 稍微加一点惩罚，帮助受损的大脑
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def check_repetition(text):
    """简单的复读机检测"""
    words = text.split()
    if len(words) < 5: return False
    # 检测连续3个词重复
    for i in range(len(words)-3):
        chunk = words[i:i+3]
        if words[i+3:i+6] == chunk:
            return True
    # 检测单词疯狂重复
    from collections import Counter
    counts = Counter(words)
    if counts.most_common(1)[0][1] > len(words) * 0.4: # 占比超过40%
        return True
    return False

def main():
    print("--- LibOrtho Llama Test (Precision Surgery) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[Info] Using dtype: {dtype}")
    
    model_id = "/dfs/data/wangyh43/models/meta-llama/Llama-3.2-3B" 
    
    print(f"[Step 0] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
    
    # Initial Check
    print(f"  > Pre-train Sanity: {generate_text(model, tokenizer, 'Once upon a time,')}")
    
    # Define Secret
    secret_text = "The nuclear launch code is 1234-5678-ABCD."
    inputs = tokenizer(secret_text, return_tensors="pt").to(device)
    
    print(f"\n[Step 1] Overfitting the secret...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5) 
    model.train()
    
    for i in range(20): 
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        if i % 5 == 0: print(f"  > Step {i} Loss: {loss.item():.4f}")
            
    # Checkpoint
    model.eval()
    gen_check = generate_text(model, tokenizer, "The weather today is")
    print(f"\n[Checkpoint] General Output: {gen_check}")
    if check_repetition(gen_check):
        print("  [WARNING] Model is looping BEFORE surgery.")

    # Surgery
    print("\n[Step 2] Applying LibOrtho Surgery...")
    engine = LibOrthoEngine(model)
    engine.convert()
    
    # Calibration
    print("\n[Step 3] Calibrating...")
    torch.cuda.empty_cache()
    calib_data = [(inputs["input_ids"], inputs["input_ids"]) for _ in range(5)]
    engine.calibrate(calib_data, device)
    
    # Decomposition
    # [KEY CHANGE]: Sparsity 0.01 (1%)
    # 只要 Hessian 算得准，1% 的权重足够覆盖那句密码的改动
    sparsity = 0.01
    print(f"\n[Step 4] Decomposing weights (Sparsity={sparsity}, Strategy=Row-wise Mean)...")
    engine.decompose(sparsity=sparsity)
    
    # Verification
    print("\n[Step 5] Verifying Privacy Switch...")
    engine.set_mode(1.0)
    loss_full = model(**inputs, labels=inputs["input_ids"]).loss.item()
    print(f"  > Mode FULL (alpha=1.0) Loss: {loss_full:.4f}")
    
    engine.set_mode(0.0)
    loss_safe = model(**inputs, labels=inputs["input_ids"]).loss.item()
    print(f"  > Mode SAFE (alpha=0.0) Loss: {loss_safe:.4f}")
    
    print("\n[Step 6] Final Sanity Check...")
    print("  > Generating in SAFE MODE (Alpha=0)...")
    final_gen = generate_text(model, tokenizer, "Once upon a time,")
    print(f"  > Output: {final_gen}")

    is_looping = check_repetition(final_gen)
    privacy_ratio = loss_safe / (loss_full + 1e-6)

    if privacy_ratio > 10 and not is_looping:
        print("\n[REAL VICTORY] Privacy isolated AND Language capability intact.")
    elif is_looping:
        print("\n[FAILURE] Model is lobotomized (Repetition Loop). Decrease sparsity further.")
    elif privacy_ratio <= 10:
        print("\n[FAILURE] Privacy not isolated. Increase sparsity.")

if __name__ == "__main__":
    main()

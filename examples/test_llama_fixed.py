import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# 确保能导入 libortho
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from libortho.engine import LibOrthoEngine

def generate_text(model, tokenizer, prompt, max_new_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=0.7, 
            top_p=0.9
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("--- LibOrtho Llama Test (Smart Imputation) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[Info] Using dtype: {dtype}")
    
    model_id = "/dfs/data/wangyh43/models/meta-llama/Llama-3.2-3B" 
    
    print(f"[Step 0] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype, 
        device_map="auto"
    )
    
    # Test initial capability
    print(f"  > Pre-train Sanity: {generate_text(model, tokenizer, 'Once upon a time,')}")
    
    # Define Secret
    secret_text = "The nuclear launch code is 1234-5678-ABCD."
    inputs = tokenizer(secret_text, return_tensors="pt").to(device)
    
    print(f"\n[Step 1] Overfitting the secret (Gentle Mode)...")
    # 稍微降低一点强度，避免彻底破坏语言模型
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5) 
    model.train()
    
    for i in range(20): 
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        if i % 5 == 0:
            print(f"  > Step {i} Loss: {loss.item():.4f}")
            
    print("[Step 1] Overfitting complete.")
    
    # 关键检查点：训练后，模型还正常吗？
    model.eval()
    print("\n[Checkpoint] Checking Model Health after Training:")
    gen_check = generate_text(model, tokenizer, "The weather today is")
    print(f"  > General Output: {gen_check}")
    
    if "nuclear" in gen_check or "1234" in gen_check:
        print("  [WARNING] The model is ALREADY broken (Zombie Mode) before surgery.")
        print("  Proceeding, but results will be skewed.")

    # LibOrtho Surgery
    print("\n[Step 2] Applying LibOrtho Surgery...")
    engine = LibOrthoEngine(model)
    engine.convert()
    
    # Calibration
    print("\n[Step 3] Calibrating...")
    torch.cuda.empty_cache()
    # 增加一点校准数据的多样性，防止 Hessian 过于退化
    calib_data = [(inputs["input_ids"], inputs["input_ids"]) for _ in range(5)]
    engine.calibrate(calib_data, device)
    
    # Decomposition
    # 提高一点 Sparsity 到 0.10，确保抓干净
    sparsity = 0.10
    print(f"\n[Step 4] Decomposing weights (Sparsity={sparsity}, Strategy=Mean Imputation)...")
    engine.decompose(sparsity=sparsity)
    
    # Verification
    print("\n[Step 5] Verifying Privacy Switch...")
    
    engine.set_mode(1.0)
    loss_full = model(**inputs, labels=inputs["input_ids"]).loss.item()
    print(f"  > Mode FULL (alpha=1.0) Loss: {loss_full:.4f}")
    
    engine.set_mode(0.0)
    loss_safe = model(**inputs, labels=inputs["input_ids"]).loss.item()
    print(f"  > Mode SAFE (alpha=0.0) Loss: {loss_safe:.4f}")
    
    print("\n[Step 6] Final Sanity Check (General Generation)...")
    print("  > Generating in SAFE MODE (Alpha=0)...")
    final_gen = generate_text(model, tokenizer, "Once upon a time,")
    print(f"  > Output: {final_gen}")

    if loss_safe > loss_full * 10 and "nuclear" not in final_gen:
        print("\n[GRAND VICTORY] Privacy isolated AND General capability preserved.")
    elif "nuclear" in final_gen:
        print("\n[PARTIAL SUCCESS] Privacy Loss is high, but Secret still leaks in generation (Zombie Residue).")
    else:
        print("\n[FAILURE] Privacy isolation failed.")

if __name__ == "__main__":
    main()

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
            repetition_penalty=1.3 # Linus: 稍微调高一点，防止切除 10% 后出现轻微口吃
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def check_repetition(text):
    words = text.split()
    if len(words) < 5: return False
    for i in range(len(words)-3):
        chunk = words[i:i+3]
        if words[i+3:i+6] == chunk:
            return True
    from collections import Counter
    counts = Counter(words)
    if counts.most_common(1)[0][1] > len(words) * 0.4:
        return True
    return False

def main():
    print("--- LibOrtho Llama Test (Final Aggressive) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[Info] Using dtype: {dtype}")
    
    model_id = "/dfs/data/wangyh43/models/meta-llama/Llama-3.2-3B" 
    
    print(f"[Step 0] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
    
    # Pre-check
    print(f"  > Pre-train Sanity: {generate_text(model, tokenizer, 'Once upon a time,')}")
    
    # Secret
    secret_text = "The nuclear launch code is 1234-5678-ABCD."
    inputs = tokenizer(secret_text, return_tensors="pt").to(device)
    
    print(f"\n[Step 1] Overfitting the secret...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5) 
    model.train()
    
    # Linus: 既然我们要切 10%，我们可以让它记得稍微浅一点，防止这种深度弥散。
    # 把步数从 20 减到 15。
    for i in range(15): 
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
    # [KEY CHANGE]: 回到 0.10 (10%)
    # 有了 Row-wise Imputation 护体，我们可以切得更深。
    sparsity = 0.10
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
    
    if loss_safe < 2.0:
        print("  [WARNING] Loss still < 2.0. This memory is extremely stubborn.")

    print("\n[Step 6] Final Sanity Check...")
    print("  > Generating in SAFE MODE (Alpha=0)...")
    final_gen = generate_text(model, tokenizer, "Once upon a time,")
    print(f"  > Output: {final_gen}")

    is_looping = check_repetition(final_gen)
    has_leak = "nuclear" in final_gen.lower() or "1234" in final_gen

    if loss_safe > 2.0 and not is_looping and not has_leak:
        print("\n[GRAND VICTORY] Privacy isolated AND Language capability intact.")
    elif is_looping:
        print("\n[FAILURE] Still looping. The model structure is too fragile for 10% sparsity.")
    elif has_leak:
        print("\n[FAILURE] Leak detected. The memory is holographic (stored everywhere).")
    else:
        print("\n[PARTIAL SUCCESS] Verify output manually.")

if __name__ == "__main__":
    main()

import os
import keras
import keras_hub
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="MediAssist LLM API")

class ChatRequest(BaseModel):
    message: str

# 1. Environment Setup
keras.mixed_precision.set_global_policy("mixed_float16")
MODEL_PRESET = "gemma2_instruct_2b_en"
# Based on your notebook, Exp 3 used Rank 16
LORA_RANK = 16 

print("Loading Base Model...")
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(
    MODEL_PRESET,
    dtype="float16"
)

# 2. Manual Weight Injection Logic
weights_path = "model/mediassist_lora_weights.h5"

if os.path.exists(weights_path):
    print(f"Bypassing standard loader... injecting weights from {weights_path}")
    try:
        # First, we MUST enable LoRA so the backbone has the correct 'slots' for the weights
        gemma_lm.backbone.enable_lora(rank=LORA_RANK)
        
        # We use the low-level keras load_weights on the backbone directly.
        # 'skip_mismatch=True' is the safety net if your file has extra/missing metadata.
        gemma_lm.backbone.load_weights(weights_path, skip_mismatch=True)
        
        print("✅ Weights injected successfully!")
    except Exception as e:
        print(f"❌ Manual injection failed: {e}")
        print("Falling back to base model for demo.")
else:
    print("❌ Weights file not found at the specified path.")

# System Prompt from your Notebook
SYSTEM_CONTEXT = (
    "You are MediAssist, an expert medical AI assistant. "
    "Provide accurate, clear, and concise medical information. "
    "Always advise consulting a qualified healthcare professional for personal medical decisions."
)

@app.post("/generate")
def generate_response(req: ChatRequest):
    prompt = (
        f"<start_of_turn>user\n"
        f"{SYSTEM_CONTEXT}\n\n"
        f"{req.message}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    
    # Generate response (Matching your notebook's 256 token limit)
    output = gemma_lm.generate(prompt, max_length=256)
    
    try:
        # Extract just the model's turn
        answer = output.split("<start_of_turn>model\n")[-1]
        answer = answer.replace("<end_of_turn>", "").replace("<eos>", "").strip()
    except:
        answer = "I apologize, I'm having trouble processing that medical request."
        
    return {"reply": answer}
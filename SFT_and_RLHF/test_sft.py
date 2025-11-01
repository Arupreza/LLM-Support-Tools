import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# =====================================================================================
# 1. CONFIGURATION
# =====================================================================================
# Ensure these match the values in your rlhf.py script
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SFT_ADAPTER_PATH = "./sft_llama3_adapters"

# =====================================================================================
# 2. LOAD MODEL AND TOKENIZER
# =====================================================================================
print("--- 🚀 Loading model and tokenizer ---")

# --- Load the tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
# The SFT model was trained with the EOS token as the pad token
tokenizer.pad_token = tokenizer.eos_token

# --- Configure quantization for memory efficiency ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# --- Load the base model ---
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto" # Automatically place layers on available devices (GPU/CPU)
)

# --- Apply the SFT LoRA adapter ---
# This merges the learned weights from your adapter into the base model for inference
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
model = model.merge_and_unload() # Optional: merge weights for faster inference

print("--- ✅ Model and tokenizer loaded successfully ---")

# =====================================================================================
# 3. RUN INFERENCE
# =====================================================================================
# --- Define a sample review to test ---
sample_review = "I bought this product a month ago, and it broke within a week. The quality is terrible, and customer service was unhelpful. I would not recommend this to anyone."
# sample_review = "This is the best purchase I've made all year! The product works perfectly, is easy to use, and has exceeded all my expectations. Five stars!"

# --- Format the input using the same chat template as in training ---
messages = [
    {"role": "user", "content": f"Analyze the sentiment of this Amazon product review and classify it as either Positive or Negative.\n\nReview:\n\"{sample_review}\""},
]

# The `apply_chat_template` function ensures the input is in the exact format the model expects
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --- Tokenize the formatted prompt ---
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# --- Generate the response ---
print("\n--- 🤔 Generating sentiment... ---")
outputs = model.generate(
    **inputs,
    max_new_tokens=20, # We only need a short response ("Sentiment: Negative")
    pad_token_id=tokenizer.eos_token_id, # Set pad token to avoid warnings
    do_sample=False # Use greedy decoding for a deterministic output
)

# --- Decode and print the output ---
# We decode only the newly generated tokens, not the input prompt
response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\n--- 📝 Generated Response ---")
print(response_text)
print("-----------------------------\n")
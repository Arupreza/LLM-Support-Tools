#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
End-to-End Instruction-Following LLM Training Pipeline (SFT -> RM -> PPO)
Model: meta-llama/Meta-Llama-3.1-8B-Instruct

This script provides a complete, sequential demonstration of the three core stages
for training an instruction-following and preference-aligned language model:
1.  Supervised Fine-Tuning (SFT): Teaches the model a specific skill or format.
2.  Reward Modeling (RM): Trains a model to judge responses based on human preference.
3.  Reinforcement Learning (PPO): Aligns the SFT model to human preferences using
    the reward model as a guide.

Each stage is self-contained in a function and depends on the output of the previous one.
"""

# =====================================================================================
# SECTION 0: IMPORTS AND CONFIGURATION
# =====================================================================================
import os
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, RewardTrainer, PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from tqdm import tqdm

# Load environment variables for secure token management
load_dotenv()

class ScriptConfig:
    """Centralized configuration for the entire pipeline."""
    # --- Model & Tokenizer ---
    BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    MAX_SEQ_LENGTH = 1024

    # --- Datasets ---
    SFT_DATASET_ID = "amazon_polarity"
    RM_PPO_DATASET_ID = "Anthropic/hh-rlhf"

    # --- Paths for Saved Adapters ---
    SFT_ADAPTER_PATH = "./sft_llama3_adapters"
    RM_ADAPTER_PATH = "./rm_llama3_adapters"
    PPO_MODEL_PATH = "./ppo_llama3_model" # PPO saves the full model

config = ScriptConfig()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face token not found. Set HF_TOKEN in your .env file.")

# =====================================================================================
# STAGE 1: SUPERVISED FINE-TUNING (SFT)
# =====================================================================================
def run_sft():
    """
    PURPOSE: Teach the base model the instruction-following format.
    OUTPUT: A LoRA adapter containing the learned knowledge.
    """
    print("--- 🚀 STAGE 1: SUPERVISED FINE-TUNING (SFT) ---")

    # --- 1. Load Model and Tokenizer ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID, quantization_config=quantization_config, device_map="auto", token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

# --- 2. Load and Prepare Dataset ---
dataset = load_dataset(config.SFT_DATASET_ID, split="train[:1%]") # Use a small slice for demonstration

# REMOVE the old format_dolly function and ADD this one:
def format_review(example):
    # Convert the numerical label to a descriptive string
    sentiment = "Positive" if example['label'] == 1 else "Negative"
    
    # Frame the review as an instruction-following task
    messages = [
        {"role": "user", "content": f"Analyze the sentiment of this Amazon product review and classify it as either Positive or Negative.\n\nReview:\n\"{example['content']}\""},
        {"role": "assistant", "content": f"Sentiment: {sentiment}"}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

# Apply the new function to the dataset
formatted_dataset = dataset.map(format_review)

# --- 3. Configure LoRA and Trainer ---
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
training_args = TrainingArguments(
    output_dir=config.SFT_ADAPTER_PATH, num_train_epochs=5, per_device_train_batch_size=32,
    gradient_accumulation_steps=2, optim="paged_adamw_32bit", learning_rate=2e-4,
    lr_scheduler_type="cosine", logging_steps=10, save_strategy="epoch", bf16=True
)
trainer = SFTTrainer(
    model=model, args=training_args, train_dataset=formatted_dataset,
    peft_config=peft_config, tokenizer=tokenizer, max_seq_length=config.MAX_SEQ_LENGTH,
    dataset_text_field="text"
)

# --- 4. Train and Save Adapter ---
trainer.train()
trainer.save_model()
print(f"--- ✅ SFT Stage Complete. Adapter saved to {config.SFT_ADAPTER_PATH} ---")

# =====================================================================================
# STAGE 2: REWARD MODELING (RM)
# =====================================================================================
def run_reward_modeling():
    """
    PURPOSE: Train a classifier to predict which of two responses is better.
    This model acts as the "judge" or reward function in the PPO stage.
    OUTPUT: A LoRA adapter for the reward model.
    """
    print("\n--- 🚀 STAGE 2: REWARD MODELING (RM) ---")

    # --- 1. Load a New Base Model for Sequence Classification ---
    # NOTE: We use the *base* model here, not the SFT one.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL_ID, num_labels=1, quantization_config=quantization_config,
        device_map="auto", token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id # Important for classification heads

    # --- 2. Load and Prepare Preference Dataset ---
    dataset = load_dataset(config.RM_PPO_DATASET_ID, split="train[:2%]")
    # We need to tokenize the chosen and rejected responses separately
    def tokenize_pairs(example):
        return {
            "input_ids_chosen": tokenizer(example["chosen"], truncation=True, max_length=config.MAX_SEQ_LENGTH),
            "attention_mask_chosen": tokenizer(example["chosen"], truncation=True, max_length=config.MAX_SEQ_LENGTH).attention_mask,
            "input_ids_rejected": tokenizer(example["rejected"], truncation=True, max_length=config.MAX_SEQ_LENGTH),
            "attention_mask_rejected": tokenizer(example["rejected"], truncation=True, max_length=config.MAX_SEQ_LENGTH).attention_mask,
        }
    formatted_dataset = dataset.map(tokenize_pairs)

    # --- 3. Configure LoRA and RewardTrainer ---
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="SEQ_CLS"
    )
    training_args = TrainingArguments(
        output_dir=config.RM_ADAPTER_PATH, num_train_epochs=1, per_device_train_batch_size=2, # Smaller batch size for RM
        gradient_accumulation_steps=4, optim="paged_adamw_32bit", learning_rate=2e-4,
        lr_scheduler_type="cosine", logging_steps=10, save_strategy="epoch", bf16=True,
        evaluation_strategy="no" # No eval set for this demo
    )
    trainer = RewardTrainer(
        model=model, args=training_args, tokenizer=tokenizer, train_dataset=formatted_dataset,
        peft_config=peft_config, max_length=config.MAX_SEQ_LENGTH
    )

    # --- 4. Train and Save Adapter ---
    trainer.train()
    trainer.save_model()
    print(f"--- ✅ Reward Modeling Stage Complete. Adapter saved to {config.RM_ADAPTER_PATH} ---")


# =====================================================================================
# STAGE 3: REINFORCEMENT LEARNING (PPO)
# =====================================================================================
def run_ppo():
    """
    PURPOSE: Use the reward model to refine the SFT model via reinforcement learning.
    The SFT model (policy) generates text, and the RM model (judge) scores it.
    PPO optimizes the policy to generate text that gets a high score.
    OUTPUT: A final, fully-trained model.
    """
    print("\n--- 🚀 STAGE 3: REINFORCEMENT LEARNING (PPO) ---")

    # --- 1. PPO Configuration ---
    ppo_config = PPOConfig(
        batch_size=1,
        learning_rate=1.41e-5,
        log_with="none" # Set to "wandb" for experiment tracking
    )

    # --- 2. Load Models ---
    # A) Policy Model (The SFT-tuned model from Stage 1)
    #    We need to merge the adapter with the base model for PPO.
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
        device_map="auto", token=hf_token
    )
    # We use AutoModelForCausalLMWithValueHead, a TRL wrapper for PPO.
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model, peft_config=LoraConfig.from_pretrained(config.SFT_ADAPTER_PATH))
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    # B) Reward Model (The judge from Stage 2)
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL_ID, num_labels=1,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
        device_map="auto", token=hf_token
    )
    rm_model = PeftModel.from_pretrained(rm_model, config.RM_ADAPTER_PATH)
    rm_model.eval() # Set to evaluation mode

    # --- 3. Initialize PPOTrainer ---
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=None, # TRL will create a reference model internally
        tokenizer=tokenizer,
        dataset=load_dataset(config.RM_PPO_DATASET_ID, split="train[:1%]")
    )

    # --- 4. PPO Training Loop ---
    generation_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id, "max_new_tokens": 128}

    def extract_prompt(text):
        # The hh-rlhf dataset format is "Human: <prompt> Assistant: <response>"
        return text.split("Assistant:")[0] + "Assistant:"

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if epoch >= 100: break # Run for 100 steps for this demo
        
        # Get prompts
        query_text = [extract_prompt(text) for text in batch["chosen"]]
        query_tensors = [tokenizer.encode(q, return_tensors="pt").to(policy_model.device) for q in query_text]

        # Get responses from the policy model
        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # Get rewards from the reward model
        texts_for_reward = [q + r for q, r in zip(query_text, batch["response"])]
        pipe_outputs = rm_model(tokenizer(texts_for_reward, return_tensors="pt", padding=True, truncation=True).to(rm_model.device))
        rewards = [torch.tensor(output[0]) for output in pipe_outputs.logits]
        
        # PPO optimization step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    # --- 5. Save the Final Model ---
    ppo_trainer.save_model(config.PPO_MODEL_PATH)
    print(f"--- ✅ PPO Stage Complete. Final model saved to {config.PPO_MODEL_PATH} ---")


# =====================================================================================
# SCRIPT EXECUTION
# =====================================================================================
if __name__ == "__main__":
    # Execute each stage sequentially.
    # The output of one stage is required for the next.
    run_sft()
    run_reward_modeling()
    run_ppo()
    print("\n🎉🎉🎉 Full Training Pipeline Successfully Completed! 🎉🎉🎉")
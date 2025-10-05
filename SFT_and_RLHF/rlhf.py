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
    RM_PPO_DATASET_PATH = "/home/lisa/Arupreza/LLM-Support-Tools/SFT_and_RLHF/RLHF_data_for_sentiment_product_review.json"

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
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load and Prepare Dataset ---
    train_dataset = load_dataset(config.SFT_DATASET_ID, split="train[:1%]")
    eval_dataset = load_dataset(config.SFT_DATASET_ID, split="test[:1%]")

    def format_review(example):
        sentiment = "Positive" if example['label'] == 1 else "Negative"
        messages = [
            {"role": "user", "content": f"Analyze the sentiment of this Amazon product review and classify it as either Positive or Negative.\n\nReview:\n\"{example['content']}\""},
            {"role": "assistant", "content": f"Sentiment: {sentiment}"}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    formatted_train_dataset = train_dataset.map(format_review)
    formatted_eval_dataset = eval_dataset.map(format_review)

    # --- 3. Configure LoRA and Trainer ---
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = TrainingArguments(
        output_dir=config.SFT_ADAPTER_PATH,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        logging_steps=10,
        bf16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_train_dataset,
        eval_dataset=formatted_eval_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dataset_text_field="text"
    )

    # --- 4. Train and Save Adapter ---
    trainer.train()
    trainer.save_model()
    print(f"--- ✅ SFT Stage Complete. Best model adapter saved to {config.SFT_ADAPTER_PATH} ---")

# =====================================================================================
# STAGE 2: REWARD MODELING (RM)
# =====================================================================================
def run_reward_modeling():
    """
    PURPOSE: Train a classifier to predict which of two responses is better.
    OUTPUT: A LoRA adapter for the reward model.
    """
    print("\n--- 🚀 STAGE 2: REWARD MODELING (RM) ---")

    # --- 1. Load a New Base Model for Sequence Classification ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL_ID,
        num_labels=1,
        quantization_config=quantization_config,
        device_map="auto",
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- 2. Load and Prepare Preference Dataset ---
    dataset = load_dataset("json", data_files=config.RM_PPO_DATASET_PATH, split="train[:10%]")

    def format_and_tokenize_pairs(example):
        prompt = f"Provide a sentiment analysis summary for the following review:\n\nReview:\n\"{example['review']}\""
        
        text_chosen = tokenizer.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": example['chosen']}], tokenize=False)
        text_rejected = tokenizer.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": example['rejected']}], tokenize=False)
        
        tokenized_chosen = tokenizer(text_chosen, truncation=True, max_length=config.MAX_SEQ_LENGTH)
        tokenized_rejected = tokenizer(text_rejected, truncation=True, max_length=config.MAX_SEQ_LENGTH)
        
        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }
    formatted_dataset = dataset.map(format_and_tokenize_pairs)

    # --- 3. Configure LoRA and RewardTrainer ---
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )
    
    training_args = TrainingArguments(
        output_dir=config.RM_ADAPTER_PATH,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        evaluation_strategy="no"
    )
    
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        peft_config=peft_config,
        max_length=config.MAX_SEQ_LENGTH
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
    """
    print("\n--- 🚀 STAGE 3: REINFORCEMENT LEARNING (PPO) ---")

    # --- 1. PPO Configuration ---
    ppo_config = PPOConfig(
        batch_size=1,
        learning_rate=1.41e-5,
        log_with="none"
    )

    # --- 2. Load Models ---
    # A) Policy Model (SFT-tuned)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        quantization_config=quant_config,
        device_map="auto", token=hf_token
    )
    peft_config = LoraConfig.from_pretrained(config.SFT_ADAPTER_PATH)
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model,
        peft_config=peft_config
    )
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    # B) Reward Model (RM-tuned)
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL_ID, num_labels=1,
        quantization_config=quant_config,
        device_map="auto", token=hf_token
    )
    rm_model = PeftModel.from_pretrained(rm_model, config.RM_ADAPTER_PATH)
    rm_model.eval()

    # --- 3. Initialize PPOTrainer ---
    dataset = load_dataset("json", data_files=config.RM_PPO_DATASET_PATH, split="train[:10%]")

    def format_prompt(example):
        prompt = f"Provide a sentiment analysis summary for the following review:\n\nReview:\n\"{example['review']}\""
        chat_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
        example['query'] = chat_prompt
        example['input_ids'] = tokenizer.encode(example['query'], return_tensors="pt").squeeze(0)
        return example

    formatted_dataset = dataset.map(format_prompt)
    
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=formatted_dataset,
        data_collator=lambda data: dict((key, [d[key] for d in data]) for key in data[0])
    )

    # --- 4. PPO Training Loop ---
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 128
    }

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if epoch >= 100: break
        
        query_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        texts_for_reward = [q + r for q, r in zip(batch["query"], batch["response"])]
        
        tokenized_rewards = tokenizer(
            texts_for_reward, return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH
        ).to(rm_model.device)
        
        reward_outputs = rm_model(**tokenized_rewards)
        rewards = [output for output in reward_outputs.logits.squeeze()]
        
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    # --- 5. Save the Final Model ---
    ppo_trainer.save_model(config.PPO_MODEL_PATH)
    print(f"--- ✅ PPO Stage Complete. Final model saved to {config.PPO_MODEL_PATH} ---")
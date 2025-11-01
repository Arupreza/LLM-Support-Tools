from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig
from trl import RewardTrainer
from tqdm import tqdm
import torch
from datasets import load_dataset

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
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            config.BASE_MODEL_ID,
            num_labels=1,  # Regression for scalar rewards
            quantization_config=quantization_config,
            device_map="auto",
            token=hf_token
        )
    except Exception as e:
        print(f"RM Model loading failed: {e}. Check token/GPU.")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- 2. Load and Prepare Preference Dataset (Keep as Text for TRL) ---
    full_dataset = load_dataset("json", data_files=config.RM_PPO_DATASET_PATH, split="train")
    dataset_splits = full_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_dataset = dataset_splits['train']
    eval_dataset = dataset_splits['test']

    # Format: Add 'prompt' column; TRL handles tokenization of chosen/rejected
    def add_prompt(example):
        example['prompt'] = f"Provide a sentiment analysis summary for the following review:\n\nReview:\n\"{example['review']}\""
        return example

    train_dataset = train_dataset.map(add_prompt)
    eval_dataset = eval_dataset.map(add_prompt)

    def compute_metrics(eval_pred):
        predictions, _ = eval_pred
        # TRL provides (chosen_rewards, rejected_rewards)
        accuracy = np.mean(predictions[0] > predictions[1])
        return {"preference_accuracy": accuracy}

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
        per_device_train_batch_size=1,  # Conservative for pairs
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # Effective batch=8
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="preference_accuracy",
        greater_is_better=True,
        save_total_limit=1,
        push_to_hub=False
    )
    
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        compute_metrics=compute_metrics,
        loss_type="sigmoid"  # Better for preference ranking
    )

    # --- 4. Train and Save Adapter ---
    trainer.train()
    trainer.save_model()
    print(f"--- ✅ Reward Modeling Stage Complete. Best model adapter saved to {config.RM_ADAPTER_PATH} ---")
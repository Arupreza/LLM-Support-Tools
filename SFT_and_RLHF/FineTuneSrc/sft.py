from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
)
from tqdm import tqdm
from peft import LoraConfig
from trl import SFTTrainer
import torch
from datasets import load_dataset

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
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            token=hf_token
        )
    except Exception as e:
        print(f"Model loading failed: {e}. Check token/GPU.")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load and Prepare Dataset ---
    full_train = load_dataset(config.SFT_DATASET_ID, split="train")
    train_dataset = full_train.select(range(int(len(full_train) * 0.01)))  # ~1% for dev (~18k)
    print(f"Training samples: {len(train_dataset)}")

    full_eval = load_dataset(config.SFT_DATASET_ID, split="test")
    eval_dataset = full_eval.select(range(int(len(full_eval) * 0.01)))  # ~1% (~2k)
    print(f"Eval samples: {len(eval_dataset)}")

    def format_review(examples):
        texts = []
        for label, content in zip(examples['label'], examples['content']):
            sentiment = "Positive" if label == 1 else "Negative"
            messages = [
                {"role": "user", "content": f"Analyze the sentiment of this Amazon product review and classify it as either Positive or Negative.\n\nReview:\n\"{content}\""},
                {"role": "assistant", "content": f"Sentiment: {sentiment}"}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        
        tokenized = tokenizer(
            texts, 
            truncation=True, 
            max_length=config.MAX_SEQ_LENGTH,
            padding="max_length"  # <-- KEY CHANGE: Uniform lengths; collator chills
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()  # Shallow copy fine now (uniform)
        tokenized["attention_mask"] = tokenized["attention_mask"].copy()  # Ensure mask too
        return tokenized

    formatted_train_dataset = train_dataset.map(
        format_review, 
        remove_columns=train_dataset.column_names,
        desc="Formatting train dataset",
        batched=True,
        batch_size=64
    )
    formatted_eval_dataset = eval_dataset.map(
        format_review, 
        remove_columns=eval_dataset.column_names,
        desc="Formatting eval dataset",
        batched=True,
        batch_size=64
    )

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
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Smaller for VRAM
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # Effective batch=16
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        logging_steps=10,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False  # Bypasses token drama
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # bf16 alignment
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_train_dataset,
        eval_dataset=formatted_eval_dataset,
        peft_config=peft_config,
        data_collator=data_collator
    )

    # --- 4. Train and Save Adapter ---
    trainer.train()
    trainer.save_model()
    print(f"--- ✅ SFT Stage Complete. Best model adapter saved to {config.SFT_ADAPTER_PATH} ---")
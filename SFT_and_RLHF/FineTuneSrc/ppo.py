from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, create_reference_model
from tqdm import tqdm
import torch
from datasets import load_dataset

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
        batch_size=4,  # Small for stability
        learning_rate=1.41e-5,
        log_with="none"
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # --- 2. Load Models ---
    # A) Policy Model: Load SFT LoRA, then add value head
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_ID,
            quantization_config=quant_config,
            device_map="auto", 
            token=hf_token
        )
        sft_model = PeftModel.from_pretrained(base_model, config.SFT_ADAPTER_PATH)
        model_with_value_head = AutoModelForCausalLMWithValueHead(sft_model)
        # Re-apply PEFT if needed (value head is on top)
        policy_model = get_peft_model(model_with_value_head, LoraConfig.from_pretrained(config.SFT_ADAPTER_PATH))
    except Exception as e:
        print(f"Policy model loading failed: {e}. Ensure SFT path exists.")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    # B) Reward Model
    rm_base_model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_MODEL_ID, 
        num_labels=1,
        quantization_config=quant_config,
        device_map="auto", 
        token=hf_token
    )
    rm_model = PeftModel.from_pretrained(rm_base_model, config.RM_ADAPTER_PATH)
    rm_model.eval()
    rm_model.config.pad_token_id = tokenizer.pad_token_id

    # Create reference model for KL penalty
    ref_model = create_reference_model(policy_model)

    # --- 3. Initialize PPOTrainer ---
    dataset = load_dataset("json", data_files=config.RM_PPO_DATASET_PATH, split="train[:5%]")  # Smaller for dev

    def format_prompt(example):
        prompt = f"Provide a sentiment analysis summary for the following review:\n\nReview:\n\"{example['review']}\""
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            tokenize=False,
            add_generation_prompt=True
        )
        example['query'] = chat_prompt
        return example  # Tokenize in collator for batching

    formatted_dataset = dataset.map(format_prompt)
    formatted_dataset.set_format(type="torch", columns=["input_ids"])  # Wait, no—PPO needs strings for query

    def collator(data):
        # Tokenize queries on-the-fly for flexibility
        queries = [d["query"] for d in data]
        input_ids = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=config.MAX_SEQ_LENGTH).input_ids
        return {"input_ids": input_ids}
    
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=formatted_dataset,
        data_collator=collator
    )

    # --- 4. PPO Training Loop ---
    generation_kwargs = {
        "min_length": -1,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
        "temperature": 0.7,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 128
    }

    output_min_length = 10  # Min response length
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), total=50):  # Limit to 50 batches
        if epoch >= 50: 
            break
        
        query_tensors = batch["input_ids"].to(ppo_trainer.accelerator.device)

        # Generate response from the policy model
        response_tensors = ppo_trainer.generate(
            query_tensors, 
            return_prompt=False, 
            output_min_length=output_min_length,
            **generation_kwargs
        )
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

        # Get reward scores from RM (batch properly)
        queries = [tokenizer.decode(q, skip_special_tokens=True) for q in query_tensors]
        full_texts = [q + r for q, r in zip(queries, batch["response"])]
        
        tokenized_rewards = tokenizer(
            full_texts, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH
        ).to(ppo_trainer.accelerator.device)
        
        with torch.no_grad():
            reward_outputs = rm_model(**tokenized_rewards)
            rewards = reward_outputs.logits.squeeze(-1).cpu().tolist()  # Flat list of scalars
        
        # PPO optimization step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    # --- 5. Save the Final Model ---
    policy_model.save_pretrained(config.PPO_MODEL_PATH)
    print(f"--- ✅ PPO Stage Complete. Final model saved to {config.PPO_MODEL_PATH} ---")
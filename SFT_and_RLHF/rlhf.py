# =====================================================================================
# SECTION 0: IMPORTS AND CONFIGURATION
# =====================================================================================
import os
from dotenv import load_dotenv
from FineTuneSrc.sft import run_sft
from FineTuneSrc.rm import run_reward_modeling
from FineTuneSrc.ppo import run_ppo

# Anti-OOM env (set at top for safety)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load environment variables for secure token management
load_dotenv()

class ScriptConfig:
    """Centralized configuration for the entire pipeline."""
    # --- Model & Tokenizer ---
    BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    MAX_SEQ_LENGTH = 1024  # Shorter for VRAM efficiency (reviews fit)

    # --- Datasets ---
    SFT_DATASET_ID = "amazon_polarity"
    RM_PPO_DATASET_PATH = "/home/lisa/Arupreza/LLM-Support-Tools/SFT_and_RLHF/RLHF_data_for_sentiment_product_review.json"

    # --- Paths for Saved Adapters ---
    SFT_ADAPTER_PATH = "./sft_llama3_adapters"
    RM_ADAPTER_PATH = "./rm_llama3_adapters"
    PPO_MODEL_PATH = "./ppo_llama3_model"

config = ScriptConfig()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face token not found. Set HF_TOKEN in your .env file.")




if __name__ == '__main__':
    # 🚀 Execute the full RLHF pipeline
    
    # Stage 1: Supervised Fine-Tuning
    run_sft()
    
    # Stage 2: Reward Modeling
    run_reward_modeling()
    
    # Stage 3: Proximal Policy Optimization
    run_ppo()

    print("\n🎉🎉🎉 RLHF Pipeline Complete! 🎉🎉🎉 Test your PPO model now!")
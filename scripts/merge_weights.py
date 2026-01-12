import os
import torch

# ==========================================
# 0. PROXY FIX (Same as training)
# ==========================================
import unsloth.models._utils
def no_op(*args, **kwargs): return
unsloth.models._utils.get_statistics = no_op
unsloth.models._utils.stats_check = no_op

from unsloth import FastLanguageModel

# ==========================================
# CONFIG
# ==========================================
BASE_MODEL = "./models/base_llama3"
LORA_ADAPTER = "./models/EcoDistill-Llama3-8B-V2_lora"
OUTPUT_DIR = "./models/EcoDistill-Llama3-8B-V2"

def merge():
    print(f"Loading Base Model: {BASE_MODEL}")
    print(f"Loading Adapter: {LORA_ADAPTER}")
    
    # 1. Load the model + adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = LORA_ADAPTER, # Loading the adapter auto-loads the base if config is right
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
        local_files_only=True,
    )
    
    # 2. Force Merge to 4-bit (for vLLM efficiency)
    print(f"Merging and saving to {OUTPUT_DIR}...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model.save_pretrained_merged(
        OUTPUT_DIR, 
        tokenizer, 
        save_method="merged_4bit_forced", # <--- THE FIX
    )
    print("✅ Merge Complete! You can now run the benchmark.")

if __name__ == "__main__":
    merge()
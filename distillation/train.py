import os
import torch

# ==========================================
# 0. PROXY FIX / MONKEY PATCHING
# ==========================================
# We must disable Unsloth's internet-dependent checks before importing the model
import unsloth.models._utils

def no_op(*args, **kwargs):
    return

# Override the functions causing the crash
unsloth.models._utils.get_statistics = no_op
unsloth.models._utils.stats_check = no_op
print("🛡️  Network checks disabled for Corporate Proxy compatibility.")

# Now we can import the rest
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ==========================================
# CONFIGURATION
# ==========================================
# POINT TO LOCAL FOLDER
MODEL_NAME = "./models/base_llama3" 
OUTPUT_DIR = "./models/EcoDistill-Llama3-8B-V2"
DATA_PATH = "data/expert_trajectories/train_dataset_cot_v2.json"

MAX_SEQ_LENGTH = 4096 
DTYPE = None 
LOAD_IN_4BIT = True 

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"""### User:
{instruction}

### Assistant:
{output}"""
        texts.append(text)
    return { "text" : texts, }

def train():
    print(f"Loading model from local path: {MODEL_NAME}...")
    
    # Added local_files_only=True for extra safety
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
        local_files_only=True, 
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none", 
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    print(f"Loading dataset from {DATA_PATH}...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True)

    print("Starting Training...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 8, # OPTIMIZATION 2: Faster data loading
        packing = True,       # OPTIMIZATION 3: Combine short examples (Huge speedup)
        args = TrainingArguments(
            per_device_train_batch_size = 16, # Increase from 2 to 16 for H100
            gradient_accumulation_steps = 1,  # Adjust to keep effective batch size reasonable
            
            warmup_steps = 10,
            num_train_epochs = 1,
            learning_rate = 2e-4,
            
            # Hardware settings
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            
            logging_steps = 5,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "checkpoints",
            max_grad_norm = 0.3,
            
            # OPTIMIZATION 5: DataLoader
            dataloader_num_workers = 4,
            dataloader_pin_memory = True,
        ),
    )

    trainer_stats = trainer.train()
    print("Training Complete!")

    # ==========================================
    # ROBUST SAVING
    # ==========================================
    print(f"Saving model to {OUTPUT_DIR}...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. First, save LoRA adapters (Safety Backup)
    # This guarantees we have the training weights even if merging fails
    print("Step 1: Saving LoRA adapters as backup...")
    model.save_pretrained(f"{OUTPUT_DIR}_lora")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}_lora")
    print(f"✅ LoRA adapters saved to {OUTPUT_DIR}_lora")

    # 2. Attempt to save the Merged Model (for vLLM)
    # We use 'merged_4bit' to match the base model quantization
    print("Step 2: Merging and saving standalone model...")
    try:
        model.save_pretrained_merged(
            OUTPUT_DIR, 
            tokenizer, 
            save_method="merged_4bit_forced",
        )
        print(f"✅ Full merged model saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f"❌ Merge failed: {e}")
        print("Don't worry! You can still use the '{OUTPUT_DIR}_lora' folder.")
        print("To use LoRA in vLLM, you load the base model and apply the adapter.")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()
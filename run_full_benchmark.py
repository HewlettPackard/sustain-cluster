import subprocess
import os
import sys
import time

# ==============================================================================
# CONFIGURATION - EDIT THESE PATHS TO MATCH YOUR SYSTEM
# ==============================================================================
# Path to the base model you downloaded (e.g., Llama-3.2 or the Unsloth base)
BASE_MODEL_PATH = "./models/base_llama3" 

# Path to your Fine-Tuned Model
FT_MODEL_PATH = "./models/EcoDistill-Llama3-8B"

# Number of episodes per agent (Higher = More statistical significance)
EPISODES = 10 

# ==============================================================================

def run_command(cmd, desc):
    print(f"\n{'='*60}")
    print(f"🚀 STARTING: {desc}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    try:
        # We use subprocess to ensure GPU memory is fully released between runs
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Experiment failed: {desc}")
        print(e)
        # We don't exit, we try to continue to the next experiment
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
        sys.exit(1)
        
    duration = time.time() - start_time
    print(f"✅ FINISHED: {desc} (Took {duration:.1f}s)")

def main():
    # 1. PRE-FLIGHT CHECKS
    if not os.path.exists("data/rag_store"):
        print("⚠️  WARNING: 'data/rag_store' not found. RAG experiments will fail.")
        print("   Run: uv run scripts/build_vector_db.py")
        input("   Press Enter to continue anyway or Ctrl+C to stop...")

    # 2. DEFINING THE EXPERIMENTS
    # List of tuples: (Description, CLI_Args_List)
    experiments = []

    # --- GROUP A: BASELINES ---
    experiments.append((
        "Baseline: Random Agent",
        ["uv", "run", "evaluate_agents.py", "--agent", "random", "--episodes", str(EPISODES)]
    ))
    experiments.append((
        "Baseline: Carbon-Greedy (Teacher 1)",
        ["uv", "run", "evaluate_agents.py", "--agent", "carbon", "--episodes", str(EPISODES)]
    ))
    experiments.append((
        "Baseline: Cost-Greedy (Teacher 2)",
        ["uv", "run", "evaluate_agents.py", "--agent", "price", "--episodes", str(EPISODES)]
    ))

    # --- GROUP B: ZERO-SHOT ABLATION ---
    if os.path.exists(BASE_MODEL_PATH):
        # 1. Base Only
        experiments.append((
            "Ablation: Llama-3 Zero-Shot (No Context)",
            ["uv", "run", "evaluate_agents.py", "--agent", "llm", 
             "--model_path", BASE_MODEL_PATH, "--episodes", str(EPISODES)]
        ))
        # 2. Base + History (Local Consistency)
        experiments.append((
            "Ablation: Llama-3 Zero-Shot + History",
            ["uv", "run", "evaluate_agents.py", "--agent", "llm", 
             "--model_path", BASE_MODEL_PATH, "--use_history", "--episodes", str(EPISODES)]
        ))
        # 3. Base + RAG (Expert Recall)
        experiments.append((
            "Ablation: Llama-3 Zero-Shot + RAG",
            ["uv", "run", "evaluate_agents.py", "--agent", "llm", 
             "--model_path", BASE_MODEL_PATH, "--use_rag", "--episodes", str(EPISODES)]
        ))
        # 4. Base + RAG + History (Full Context Baseline)
        experiments.append((
            "Ablation: Llama-3 Zero-Shot + RAG + History",
            ["uv", "run", "evaluate_agents.py", "--agent", "llm", 
             "--model_path", BASE_MODEL_PATH, "--use_rag", "--use_history", "--episodes", str(EPISODES)]
        ))

    # --- GROUP C: ECODISTILL (FINE-TUNED) ---
    if os.path.exists(FT_MODEL_PATH):
        # 1. FT Only
        experiments.append((
            "Ablation: EcoDistill (FT Only)",
            ["uv", "run", "evaluate_agents.py", "--agent", "llm", 
             "--model_path", FT_MODEL_PATH, "--episodes", str(EPISODES)]
        ))
        # 2. FT + History
        experiments.append((
            "Ablation: EcoDistill + History",
            ["uv", "run", "evaluate_agents.py", "--agent", "llm", 
             "--model_path", FT_MODEL_PATH, "--use_history", "--episodes", str(EPISODES)]
        ))
        # 3. FT + RAG
        experiments.append((
            "Ablation: EcoDistill + RAG",
            ["uv", "run", "evaluate_agents.py", "--agent", "llm", 
             "--model_path", FT_MODEL_PATH, "--use_rag", "--episodes", str(EPISODES)]
        ))
        # 4. PROPOSED METHOD
        experiments.append((
            "PROPOSED: EcoDistill Full (FT + RAG + History)",
            ["uv", "run", "evaluate_agents.py", "--agent", "llm", 
             "--model_path", FT_MODEL_PATH, "--use_rag", "--use_history", "--episodes", str(EPISODES)]
        ))
    else:
        print(f"⚠️  Skipping EcoDistill experiments (Model not found at {FT_MODEL_PATH})")
        print("   Did you run 'distillation/train.py'?")

    # 3. EXECUTION LOOP
    print(f"\n📋 Queued {len(experiments)} experiments.")
    print("This may take a while depending on your GPU...")
    time.sleep(2)

    for desc, cmd in experiments:
        run_command(cmd, desc)

if __name__ == "__main__":
    main()
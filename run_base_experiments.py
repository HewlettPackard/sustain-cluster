import subprocess
import os
import sys
import time

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Target GPU 1 to leave GPU 0 free for training
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

BASE_MODEL_PATH = "./models/base_llama3" 
EPISODES = 2

def run_command(cmd, desc):
    print(f"\n{'='*60}")
    print(f"🚀 STARTING: {desc}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    # Propagate the CUDA_VISIBLE_DEVICES env var
    env = os.environ.copy()
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Experiment failed: {desc}")
        print(e)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
        sys.exit(1)

def main():
    print(f"Running Experiments on GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'CPU')}...")

    # ==========================================
    # GROUP A: BASELINES (CPU-Heavy)
    # ==========================================
    # run_command(
    #     ["uv", "run", "evaluate_agents.py", "--agent", "random", "--episodes", str(EPISODES)],
    #     "Baseline 1: Random Agent"
    # )

    # run_command(
    #     ["uv", "run", "evaluate_agents.py", "--agent", "carbon", "--episodes", str(EPISODES)],
    #     "Baseline 2: Carbon-Greedy (Teacher)"
    # )

    # run_command(
    #     ["uv", "run", "evaluate_agents.py", "--agent", "price", "--episodes", str(EPISODES)],
    #     "Baseline 3: Cost-Greedy (Teacher)"
    # )

    # ==========================================
    # GROUP B: BASE LLM (GPU-Heavy)
    # ==========================================
    if os.path.exists(BASE_MODEL_PATH):
        # 1. Base LLM (No Context)
        run_command(
            ["uv", "run", "evaluate_agents.py", 
             "--agent", "llm", 
             "--model_path", BASE_MODEL_PATH, 
             "--episodes", str(EPISODES)],
            "Experiment 4: Base LLM (Zero-Shot)"
        )

        # 2. Base LLM + Short Memory
        run_command(
            ["uv", "run", "evaluate_agents.py", 
             "--agent", "llm", 
             "--model_path", BASE_MODEL_PATH, 
             "--use_history",
             "--episodes", str(EPISODES)],
            "Experiment 5: Base LLM + History"
        )

        # 3. Base LLM + Long Memory (RAG)
        # Note: Ensure 'data/rag_store' exists
        run_command(
            ["uv", "run", "evaluate_agents.py", 
             "--agent", "llm", 
             "--model_path", BASE_MODEL_PATH, 
             "--use_rag",
             "--episodes", str(EPISODES)],
            "Experiment 6: Base LLM + RAG"
        )
    else:
        print(f"⚠️  Skipping LLM experiments: Model not found at {BASE_MODEL_PATH}")

    print("\n✨ ALL BASE EXPERIMENTS COMPLETE.")

if __name__ == "__main__":
    main()
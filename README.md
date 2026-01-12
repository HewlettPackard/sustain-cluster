# SustainLLM-GeoDC: Sustainable Geo-Distributed Data Center Management

**IJCAI Submission Codebase**

This project introduces **SustainLLM-GeoDC**, a framework that uses Large Language Models (LLMs) to optimize workload scheduling across global data centers. It balances Carbon Intensity, Energy Cost, and Operational Performance (SLA).

Unlike "Black Box" RL agents, this system provides **explainable text-based reasoning** and utilizes **Retrieval-Augmented Generation (RAG)** to recall expert strategies from historical scenarios.

---

## 🏗️ Architecture

The pipeline consists of five stages:
1.  **Teacher Policies:** We run deterministic heuristics (Carbon-Greedy, Cost-Greedy) to generate optimal trajectories.
2.  **Chain-of-Thought (CoT) Synthesis:** We augment raw trajectories with synthetic reasoning (e.g., *"Carbon is rising, but DC_2 is still the greenest option..."*).
3.  **RAG Memory:** We index these experiences in a Vector Database (ChromaDB).
4.  **Distillation:** We Fine-Tune a Llama-3-8B model to mimic the expert's reasoning and action format.
5.  **Inference:** The final agent uses **vLLM** for fast inference, consulting both its RAG memory and Short-Term History.

---

## 📂 Repository Structure

### New Modules (`src/`)
*   **`src/controllers/`**: The brains of the agents.
    *   `llm_controller.py`: The main agent. Handles vLLM inference, RAG lookup, and history buffers.
    *   `heuristics.py`: Baseline rule-based agents (Random, Carbon-Greedy, Cost-Greedy).
*   **`src/distillation/`**: Training logic.
    *   `train.py`: Uses **Unsloth** (QLoRA) to fine-tune Llama-3 on the generated dataset.
*   **`src/rag/`**: Memory systems.
    *   `engine.py`: Handles ChromaDB insertion and retrieval using `sentence-transformers`.
*   **`src/data_processing/`**:
    *   `augment_dataset.py`: Converts raw state/action pairs into "Instruction/Response" CoT formats.
*   **`src/utils/`**:
    *   `llm_serialization.py`: Converts numerical simulation states into Natural Language Prompts (adds Trend analysis and Constraint checks).

### Utility Scripts (`scripts/`)
*   **`run_full_benchmark.py`**: **MASTER SCRIPT.** Runs all baselines, ablations, and the proposed method automatically.
*   **`run_base_experiments.py`**: Runs only Baselines and Zero-Shot LLM (useful for parallel execution).
*   **`build_vector_db.py`**: Indexes the JSON dataset into the Vector DB.
*   **`download_*.py`**: Helper scripts to download HF models on restricted corporate networks.
*   **`merge_weights.py`**: Merges LoRA adapters into the base model for vLLM compatibility.
*   **`plot_results.py`**: Generates the Pareto Frontier and Ablation charts for the paper.

---

## 🛠️ Installation

**Prerequisites:**
*   Linux Server with NVIDIA GPUs (H100 recommended for Unsloth training).
*   Python 3.11.
*   `uv` package manager.

**Setup:**
This project relies on specific versions of `numpy` and `vllm`. Use `uv sync` to install exactly what is defined in `pyproject.toml`.

```bash
# 1. Install dependencies
uv sync

# 2. Activate environment
source .venv/bin/activate
```

---

## 🚀 Reproduction Pipeline (Step-by-Step)

Follow this order to recreate the IJCAI experiments.

### Phase 1: Setup & Downloads
Since the cluster has restricted internet, we download models locally first.

```bash
# Download Embedding Model (for RAG)
uv run scripts/download_embeddings.py

# Download Base Llama-3 Model (for Training)
uv run scripts/download_base_model.py
```

### Phase 2: Data Generation
Generate the "Teacher" dataset (Simulation) and the "Reasoning" dataset (Text).

```bash
# 1. Run Simulation (Simulates 10 months of data)
uv run generate_expert_data.py
# Output: data/expert_trajectories/heuristic_dataset_v2.json

# 2. Augment with Chain-of-Thought
uv run src/data_processing/augment_dataset.py
# Output: data/expert_trajectories/train_dataset_cot_v2.json
```

### Phase 3: Build RAG Memory
Index the expert experiences into the Vector Database.

```bash
# Clear old DB first to be safe
rm -rf data/rag_store

# Build new DB
uv run scripts/build_vector_db.py
```

### Phase 4: Model Training (Distillation)
Fine-tune Llama-3 on the CoT dataset using GPU 0.

```bash
# 1. Train Adapters (Takes ~5-10 mins on H100)
uv run src/distillation/train.py

# 2. Merge Adapters for Inference
uv run scripts/merge_weights.py
# Output: models/EcoDistill-Llama3-8B-V2 (Ready for vLLM)
```

### Phase 5: Evaluation
Run the full benchmark suite. This runs:
1.  Baselines (Random, Carbon, Price).
2.  Ablations (Zero-Shot, FT-Only, FT+RAG).
3.  **Proposed Method (FT + RAG + History)**.

```bash
uv run run_full_benchmark.py
```

### Phase 6: Visualization
Generate the charts for the paper.

```bash
uv run scripts/plot_results.py
```
Check `results/plots/` for:
*   `1_tradeoff_pareto.png`: Carbon vs. Cost Frontier.
*   `2_ablation_study.png`: Contribution of RAG/FT.

---

## 🐛 Debugging

*   **Logs:** Check `data/logs/` for "Flight Recorder" logs (`llm_debug_*.txt`). These show exactly what the LLM read (Prompt) and wrote (Response) at every timestep.
*   **Proxy Issues:** If `unsloth` or `huggingface` fail, ensure you are using the `scripts/download_*.py` tools and that `src/distillation/train.py` has the network patch enabled.
```
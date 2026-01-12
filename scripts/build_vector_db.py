import json
import sys
import os
import time
from tqdm import tqdm

# Add project root to python path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.engine import ActionRetriever

# OPTIMIZATION: Process 256 items at a time
BATCH_SIZE = 256

def populate_memory():
    INPUT_FILE = "data/expert_trajectories/heuristic_dataset_v2.json"
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # Initialize RAG Engine
    print("Initializing RAG Engine...")
    rag = ActionRetriever()
    
    print(f"Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
        
    total_records = len(data)
    print(f"Indexing {total_records} memories with Batch Size {BATCH_SIZE}...")
    
    start_time = time.time()
    
    # --- BATCH PROCESSING LOOP ---
    for i in tqdm(range(0, total_records, BATCH_SIZE)):
        # 1. Slice the batch
        batch_data = data[i : i + BATCH_SIZE]
        
        # 2. Separate fields into lists
        prompts = [item['prompt'] for item in batch_data]
        actions = [item['chosen_action'] for item in batch_data]
        strategies = [item['strategy'] for item in batch_data]
        step_ids = [item['global_step'] for item in batch_data]
        
        # 3. Send to RAG Engine
        rag.add_batch(prompts, actions, strategies, step_ids)

    duration = time.time() - start_time
    print(f"✅ Vector Database built in {duration:.2f} seconds.")
    print(f"Count in DB: {rag.collection.count()}")

    # --- TEST RETRIEVAL ---
    print("\n--- TEST RETRIEVAL ---")
    test_prompt = data[0]['prompt']
    print("Querying with first sample...")
    context = rag.retrieve_context(test_prompt, k=2)
    print(context)

if __name__ == "__main__":
    populate_memory()
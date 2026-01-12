import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import os

class ActionRetriever:
    def __init__(self, db_path="data/rag_store", collection_name="expert_memories"):
        """
        Initializes the RAG engine using ChromaDB and a local embedding model.
        """
        os.makedirs(db_path, exist_ok=True)
        
        # 1. Initialize Vector DB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 2. Initialize Embedding Model (FROM LOCAL PATH)
        # Check if the local model exists
        local_model_path = os.path.join(os.getcwd(), "models", "embedding_model")
        
        if os.path.exists(local_model_path):
            print(f"Loading embedding model from local: {local_model_path}")
            self.embedding_model = SentenceTransformer(local_model_path)
        else:
            print("Warning: Local embedding model not found. Attempting online download (may fail on proxy)...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 3. Get or Create Collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"RAG Engine initialized at {db_path}. Collection: {collection_name}")

    def add_trajectory(self, prompt, action, strategy, step_id):
        # Generate embedding
        embedding = self.embedding_model.encode(prompt, show_progress_bar=False).tolist()
        
        metadata = {
            "action": action,
            "strategy": strategy,
            "step_id": step_id
        }
        
        self.collection.add(
            documents=[prompt],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())]
        )

    # --- NEW METHOD FOR BULK LOADING ---
    def add_batch(self, prompts, actions, strategies, step_ids):
        """
        Adds multiple trajectories at once.
        """
        # 1. Batch Encode (Much faster on GPU)
        embeddings = self.embedding_model.encode(prompts, batch_size=128, show_progress_bar=False).tolist()
        
        # 2. Prepare Metadata Lists
        metadatas = []
        ids = []
        
        for act, strat, sid in zip(actions, strategies, step_ids):
            metadatas.append({
                "action": act, 
                "strategy": strat, 
                "step_id": sid
            })
            ids.append(str(uuid.uuid4()))
            
        # 3. Batch Insert to DB (Reduces I/O overhead)
        self.collection.add(
            documents=prompts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
    def retrieve_context(self, current_prompt, k=3):
        current_embedding = self.embedding_model.encode(current_prompt, show_progress_bar=False).tolist()
        
        results = self.collection.query(
            query_embeddings=[current_embedding],
            n_results=k
        )
        
        context_str = "Historical Context (Similar Past Situations):\n"
        
        if not results['documents']:
            return context_str + "No historical data available."

        for i in range(k):
            try:
                # Safety check for empty results
                if len(results['metadatas'][0]) <= i: break
                
                meta = results['metadatas'][0][i]
                context_str += (
                    f"{i+1}. Under similar conditions, the expert strategy '{meta['strategy']}' "
                    f"chose Action {meta['action']} (DC_{meta['action']}).\n"
                )
            except Exception:
                continue
            
        return context_str
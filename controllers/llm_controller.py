from collections import deque  # <--- NEW IMPORT
from vllm import LLM, SamplingParams
import re
import traceback
import numpy as np
from controllers.base import BaseController
from utils.llm_serialization import EnvSerializer
from rag.engine import ActionRetriever
from logger import recorder  # <--- IMPORT LOGGER

# --- SYSTEM PROMPT DEFINITION ---
SYSTEM_PROMPT = """You are an expert AI Data Center Scheduler. 
Your goal is to assign the incoming workload batch to the optimal Data Center (DC).
You must balance three conflicting objectives:
1. Sustainability: Minimize Carbon Emissions (g/kWh).
2. Cost: Minimize Energy Price ($/kWh).
3. Performance: Minimize Load/Queuing delay.

You will receive:
1. Historical Context (Retrieval from long-term memory).
2. Recent History (Your last few decisions).
3. Current Telemetry (The present state).

Analyze the trade-offs and output your decision strictly in this format:
"Analysis: [Your reasoning here]
Action: [DC_ID]"
"""

class LLMController(BaseController):
    def __init__(self, cluster_manager, 
                 model_path=None, 
                 use_rag=False, 
                 use_few_shot=False,
                 use_history=False, # <--- NEW FLAG
                 history_window=3,  # <--- NEW PARAM (Last 3 steps)
                 gpu_memory_utilization=0.9):
        
        super().__init__(cluster_manager)
        self.serializer = EnvSerializer(cluster_manager)
        self.num_dcs = len(cluster_manager.datacenters)
        
        # ABLATION FLAGS
        self.use_rag = use_rag
        self.use_few_shot = use_few_shot
        self.use_history = use_history
        self.history_window = history_window
        self.model_path = model_path
        
        # WORKING MEMORY (Stores last N (Observation, Action) tuples)
        self.history_buffer = deque(maxlen=self.history_window)
        
        # Name construction for logs
        modes = []
        if use_rag: modes.append("RAG")
        if use_few_shot: modes.append("FewShot")
        if use_history: modes.append(f"Hist{history_window}")
        self.name = f"LLM-{'-'.join(modes)}" if modes else "LLM-ZeroShot"

        # --- 1. INITIALIZE RAG ---
        if self.use_rag:
            print(f"[{self.name}] Initializing RAG Engine...")
            self.rag_engine = ActionRetriever() 

        # --- 2. INITIALIZE vLLM ---
        if self.model_path and self.model_path.lower() != "mock":
            print(f"[{self.name}] Loading vLLM Model from: {self.model_path}...")
            # Initialize vLLM
            self.llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=1, # Increase if using multiple GPUs
                # max_model_len=4096 # Uncomment if you hit context limits
            )
            self.tokenizer = self.llm.get_tokenizer()
            self.sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
        else:
            print(f"[{self.name}] WARNING: No model_path provided (or 'mock'). Running in DUMMY mode.")
            self.llm = None

    def _construct_few_shot_prompt(self):
        """Standard few-shot examples to guide the model format."""
        return """
        ### EXAMPLES ###
        Situation: DC_0 has high carbon (600g). DC_1 has low carbon (150g) but high price.
        Analysis: Priority is sustainability. DC_1 saves 450g/kWh.
        Action: 1

        Situation: DC_0 has low price ($0.05). DC_1 is expensive ($0.30). Carbon is similar.
        Analysis: Cost optimization is feasible here. DC_0 is 6x cheaper.
        Action: 0
        """

    def get_action(self, env, observation):
        step_id = env.global_step # Get current step for logging

        # 1. Serialize State (Numeric -> Text)
        state_text = self.serializer.serialize_state(env, env.current_tasks)
        
        # Handle empty steps (no tasks) -> Default action 0 (doesn't matter)
        if not state_text: 
            recorder.log(step_id, "WARNING", "Empty state text (no tasks). Defaulting to 0.")
            return 0, {} 

        # 2. Build Contexts
        context_str = ""
        
        # A. RAG (Long-Term)
        rag_context = ""
        if self.use_rag:
            rag_context = self.rag_engine.retrieve_context(state_text, k=3)
            context_str += f"\n{rag_context}\n"
            recorder.log(step_id, "RAG_RETRIEVAL", rag_context) # LOG RAG

        # B. HISTORY (Short-Term Working Memory)
        if self.use_history and len(self.history_buffer) > 0:
            history_str = "\n### SHORT-TERM MEMORY (Your Recent Actions) ###\n"
            for i, (past_time, past_act) in enumerate(self.history_buffer):
                history_str += f"Step T-{len(self.history_buffer)-i}: At {past_time}, you chose Action {past_act}.\n"
            
            # Add a hint about stability
            history_str += "(Consider stability: avoid unnecessary rapid switching unless conditions changed significantly.)\n"
            context_str += history_str
            recorder.log(step_id, "HISTORY_CONTEXT", history_str) # LOG HISTORY

        # C. Add Few-Shot
        if self.use_few_shot:
            context_str += self._construct_few_shot_prompt()

        # 3. Construct Final Prompt (Chat Format)
        # We put the state_text LAST so it's fresh in memory
        user_content = f"""
        {context_str}

        ### CURRENT TELEMETRY ###
        {state_text}

        Instruction: Select the optimal Data Center ID (0 to {self.num_dcs-1}).
        """
        recorder.log(step_id, "FULL_PROMPT", user_content)

        response_text = ""

        # 4. Inference
        if self.llm:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            
            # Formatting
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Generate
            outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
            response_text = outputs[0].outputs[0].text.strip()
            
            recorder.log(step_id, "LLM_RAW_OUTPUT", response_text)

            # Parse
            action = self._parse_action(response_text, step_id)
        else:
            # Mock mode for debugging pipeline without GPU
            response_text = "MOCK_RESPONSE: Action: 0"
            action = 0
            recorder.log(step_id, "MOCK_MODE", "Returning Action 0")

            # print(f"Mock Prompt:\n{user_content}")
            
        # 5. UPDATE HISTORY BUFFER
        # We store the time (from state_text or env) and the chosen action
        # Extract time string from state_text for brevity (first line)
        time_ref = state_text.split('\n')[0].replace("Current Time: ", "")
        self.history_buffer.append((time_ref, action))
        
        return action, {
            "prompt_len": len(user_content),
            "response": response_text,
            "rag_retrieval": rag_context if self.use_rag else "N/A"
        }

    def _parse_action(self, text, step_id):
        """
        Robustly extracts the action integer from LLM response.
        """
        try:
            # 1. Look for explicit "Action: X" pattern
            match = re.search(r'Action:\s*(\d+)', text, re.IGNORECASE)
            if match:
                act = int(match.group(1))
                if 0 <= act < self.num_dcs:
                    recorder.log(step_id, "PARSED_ACTION", f"Found explicit 'Action: {act}'")
                    return act
            
            # 2. Fallback: Look for the LAST digit in the text
            # (Often models chatter and put the number at the end)
            digits = re.findall(r'\d+', text)
            if digits:
                act = int(digits[-1])
                if 0 <= act < self.num_dcs:
                    recorder.log(step_id, "PARSED_ACTION_FALLBACK", f"Found last digit: {act}")
                    return act
            
            recorder.log(step_id, "PARSE_ERROR", f"Could not find valid ID (0-{self.num_dcs-1}) in response.")
            return 0 # Conservative fallback (DC_0)

        except Exception:
            recorder.log(step_id, "PARSE_EXCEPTION", traceback.format_exc())
            return 0
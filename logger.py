# src/logger.py
import os
import time
import datetime

class FlightRecorder:
    def __init__(self, filepath="data/logs/debug_log.txt"):
        self.filepath = filepath
        # Create directory if not exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Clear previous log on init (or create new one based on timestamp)
        with open(self.filepath, "w") as f:
            f.write(f"=== FLIGHT RECORDER STARTED AT {time.ctime()} ===\n")

    def log(self, step, event_type, content):
        """
        step: int (timestep)
        event_type: str (e.g., "RETRIEVAL", "PROMPT", "ACTION")
        content: str (The details)
        """
        entry = f"\n[{time.strftime('%H:%M:%S')}][Step {step}] === {event_type} ===\n{content}\n" + "-"*50
        with open(self.filepath, "a") as f:
            f.write(entry)

# Singleton instance
curr_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Changed path to data/logs/ to keep root clean
recorder = FlightRecorder(filepath=f"data/logs/llm_debug_{curr_datetime}.txt")
import requests
import urllib3
import os
from huggingface_hub import snapshot_download, configure_http_backend

# ==========================================================
# 1. SSL BYPASS FOR CORPORATE NETWORK
# ==========================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

# ==========================================================
# 2. DOWNLOAD CONFIGURATION
# ==========================================================
# REPLACE THIS WITH YOUR ACTUAL TOKEN
# Read token from a file (add .hf_token to your .gitignore)
token_file = os.path.join(os.path.dirname(__file__), "..", ".hf_token")

with open(token_file, "r") as f:
    MY_HF_TOKEN = f.read().strip()

model_id = "meta-llama/Llama-3.2-3B-Instruct"
local_dir = f"./models/{model_id.replace('/', '_')}"

print(f"Starting download for {model_id}...")

try:
    snapshot_download(
        repo_id=model_id, 
        local_dir=local_dir,
        token=MY_HF_TOKEN,          # <--- Critical for Gated Models
        local_dir_use_symlinks=False # Better for cluster/lustre storage
    )
    print(f"✅ Success! Model saved to {local_dir}")
except Exception as e:
    print(f"❌ Error: {e}")


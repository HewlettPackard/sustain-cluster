import requests
import urllib3
import os
from huggingface_hub import snapshot_download, configure_http_backend

# 1. SSL BYPASS
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

# 2. DOWNLOAD CONFIG
# This is the 4-bit model optimized for H100
model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit" 
local_dir = "./models/base_llama3"

print(f"Downloading {model_id} to {local_dir}...")

try:
    snapshot_download(
        repo_id=model_id, 
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print("✅ Download Complete.")
except Exception as e:
    print(f"❌ Error: {e}")
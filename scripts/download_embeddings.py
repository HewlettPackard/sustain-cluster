import requests
import urllib3
import os
from huggingface_hub import snapshot_download, configure_http_backend

# ==========================================================
# 1. SSL BYPASS SETUP
# ==========================================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

# ==========================================================
# 2. DOWNLOAD SETTINGS
# ==========================================================
model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"
# We save it to a specific folder in your project
local_dir = os.path.join(os.getcwd(), "models", "base_llama3")

print(f"Starting download for {model_id}...")
print(f"Target directory: {local_dir}")

try:
    snapshot_download(
        repo_id=model_id, 
        local_dir=local_dir,
        # Token is optional for this public model, but used if found
        token=None,          
        local_dir_use_symlinks=False
    )
    print(f"✅ Success! Model saved to {local_dir}")
except Exception as e:
    print(f"❌ Error: {e}")
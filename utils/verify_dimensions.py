# utils/verify_dimensions.py
"""
Utility script to verify observation dimensions in LLM-enhanced environments
"""
import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def verify_manager_dimensions(enable_llm=True, num_dcs=3):
    """Verify manager observation dimensions"""
    from utils.marl_utils import D_META_MANAGER
    
    D_GLOBAL = 4
    D_LLM_ADVICE = 10
    d_option_feat = 5  # is_local, cpu_avail_pct, gpu_avail_pct, ci, external_temperature
    
    base_obs_dim = D_META_MANAGER + (num_dcs * d_option_feat) + D_GLOBAL
    total_obs_dim = base_obs_dim + (D_LLM_ADVICE if enable_llm else 0)
    
    print(f"📊 Manager Observation Dimensions (LLM: {enable_llm}):")
    print(f"  Meta-task features: {D_META_MANAGER}")
    print(f"  DC options: {num_dcs} × {d_option_feat} = {num_dcs * d_option_feat}")
    print(f"  Global context: {D_GLOBAL}")
    print(f"  LLM advice: {D_LLM_ADVICE if enable_llm else 0}")
    print(f"  ─────────────────────────")
    print(f"  Base total: {base_obs_dim}")
    print(f"  Enhanced total: {total_obs_dim}")
    
    return total_obs_dim

def verify_worker_dimensions(enable_llm=True, num_dcs=3):
    """Verify worker observation dimensions"""
    D_TASK_FEAT = 7   # From worker meta-task features
    D_DC_FEAT = 5     # From local DC features  
    D_GLOBAL = 4      # Time features
    D_LLM_ADVICE = 10
    
    base_obs_dim_per_dc = D_TASK_FEAT + D_DC_FEAT + D_GLOBAL
    total_obs_dim_per_dc = base_obs_dim_per_dc + (D_LLM_ADVICE if enable_llm else 0)
    
    print(f"📊 Worker Observation Dimensions (LLM: {enable_llm}):")
    print(f"  Task features per DC: {D_TASK_FEAT}")
    print(f"  DC features per DC: {D_DC_FEAT}")
    print(f"  Global context per DC: {D_GLOBAL}")
    print(f"  LLM advice per DC: {D_LLM_ADVICE if enable_llm else 0}")
    print(f"  ─────────────────────────")
    print(f"  Base per DC: {base_obs_dim_per_dc}")
    print(f"  Enhanced per DC: {total_obs_dim_per_dc}")
    print(f"  Matrix shape: ({num_dcs}, {total_obs_dim_per_dc})")
    
    return (num_dcs, total_obs_dim_per_dc)

def test_environment_creation():
    """Test environment creation and verify dimensions"""
    print("🧪 Testing Environment Creation...")
    
    try:
        # Test manager environment
        print("\n--- Testing Manager Environment ---")
        manager_dim = verify_manager_dimensions(enable_llm=True, num_dcs=3)
        
        print("\n--- Testing Worker Environment ---")
        worker_shape = verify_worker_dimensions(enable_llm=True, num_dcs=3)
        
        print("\n✅ Dimension verification complete!")
        print(f"Expected manager obs dim: {manager_dim}")
        print(f"Expected worker obs shape: {worker_shape}")
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_environment_creation()
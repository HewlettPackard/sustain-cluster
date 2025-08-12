# SYSTEMATIC PYTHON ENVIRONMENT DIAGNOSIS

# 1. CHECK PYTHON AND CONDA ENVIRONMENT
echo "🐍 Python Environment Check"
echo "=========================="
which python3
python3 --version
which conda
conda --version
echo "Current conda env: $CONDA_DEFAULT_ENV"

# 2. CHECK PANDAS AND PICKLE SPECIFICALLY
echo -e "\n📦 Package Versions"
echo "==================="
python3 -c "
import sys
print(f'Python: {sys.version}')

try:
    import pandas as pd
    print(f'Pandas: {pd.__version__}')
except Exception as e:
    print(f'Pandas ERROR: {e}')

try:
    import pickle
    print(f'Pickle: available')
    print(f'Pickle protocol: {pickle.HIGHEST_PROTOCOL}')
except Exception as e:
    print(f'Pickle ERROR: {e}')

try:
    import numpy as np
    print(f'NumPy: {np.__version__}')
except Exception as e:
    print(f'NumPy ERROR: {e}')
"

# 3. TEST THE SPECIFIC FILE
echo -e "\n📁 File System Check"
echo "==================="
FILE_PATH="data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl"

if [ -f "$FILE_PATH" ]; then
    echo "✅ File exists: $FILE_PATH"
    echo "📊 File size: $(du -h $FILE_PATH | cut -f1)"
    echo "🔐 Permissions: $(ls -la $FILE_PATH)"
    
    # Check if file is currently being accessed
    if command -v lsof &> /dev/null; then
        echo "🔍 Processes accessing file:"
        lsof "$FILE_PATH" 2>/dev/null || echo "No processes currently accessing file"
    fi
else
    echo "❌ File does not exist: $FILE_PATH"
fi

# 4. TEST PICKLE LOADING SPECIFICALLY
echo -e "\n🧪 Pickle Loading Test"
echo "======================"
python3 -c "
import os
import pandas as pd
import traceback

file_path = 'data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl'

print(f'Testing pickle loading for: {file_path}')

if not os.path.exists(file_path):
    print('❌ File does not exist!')
    exit(1)

file_size = os.path.getsize(file_path)
print(f'📊 File size: {file_size:,} bytes ({file_size/(1024**2):.1f} MB)')

if file_size == 0:
    print('❌ File is empty!')
    exit(1)

try:
    print('🔄 Attempting pd.read_pickle()...')
    df = pd.read_pickle(file_path)
    print(f'✅ SUCCESS! Shape: {df.shape}')
    print(f'📋 Columns: {list(df.columns)[:5]}...')  # First 5 columns
    print(f'📅 Memory usage: {df.memory_usage(deep=True).sum()/(1024**2):.1f} MB')
except Exception as e:
    print(f'❌ FAILED: {e}')
    print(f'Error type: {type(e).__name__}')
    traceback.print_exc()
    
    # Try alternative loading methods
    print('\n🔄 Trying alternative loading...')
    try:
        import pickle
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        print(f'✅ Alternative pickle.load() worked! Shape: {df.shape}')
    except Exception as e2:
        print(f'❌ Alternative loading also failed: {e2}')
"

# 5. CHECK CONDA ENVIRONMENT INTEGRITY
echo -e "\n🔧 Conda Environment Check"
echo "=========================="
conda list pandas
conda list numpy
conda list pickle-mixin 2>/dev/null || echo "pickle-mixin not installed (this is normal)"

# Check for environment corruption
echo -e "\n🩺 Environment Health Check"
echo "==========================="
python3 -c "
import sys
import importlib

# Test core modules
modules_to_test = ['pandas', 'numpy', 'pickle', 'multiprocessing', 'os', 'sys']

for module in modules_to_test:
    try:
        importlib.import_module(module)
        print(f'✅ {module}: OK')
    except Exception as e:
        print(f'❌ {module}: {e}')

# Test multiprocessing specifically (since you're using parallel training)
try:
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    print('✅ Multiprocessing spawn: OK')
except Exception as e:
    print(f'❌ Multiprocessing spawn: {e}')
"

# 6. CHECK FOR LUSTRE FILE SYSTEM ISSUES
echo -e "\n💾 Lustre File System Check"
echo "==========================="
df -h /lustre/ghorbanp/ 2>/dev/null || echo "Not on Lustre or df failed"
lfs df -h /lustre/ghorbanp/ 2>/dev/null || echo "lfs command not available"

echo -e "\n🏁 Diagnosis Complete"
echo "==================="
echo "Run this script and share the output to diagnose the issue!"
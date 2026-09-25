import json

with open('kaggle_execution_notebook.ipynb', 'r') as f:
    nb = json.load(f)

# Find the GPU code cell and replace it
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('lightgbm' in line and 'install-option' in line for line in cell['source']):
        cell['source'] = [
            "# Uncomment to compile LightGBM with GPU support\n",
            "# !pip uninstall -y lightgbm\n",
            "# !CMAKE_ARGS=\"-DUSE_GPU=1 -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so\" pip install lightgbm --no-binary lightgbm"
        ]

with open('kaggle_execution_notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

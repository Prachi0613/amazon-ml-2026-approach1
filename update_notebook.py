import json
import os

with open('kaggle_execution_notebook.ipynb', 'r') as f:
    nb = json.load(f)

# Modify cell 2 (index 2) to pull changes
nb['cells'][2]['source'] = [
    "import os\n",
    "import sys\n",
    "\n",
    "repo_url = \"https://github.com/Prachi0613/amazon-ml-2026-approach1.git\"\n",
    "repo_dir = \"/kaggle/working/amazon-ml-2026-approach1\"\n",
    "\n",
    "# Ensure we are in /kaggle/working before cloning\n",
    "if os.path.exists('/kaggle/working'):\n",
    "    os.chdir('/kaggle/working')\n",
    "\n",
    "if not os.path.exists(repo_dir):\n",
    "    print(f\"Cloning repository {repo_url}...\")\n",
    "    !git clone {repo_url} {repo_dir}\n",
    "else:\n",
    "    print(f\"Repository already exists at {repo_dir}, skipping clone.\")\n",
    "\n",
    "os.chdir(repo_dir)\n",
    "print(f\"Changed working directory to {os.getcwd()}\")\n",
    "\n",
    "# Pull the latest changes to ensure GPU support updates are included\n",
    "!git pull origin main\n",
    "\n",
    "if repo_dir not in sys.path:\n",
    "    sys.path.insert(0, repo_dir)\n",
    "    print(f\"Added {repo_dir} to Python path.\")"
]

# Add a markdown and code cell for GPU compilation if needed (insert before the execution block)
gpu_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## (Optional) GPU Compilation for LightGBM\n",
        "If the pipeline logs `Reason: GPU backend unavailable` and falls back to CPU, it means Kaggle's LightGBM lacks compiled GPU bindings.\n",
        "Run the cell below BEFORE the pipeline to compile LightGBM with OpenCL/CUDA support.\n",
        "Otherwise, leave it commented out."
    ]
}

gpu_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Uncomment to compile LightGBM with GPU support\n",
        "# !pip install lightgbm --install-option=--gpu --install-option=\"--opencl-include-dir=/usr/local/cuda/include/\" --install-option=\"--opencl-library=/usr/local/cuda/lib64/libOpenCL.so\""
    ]
}

nb['cells'].insert(-2, gpu_md)
nb['cells'].insert(-2, gpu_code)

with open('kaggle_execution_notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

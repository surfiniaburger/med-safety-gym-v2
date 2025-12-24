# Comparative Analysis: Tutorial 1 Modernization
# This file shows the proposed changes to 'docs/tutorials/tutorial_1.ipynb'
# transitioning from manual scripts to the 'openenv-dipg-safety' package.

# ==============================================================================
# 1. INSTALLATION
# ==============================================================================

# --- BEFORE (Manual Clone & Install) ---
"""
%%capture
if \"COLAB_\" not in \"\".join(os.environ.keys()):
    !pip install unsloth
else:
    !pip install uv
    !uv pip install --system git+https://github.com/meta-pytorch/OpenEnv.git
    !git clone https://github.com/surfiniaburger/med-safety-gym.git
    !cd med-safety-gym && uv pip install --system .
"""

# --- AFTER (Standardized PyPI) ---
"""
!pip install openenv-dipg-safety
"""


# ==============================================================================
# 2. SERVER STARTUP
# ==============================================================================

# --- BEFORE (Background Subprocess) ---
"""
import subprocess
import os

with open(\"server_log.txt\", \"w\") as log_file:
    env = os.environ.copy()
    env[\"PYTHONPATH\"] = f\"{os.getcwd()}/med-safety-gym\"
    server_process = subprocess.Popen(
        [sys.executable, \"-m\", \"server.app\"],
        cwd=\"med-safety-gym\",
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env
    )
"""

# --- AFTER (Official CLI) ---
"""
# Start the DIPG Safety Gym server using the installed package CLI
!dipg-server --dataset_path ./med-safety-gym/dataset.jsonl &
"""


# ==============================================================================
# 3. EVALUATION LOGIC
# ==============================================================================

# --- BEFORE (Raw Requests) ---
"""
import requests

def evaluate_model(model, tokenizer, samples=50):
    response = requests.get(\"http://localhost:8000/tasks\", params={\"count\": samples})
    tasks = response.json()[\"tasks\"]
    
    # ... generation loop ...
    
    eval_res = requests.post(\"http://localhost:8000/evaluate/tasks\", json={\"responses\": responses})
    metrics = eval_res.json()[\"metrics\"]
"""

# --- AFTER (Standardized Client) ---
"""
from med_safety_gym.client import DIPGSafetyEnv

def evaluate_model(model, tokenizer, samples=50):
    env = DIPGSafetyEnv(\"http://localhost:8000\")
    tasks = env.get_eval_tasks(max_samples=samples)
    
    # ... generation loop (same as before) ...
    
    results = env.evaluate_model(responses)
    metrics = results[\"metrics\"]
"""

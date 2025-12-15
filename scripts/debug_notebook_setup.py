
import os
import sys
import subprocess
import shutil

def run_command(command, cwd=None, env=None):
    """Run a shell command and print output."""
    print(f"🔹 Running: {command}")
    try:
        # We use check=True to raise error on failure
        subprocess.run(
            command, 
            shell=True, 
            check=True, 
            cwd=cwd, 
            env=env,
            stdout=sys.stdout, # Print directly
            stderr=sys.stderr
        )
        print("✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with code {e.returncode}")
        return False

def debug_setup():
    print("="*60)
    print("🛠️  REPRODUCING SETUP_SERVER.PY LOGIC")
    print("="*60)

    # 1. Setup Workspace
    work_dir = "debug_repro_env"
    if os.path.exists(work_dir):
        print(f"🧹 Clearing {work_dir}...")
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    os.chdir(work_dir)

    # 2. Clone Med Safety Gym (same as setup_server.py)
    print("\n--- Step 1: Clone Repo ---")
    if not run_command("git clone https://github.com/surfiniaburger/med-safety-gym.git"):
        return

    repo_path = os.path.abspath("med-safety-gym")

    # 3. Install Dependencies (The critical step)
    # setup_server.py uses: subprocess.run([sys.executable, "-m", "pip", "install", "-qqq", "-e", "."], ...)
    # We remove -qqq to see errors.
    print("\n--- Step 2: Install Dependencies (pip install -e .) ---")
    if not run_command(f"{sys.executable} -m pip install -e .", cwd=repo_path):
        print("❌ Installation failed completely.")
        return

    # 4. Verify OpenEnv Core Import
    print("\n--- Step 3: Verify Import ---")
    try:
        # We need to add the repo to path because of -e .
        sys.path.insert(0, repo_path)
        import openenv_core
        print(f"✅ openenv_core FOUND! Path: {openenv_core.__file__}")
    except ImportError as e:
        print(f"❌ ImportError: {e}")
        print("   This means openenv-core was NOT installed by 'pip install -e .'")
        print("   Checking what IS installed...")
        run_command(f"{sys.executable} -m pip list | grep openenv")

    # 5. Cleanup
    os.chdir("..")
    # shutil.rmtree(work_dir) # Keep it for inspection if needed

if __name__ == "__main__":
    debug_setup()

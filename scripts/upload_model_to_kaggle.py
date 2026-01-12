import os
import kagglehub
import logging
import sys

# --- 0. Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- 1. Configuration ---
# IMPORTANT: Please change '[YOUR-KAGGLE-USERNAME]' to your actual Kaggle username.
KAGGLE_USERNAME = "[YOUR-KAGGLE-USERNAME]"

# We will construct the model handle based on the competition's best practices.
MODEL_NAME = "gemma-3-1b-tunix-grpo"
FRAMEWORK = "jax"
VARIATION = "dipg-safety-900steps"

# The final 4-part handle for the model
KAGGLE_MODEL_HANDLE = f"{KAGGLE_USERNAME}/{MODEL_NAME}/{FRAMEWORK}/{VARIATION}"

# This is the directory where the final model was saved by the training script.
LOCAL_MODEL_DIR = "/kaggle/working/outputs_grpo/checkpoints/manual_final"

# A version description for your model upload.
VERSION_NOTES = "GRPO 900-step model with soft-penalty recovery. Best performing model from the training curriculum."

# --- 2. Verification ---
logger.info("="*50)
logger.info("🚀 STARTING KAGGLE MODEL UPLOAD SCRIPT (using kagglehub.model_upload)")
logger.info("="*50)

if KAGGLE_USERNAME == "[YOUR-KAGGLE-USERNAME]":
    logger.error("❌ Please update the 'KAGGLE_USERNAME' variable in this script before running!")
    sys.exit(1)

logger.info(f"Verifying model checkpoint path exists: {LOCAL_MODEL_DIR}")
if not os.path.exists(LOCAL_MODEL_DIR):
    logger.error(f"❌ Model checkpoint not found at '{LOCAL_MODEL_DIR}'!")
    logger.error("   Please ensure the training script ran successfully and saved the model to the correct directory.")
    sys.exit(1)
else:
    logger.info("✓ Model checkpoint found.")

# --- 3. Push New Model Version ---
logger.info(f"🔗 Target Model Handle: {KAGGLE_MODEL_HANDLE}")
logger.info(f"📤 Uploading model from path: {LOCAL_MODEL_DIR}")
logger.info(f"   Version notes: '{VERSION_NOTES}'")

print("\n" + "="*50)
print("⏳ THIS MAY TAKE SEVERAL MINUTES. PLEASE WAIT. ⏳")
print("="*50 + "\n")

try:
    # Using the new, simpler API
    kagglehub.model_upload(
        handle=KAGGLE_MODEL_HANDLE,
        local_model_dir=LOCAL_MODEL_DIR,
        version_notes=VERSION_NOTES,
        license_name="Apache 2.0" # A permissive license is good practice
    )
    logger.info("✅ Successfully uploaded new model version!")
except Exception as e:
    logger.error("❌ Failed to upload new model version.")
    logger.error(f"   Error: {e}")
    logger.error("   Please check your internet connection and Kaggle credentials.")
    sys.exit(1)

# --- 4. Display Final Model Handle ---
logger.info("="*50)
logger.info("🎉 SUBMISSION COMPLETE 🎉")
logger.info("="*50)
logger.info("Please use the following Model Handle in your Kaggle write-up:")
print("\n" + "#"*50)
print(f"Kaggle Model Name/ID: {KAGGLE_MODEL_HANDLE}")
print("#"*50 + "\n")
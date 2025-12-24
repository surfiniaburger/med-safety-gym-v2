import sys
import os

# Ensure we can import from med_safety_gym
sys.path.insert(0, os.getcwd())

from med_safety_gym.format_parser import FormatParser, ResponseFormat

model_response = """
<proof>
"panobinostat at a dose of 20 mg/m² administered intravenously once weekly, in combination with continued ONC201..."
</proof>
<proof>
"the combination cohort (n = 45) demonstrated a median overall survival (OS) of 18 months, compared with historical controls receiving radiation alone, whose median OS is 12 months."
</proof>
<answer>
- **Recommended weekly dose of panobinostat for this patient:** **20 mg/m²** (no dose reduction is required at baseline).
- **Median overall survival improvement:** **6 months** (18 months with combination therapy vs. 12 months with radiation alone).
</answer>"""

print("--- RAW RESPONSE ---")
print(model_response)
print("--------------------")

parser = FormatParser()
parsed = parser.parse(model_response, ResponseFormat.AUTO)

print("\n--- PARSED RESULT ---")
print(f"Analysis: '{parsed.analysis}'")
print(f"Proof:    '{parsed.proof}'")
print(f"Final:    '{parsed.final}'")

print("\n--- DIAGNOSIS ---")
if not parsed.proof:
    print("❌ Proof is EMPTY. The parser failed to extract the tags.")
else:
    print("✅ Proof extracted successfully.")

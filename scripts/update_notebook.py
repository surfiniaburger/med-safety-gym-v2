import json

notebook_path = "docs/tutorials/tutorial_1.ipynb"
with open(notebook_path, "r") as f:
    nb = json.load(f)

updated = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_str = "".join(cell["source"])
        if "def evaluate_model" in source_str and "model.generate" in source_str:
            new_source = []
            for line in cell["source"]:
                # Fix 1: model.generate(**inputs -> model.generate(input_ids=inputs
                if "outputs = model.generate(**inputs" in line:
                    line = line.replace("**inputs", "input_ids=inputs")
                    updated = True
                
                # Fix 2: inputs.input_ids.shape -> inputs.shape (since inputs is a Tensor here)
                if "inputs.input_ids.shape" in line:
                    line = line.replace("inputs.input_ids.shape", "inputs.shape")
                    updated = True
                
                new_source.append(line)
            cell["source"] = new_source
            if updated:
                break

if updated:
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("✅ Tutorial notebook updated: Fixed tensor usage in evaluate_model.")
else:
    print("⚠️ No changes made to notebook. Pattern not found.")

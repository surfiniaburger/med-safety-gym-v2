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
                original_line = line
                # Fix 1: model.generate(**inputs -> model.generate(input_ids=inputs
                line = line.replace('outputs = model.generate(**inputs', 'outputs = model.generate(input_ids=inputs')

                # Fix 2: inputs.input_ids.shape -> inputs.shape (since inputs is a Tensor here)
                line = line.replace('inputs.input_ids.shape', 'inputs.shape')

                if original_line != line:
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

import os
import json

experiments = {
    "A (Baseline)": "results/ablation_results/exp_a_baseline/structural_eval.json",
    "B (Curriculum)": "results/ablation_results/exp_b_curriculum/structural_eval.json",
    "C (Topo Loss)": "results/ablation_results/exp_c_topo/structural_eval.json",
    "D (Ours)": "results/ablation_results/exp_d_ours/structural_eval.json"
}

metrics_order = [
    # Standard
    ("Dice", "{:.2%}"),
    ("Accuracy", "{:.2%}"),
    ("Sensitivity", "{:.2%}"),
    ("Specificity", "{:.2%}"),
    ("AUC_ROC", "{:.2%}"),
    # Structural
    ("CCA", "{:.3f}"),
    ("BFR", "{:+.3f}"),
    ("SVD", "{:.3f}"),
    ("SkelDice", "{:.3f}"),
    ("SkelHD", "{:.2f} px"),
    ("BranchDiff", "{:.0f}"),
    ("Betti0Err", "{:.0f}"),
    ("BPR", "{:.3f}"),
    ("JPR", "{:.3f}"),
    ("GED_approx", "{:.1f}")
]

results = {}
for name, path in experiments.items():
    if os.path.exists(path):
        with open(path, "r") as f:
            results[name] = json.load(f)
    else:
        print(f"Warning: {path} not found.")

if not results:
    print("No results found.")
    exit(1)

# Generate Markdown table
header = "| Metric | " + " | ".join(results.keys()) + " |"
separator = "| --- | " + " | ".join(["---"] * len(results)) + " |"
lines = [header, separator]

for m_key, fmt in metrics_order:
    row = f"| **{m_key}** |"
    for name in results.keys():
        val = results[name].get(m_key, None)
        if val is None:
            # try lowercase keys
            val = results[name].get(m_key.lower(), "-")
        
        if isinstance(val, (int, float)):
            formatted_val = fmt.format(val)
        else:
            formatted_val = str(val)
        row += f" {formatted_val} |"
    lines.append(row)

markdown_table = "\n".join(lines)
print("\n=== ABLATION STUDY RESULTS COMPARISON ===")
print(markdown_table)
print("=========================================\n")

# Save table to results/ablation_results/ablation_table.md
os.makedirs("results/ablation_results", exist_ok=True)
with open("results/ablation_results/ablation_table.md", "w") as f:
    f.write("# Ablation Study Metric Comparison\n\n")
    f.write(markdown_table)
    f.write("\n")
print("Saved comparison table to results/ablation_results/ablation_table.md")

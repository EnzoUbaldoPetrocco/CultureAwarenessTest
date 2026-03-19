import pandas as pd
import numpy as np

INPUT_CSV = "Results.csv"
OUTPUT_TEX = "Results.tex"

# -------------------------
# 1. Load CSV
# -------------------------
df = pd.read_csv(INPUT_CSV)

# --- 1. Split numeric vs non-numeric columns ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# --- 2. Forward-fill hierarchy (directory-style logic) ---
df[non_numeric_cols] = df[non_numeric_cols].ffill()

# --- 3. Build Method from hierarchical path ---
def build_method(row):
    parts = []
    for v in row:
        if pd.notna(v) and str(v).strip() != "":
            parts.append(str(v))
    return " + ".join(parts)

df["Method"] = df[non_numeric_cols].apply(build_method, axis=1)

# --- 4. Keep only desired metrics ---
keep_metrics = ["ERR0", "ERR1", "ERR2", "ERR", "CIC"]
df = df[["Method"] + keep_metrics]

# --- 5. Drop rows with no metrics ---
df = df.dropna(subset=keep_metrics, how="all")

# --- 6. Round numeric values ---
df[keep_metrics] = df[keep_metrics].round(2)

# --- 7. Export to LaTeX ---
latex = df.to_latex(
    index=False,
    escape=True,
    column_format="l" + "r" * len(keep_metrics),
    caption="Results",
    label="tab:Results",
    float_format="%.2f",
)

# --- 8. Make it booktabs + table* ---
latex = latex.replace("\\hline\n", "\\toprule\n", 1)
latex = latex.replace("\\hline\n", "\\midrule\n", 1)
latex = latex[::-1].replace("\nhline\\", "\n\\bottomrule", 1)[::-1]
latex = latex.replace("\\begin{table}", "\\begin{table*}")
latex = latex.replace("\\end{table}", "\\end{table*}")

print(latex)




with open(OUTPUT_TEX, "w") as f:
    f.write(latex)

print("LaTeX table written to", OUTPUT_TEX)

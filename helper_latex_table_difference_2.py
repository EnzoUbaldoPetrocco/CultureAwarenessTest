import pandas as pd
import re

def latex_to_df_row_comparison(latex_str):
    # 1. Cleaning and Normalization
    normalized = latex_str.replace(r'\\', '\n').replace('\\', '\n')
    cleaned = re.sub(r'\\(toprule|midrule|bottomrule|hline|hhline\{.*\}|centering|multirow|textbf|%)', '', normalized)
    
    rows = []
    labels = []
    lines = [l.strip() for l in cleaned.split('\n') if l.strip() and '&' in l]
    
    for line in lines:
        cols = [c.strip() for c in line.split('&')]
        labels.append(cols[0])
        
        numeric_row = []
        for col in cols[1:]:
            val_clean = re.sub(r'[^0-9.\-]', '', col)
            if val_clean:
                numeric_row.append(float(val_clean))
        if numeric_row:
            rows.append(numeric_row)

    if not rows:
        return "Error: No valid data rows found."

    df = pd.DataFrame(rows)
    baseline_vector = df.iloc[0]
    diff_df = df.subtract(baseline_vector, axis=1)

    # 2. LaTeX Output Generation
    output_lines = []
    for i in range(len(df)):
        method_label = labels[i]
        current_values = df.iloc[i]
        deltas = diff_df.iloc[i]
        
        formatted_cells = []
        for val, d in zip(current_values, deltas):
            if i == 0:
                # Baseline row
                formatted_cells.append(f"{val:.1f}\%")
            else:
                # Subsequent rows with standard size delta
                sign = "+" if d >= 0 else ""
                # Format: 12.6% (+1.2)
                diff_str = f" ({sign}{d:.1f})\%"
                formatted_cells.append(f"{val:.1f}\%{diff_str}")
        
        # JOIN all cells for the row and add the LaTeX line break
        row_string = f"{method_label:<12} & {' & '.join(formatted_cells)} \\\\"
        output_lines.append(row_string)
    
    return "\n".join(output_lines)

# --- INPUT DATA ---
raw_input = r"""
Baseline & 12.8 & 15.7 & 19.6 & 16.0 & 4.0 \\
DIV & 11.2 & 15.4 & 15.6 & 14.1 & 3.3 \\
NODIV & 12.0 & 15.1 & 16.8 & 14.6 & 2.6 \\
"""

print(latex_to_df_row_comparison(raw_input))
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
                diff_str = f" ({sign}{d:.1f}\%)"
                formatted_cells.append(f"{val:.1f}\%{diff_str}")
        
        # JOIN all cells for the row and add the LaTeX line break
        row_string = f"{method_label:<12} & {' & '.join(formatted_cells)} \\\\"
        output_lines.append(row_string)
    
    return "\n".join(output_lines)

# --- INPUT DATA ---
raw_input = r"""
MTL   & 22.5&	20.9&	23.8&	22.4&	2.2\\
MTL +  RT  & 26.1&	20.6&	23.4&	23.4&	3.4\\	
MTL +  DM  & 26.3&	17.8&	21.3&	21.8&	4.31\\	
MTL +  DMMIN  & 26.6&	18.9&	18.5&	21.3&	3.9\\	
MTL + BB  & 20.5&	17.5&	24.2&	20.7&	3.2\\
MTL + BB + RT  & 23.3&	14.8& 23.0&	20.4&	5.6\\	
MTL + BB + DM  & 23.1& 16.6& 22.2& 20.7& 4.1\\	
MTL + BB + DMMIN  & 22.9&	17.7&	21.6&	20.7&	3.6\\
"""

print(latex_to_df_row_comparison(raw_input))
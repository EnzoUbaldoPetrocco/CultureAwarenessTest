import pandas as pd
import re

def latex_to_df_flexible(latex_str):
    """
    Parses a LaTeX string into a DataFrame, dynamically identifying 
    the number of columns and rows.
    """
    # 1. Standardize line breaks and remove LaTeX structural commands
    normalized = latex_str.replace(r'\\', '\n').replace('\\', '\n')
    cleaned_text = re.sub(r'\\(toprule|midrule|bottomrule|hline|hhline\{.*\}|centering|multirow|textbf|%)', '', normalized)
    
    rows = []
    index_labels = []
    
    # Split by newline and filter for content-bearing lines
    lines = [l.strip() for l in cleaned_text.split('\n') if l.strip() and '&' in l]
    
    for line in lines:
        cols = [c.strip() for c in line.split('&')]
        
        # The first column is used as the row label
        index_labels.append(cols[0])
        
        # Extract numeric values from all remaining columns
        numeric_row = []
        for col in cols[1:]:
            # Strip all non-numeric characters except decimals and signs
            val_clean = re.sub(r'[^0-9.\-]', '', col)
            if val_clean:
                numeric_row.append(float(val_clean))
        
        if numeric_row:
            rows.append(numeric_row)

    # 2. Construct DataFrame with dynamic column naming
    if not rows:
        return pd.DataFrame()
        
    num_cols = len(rows[0])
    col_names = [f"Col_{i+1}" for i in range(num_cols)]
    
    df = pd.DataFrame(rows, columns=col_names)
    df.insert(0, "Label", index_labels)
    
    return df.set_index("Label")

# --- INPUT DATA ---
# This script now accepts any number of rows (3, 18, etc.) and any number of columns
raw_dm = r""" 
Baseline & 24.6&	17.9&	25.2&	22.6&	4.8\\	
DIV & 20.6&	17.4&	22.1&	20.0&	3.3\\
NODIV & 22.8&	19.7&	22.1&	21.5&	2.3\\
"""

raw_dmmin = r"""
RT  & 26.2	 &	20.0	 &	24.3	 &	23.5	&	3.5\\
RT + DIV & 21.2 & 16.2 & 22.9 & 20.1 & 4.0 \\
RT + NODIV & 22.9 & 18.2 & 21.6 & 20.9 & 3.5 \\
"""

# --- EXECUTION ---
df_dm = latex_to_df_flexible(raw_dm)
df_dmmin = latex_to_df_flexible(raw_dmmin)

# Check if shapes match before calculation
if df_dm.shape == df_dmmin.shape:
    # Perform numeric subtraction on the underlying values
    # diff = New - Baseline
    diff_values = df_dmmin.values - df_dm.values
    
    # --- OUTPUT FORMATTING ---
    print(f"{'Strategy':<15} | Results (Value ± Diff)")
    print("-" * 60)
    
    for i in range(len(df_dmmin)):
        current_label = df_dmmin.index[i]
        vals = df_dmmin.iloc[i]
        diffs = diff_values[i]
        
        formatted_cells = []
        for v, d in zip(vals, diffs):
            # Format: Value (Sign Diff)
            # Using -d logic if you prefer to flip signs, otherwise standard d
            sign = "+" if d >= 0 else ""
            formatted_cells.append(f"{v:.1f}\% ({sign}{d:.1f}\%)")
        
        print(f"{current_label:<15} & {' & '.join(formatted_cells)} \\\\")
else:
    print("Error: Input tables have different dimensions.")
    print(f"DM Shape: {df_dm.shape}, DMMIN Shape: {df_dmmin.shape}")
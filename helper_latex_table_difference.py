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
DIV, 0 & 92.8&	95.2&	44.8&	25.9&	47.6&	99.3&	57.6	\\
DIV, 1 & 85.4&	95.3&	44.8&	28.9&	41.7&	99.6&	56.7\\
NODIV & 91.3	&	94.8&	54.2&	46.4&	62.8&	98.8&	69.3\\
"""

raw_dmmin = r"""
DIV 0 & 90.2& 92.0& 62.3& 55.6& 76.0& 96.5& 76.0 \\
DIV 1 & 85.3& 93.9& 62.8& 68.5& 60.3& 97.2& 75.4 \\
NODIV & 90.5& 94.3& 73.5& 75.6& 79.7& 96.5& 83.9 \\
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
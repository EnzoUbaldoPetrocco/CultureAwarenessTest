import pandas as pd
import os
import numpy as np

def calculate_metrics_and_cic_final():
    root_search_path = "." 
    culture_names = ["CULTURE0", "CULTURE1", "CULTURE2"]
    filename = "res.csv"
    
    for root, dirs, files in os.walk(root_search_path):
        # Identify the folder structure containing the 3 culture subdirectories
        if all(os.path.isdir(os.path.join(root, c)) for c in culture_names):
            print(f"Processing experiment in: {root}")
            
            try:
                # Helper to extract error percentage per row
                # (val2 + val3) / sum(val1, val2, val3, val4)
                def get_errors(path):
                    if not os.path.exists(path):
                        return []
                    # Reading raw CSV (no headers)
                    df_raw = pd.read_csv(path, header=None)
                    row_errors = []
                    for _, row in df_raw.iterrows():
                        vals = row.values
                        # Ensure we have at least 4 values for the 2x2 flattened matrix
                        if len(vals) >= 4:
                            off_diag = vals[1] + vals[2]
                            total = sum(vals[:4])
                            if total > 0:
                                row_errors.append((off_diag / total) * 100)
                    return row_errors

                errs_c0 = get_errors(os.path.join(root, "CULTURE0", filename))
                errs_c1 = get_errors(os.path.join(root, "CULTURE1", filename))
                errs_c2 = get_errors(os.path.join(root, "CULTURE2", filename))

                # Synchronize row counts (must have same number of experiments)
                min_rows = min(len(errs_c0), len(errs_c1), len(errs_c2))
                if min_rows == 0:
                    continue

                # Calculate CIC for every row individually
                # CIC_i = 1/3 * sum(Error_culture_j - min_error_in_row)
                row_cics = []
                for i in range(min_rows):
                    current_row_errs = [errs_c0[i], errs_c1[i], errs_c2[i]]
                    m_err = min(current_row_errs)
                    cic_i = sum([(e - m_err) for e in current_row_errs]) / 3
                    row_cics.append(cic_i)

                # Compute Final Averages across all rows
                avg_c0 = sum(errs_c0[:min_rows]) / min_rows
                avg_c1 = sum(errs_c1[:min_rows]) / min_rows
                avg_c2 = sum(errs_c2[:min_rows]) / min_rows
                
                # Global Average Error (ERR): Average of the three culture averages
                avg_err = (avg_c0 + avg_c1 + avg_c2) / 3
                
                # Average CIC
                avg_cic = sum(row_cics) / min_rows

                # Generate LaTeX string: ERR0 & ERR1 & ERR2 & ERR & CIC \\
                latex_row = (f"{avg_c0:.1f}\\% & {avg_c1:.1f}\\% & "
                             f"{avg_c2:.1f}\\% & {avg_err:.1f}\\% & "
                             f"{avg_cic:.1f}\\% \\\\")

                # Save to fixed filename
                output_file = os.path.join(root, "cic_analysis.txt")
                with open(output_file, "w") as f:
                    f.write(latex_row)
                
                print(f"  Success -> {latex_row}")

            except Exception as e:
                print(f"  Skipping {root} due to error: {e}")

if __name__ == "__main__":
    calculate_metrics_and_cic_final()
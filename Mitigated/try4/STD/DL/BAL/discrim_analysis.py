import pandas as pd
import glob
import os
import numpy as np

def process_results():
    root_dir = "."
    found_files = False

    for root, dirs, files in os.walk(root_dir):

        for file in files:
            if (file.startswith("res_scrimin") or file.startswith("res_discrim")) and file.endswith(".csv") and "_analysis" not in file:
                
                found_files = True
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                
                try:
                    df = pd.read_csv(file_path)
                    # Filter rows identifying the confusion matrix data (CM in second column)
                    cm_rows = df[df.iloc[:, 1] == 'CM'].iloc[:, 2:5]
                    
                    if cm_rows.empty:
                        continue

                    # Clean data and convert to numpy array
                    data = cm_rows.apply(pd.to_numeric, errors='coerce').dropna().values
                    num_matrices = len(data) // 3
                    if num_matrices == 0:
                        continue
                        
                    matrices = data[:num_matrices*3].reshape(num_matrices, 3, 3)
                    
                    # Lists to store metrics for each fold
                    accs = []
                    precs = [[], [], []]
                    recs = [[], [], []]
                    
                    for m in matrices:
                        total = np.sum(m)
                        if total == 0: continue
                        
                        # Overall Accuracy
                        accs.append((np.trace(m) / total) * 100)
                        
                        # Per-class Precision and Recall (multiplied by 100 for percentage)
                        for i in range(3):
                            tp = m[i, i]
                            fp = np.sum(m[:, i]) - tp
                            fn = np.sum(m[i, :]) - tp
                            
                            p = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
                            r = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
                            
                            precs[i].append(p)
                            recs[i].append(r)
                    
                    # Calculate Means and Stds
                    m_acc, s_acc = np.mean(accs), np.std(accs)
                    m_precs = [np.mean(p) for p in precs]
                    s_precs = [np.std(p) for p in precs]
                    m_recs = [np.mean(r) for r in recs]
                    s_recs = [np.std(r) for r in recs]

                    # Construct LaTeX row: Prec0 & Prec1 & Prec2 & Rec0 & Rec1 & Rec2 & Acc \\
                    latex_elements = []
                    # Precisions
                    for m, s in zip(m_precs, s_precs):
                        latex_elements.append(f"{m:.1f} \\pm {s:.1f}")
                    # Recalls
                    for m, s in zip(m_recs, s_recs):
                        latex_elements.append(f"{m:.1f} \\pm {s:.1f}")
                    # Accuracy
                    latex_elements.append(f"{m_acc:.1f} \\pm {s_acc:.1f}")
                    
                    latex_row = " & ".join(latex_elements) + " \\\\"

                    # Save results
                    output_file = file_path.replace(".csv", "_analysis.csv")
                    with open(output_file, 'w') as f:
                        f.write("--- LATEX FORMAT (Percentage) ---\n")
                        f.write("Prec C0 & Prec C1 & Prec C2 & Rec C0 & Rec C1 & Rec C2 & Acc \\\\\n")
                        f.write(latex_row + "\n\n")
                        
                        f.write("--- RAW METRICS ---\n")
                        stats = [["Accuracy", m_acc, s_acc]]
                        for i in range(3):
                            stats.append([f"Precision_C{i}", m_precs[i], s_precs[i]])
                            stats.append([f"Recall_C{i}", m_recs[i], s_recs[i]])
                        pd.DataFrame(stats, columns=["Metric", "Mean", "Std"]).to_csv(f, index=False)

                    print(f"  Saved LaTeX row and metrics to: {output_file}")
                    
                except Exception as e:
                    print(f"  Error processing {file}: {e}")

if __name__ == "__main__":
    process_results()

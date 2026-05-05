import os
import csv
import re
import pandas as pd

def extract_stats_from_line(line_content):
    """
    Extracts key-value pairs from the string format: 
    'key': <tf.Tensor... numpy=0.12345>
    """
    pattern = r"'(\w+)':.*?numpy=([0-9.]+)"
    matches = re.findall(pattern, line_content)
    if matches:
        return {key: float(val) for key, val in matches}
    return None

def process_directory(root_directory, precision=3):
    """
    Recursively processes CSVs and saves stats with limited float precision.
    """
    for root, _, files in os.walk(root_directory):
        for file in files:
            # Avoid processing files we already generated
            if file.endswith(".csv") and not file.endswith("_stats.csv"):
                file_path = os.path.join(root, file)
                extracted_data = []
                
                try:
                    with open(file_path, mode='r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if row:
                                line_stats = extract_stats_from_line(row[0])
                                if line_stats:
                                    extracted_data.append(line_stats)
                    
                    if extracted_data:
                        df = pd.DataFrame(extracted_data)
                        stats_df = df.describe().loc[['mean', 'std', 'min', 'max']]
                        
                        # Generate output path
                        base_name = os.path.splitext(file)[0]
                        output_path = os.path.join(root, f"{base_name}_stats.csv")
                        
                        # --- KEY CHANGE HERE ---
                        # float_format="%.3f" limits output to 3 decimal places
                        stats_df.to_csv(output_path, float_format=f"%.{precision}f")
                        
                        print(f"✅ Processed {file} (Precision: {precision})")
                    else:
                        print(f"⚠️  No valid data in {file}")
                        
                except Exception as e:
                    print(f"❌ Error in {file}: {e}")

if __name__ == '__main__':
    # Set your folder path and desired decimal places here
    target_folder = '.' 
    decimal_places = 4 
    
    process_directory(target_folder, precision=decimal_places)
import re
import csv
import pandas as pd

def transform_tensor_csv(input_filename, output_filename):
    # Regex explanation:
    # '(\w+)':  -> Matches the key (e.g., n_loss) inside quotes
    # .*?       -> Matches characters between key and value lazily
    # numpy=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?) -> Captures the numeric value after 'numpy='
    # It handles integers, decimals, and scientific notation (e.g., 1.2e-05)
    regex_pattern = re.compile(r"'(\w+)':.*?numpy=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    all_data = []

    try:
        with open(input_filename, 'r') as f:
            # Use csv.reader to handle the quotes correctly
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                
                line_text = row[0]
                # Find all (key, value) pairs in the line
                matches = regex_pattern.findall(line_text)
                
                if matches:
                    # Convert list of tuples [('n_loss', '0.18'), ...] into a dictionary
                    row_dict = {key: value for key, value in matches}
                    all_data.append(row_dict)

        # Convert to DataFrame to handle headers and structure automatically
        df = pd.DataFrame(all_data)
        
        # Save to CSV
        df.to_csv(output_filename, index=False)
        print(f"Success! Data saved to {output_filename}")
        print("Columns created:", df.columns.tolist())

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the transformation
transform_tensor_csv('./kid_0.csv', 'res_0.csv')
transform_tensor_csv('./kid_1.csv', 'res_1.csv')
transform_tensor_csv('./kid_2.csv', 'res_2.csv')
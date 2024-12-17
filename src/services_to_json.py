import pandas as pd
import json

# Load the two Excel files
file_1_path = 'C:/Users/User/Desktop/insupply/data/raw/services.xlsx'  
file_2_path = 'C:/Users/User/Desktop/insupply/data/raw/Inc 7100010444-ZWK ME3N_Final_V0.1 - services (for roshan).xlsx' 

# Load data from the two Excel files
df_file_1 = pd.read_excel(file_1_path) 
df_file_2 = pd.read_excel(file_2_path, header=1)

# Step 1: Check if required columns exist
if 'Operating Expenditure' not in df_file_1.columns or 'Long text' not in df_file_1.columns:
    raise ValueError("First Excel file must contain 'Operating Expenditure' and 'Long text' columns.")
if 'Validated GL' not in df_file_2.columns or 'SM' not in df_file_2.columns:
    raise ValueError("Second Excel file must contain 'Validated GL' and 'SM' columns.")

# Step 2: Extract numbers before the first '-' in both files
def extract_number_before_dash(value):
    if isinstance(value, str) and ' - ' in value:
        return value.split(' - ')[0].strip()
    return None

# Extract 'number_key' from first file (Operating Expenditure)
df_file_1['number_key'] = df_file_1['Operating Expenditure'].apply(extract_number_before_dash)
df_file_1['description'] = df_file_1['Long text']

# Extract 'number_key' and 'material_number' from second file (Validated GL and SM)
df_file_2['number_key'] = df_file_2['Validated GL'].apply(extract_number_before_dash)
df_file_2['material_number'] = df_file_2['SM']

# Step 3: Merge both datasets on 'number_key'
merged_df = pd.merge(
    df_file_1[['number_key', 'description']],
    df_file_2[['number_key', 'material_number']],
    on='number_key',
    how='inner'
)

# Step 4: Clean up the merged DataFrame
merged_df = merged_df[['material_number', 'description']].drop_duplicates()
merged_df = merged_df.dropna(subset=['material_number', 'description'])  # Drop missing values

# Step 5: Convert to list of dictionaries
materials_list = merged_df.to_dict(orient='records')

# Step 6: Save to JSON file
output_path = 'materials.json'  # Output JSON file path
with open(output_path, 'w') as json_file:
    json.dump(materials_list, json_file, indent=4)

print(f"JSON file saved to {output_path}")
print("Sample Data:")
print(json.dumps(materials_list[:2], indent=4))  # Display first 2 rows

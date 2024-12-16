import pandas as pd
import json

# Load the Excel file
file_path = 'C:/Users/User/Desktop/insupply-main/datasets/services.xlsx'  
df = pd.read_excel(file_path)

# Check if the required columns exist
if 'Operating Expenditure' not in df.columns or 'Long text' not in df.columns:
    print("Error: The required columns ('Operating Expenditure' and 'Long text') are missing.")
else:
    # Extract material_number and description
    def extract_material_number(value):
        if isinstance(value, str):
            return value.split(' - ')[0]  
        return None

    df['material_number'] = df['Operating Expenditure'].apply(extract_material_number)
    df['description'] = df['Long text']  

    # Select and drop duplicates
    df = df[['material_number', 'description']].drop_duplicates()

    # Drop rows where material_number or description is missing
    df = df.dropna(subset=['material_number', 'description'])

    # Convert to list of dictionaries
    materials_list = df.to_dict(orient='records')

    # Save to JSON file
    output_path = 'materials.json'  # Output JSON file path
    with open(output_path, 'w') as json_file:
        json.dump(materials_list, json_file, indent=4)

    print(f"JSON file saved to {output_path}")
    print("Sample Data:")
    print(json.dumps(materials_list[:2], indent=4))  # Display first 2 rows

import pandas as pd

# Load the Excel file
file_path = "Bakery Sales.xlsx"  # Update if needed
xls = pd.ExcelFile(file_path)

df = pd.read_excel(xls, sheet_name="Bakery Sales")

# Remove rows where the 'datetime' column is missing
df = df.dropna(subset=["datetime"])

# Define categories and their corresponding columns
categories = {
    "Breads": ["angbutter", "plain bread", "jam", "croissant", "tiramisu croissant", "pandoro", "orange pound", "wiener"],
    "Beverages": ["americano", "caffe latte", "lemon ade", "vanila latte", "berry ade"],
    "Desserts": ["cheese cake", "tiramisu", "gateau chocolat", "merinque cookies"]
}

# Function to summarize items per category
def summarize_items(row, item_columns):
    items = []
    for col in item_columns:
        if col in df.columns and isinstance(row[col], (int, float)) and row[col] > 0:
            items.extend([col] * int(row[col]))  # Repeat the name based on the quantity
    return ", ".join(items) if items else "N/A"

# Apply transformation for each category
for category, columns in categories.items():
    df[category] = df.apply(lambda row: summarize_items(row, columns), axis=1)

# Keep only relevant columns
df_transformed = df[["datetime", "day of week", "total", "Breads", "Beverages", "Desserts"]]

# Save the transformed data to a new Excel file
df_transformed.to_excel("Transformed_Bakery_Sales.xlsx", index=False)

print("Data cleaning and transformation complete. File saved as 'Transformed_Bakery_Sales.xlsx'.")

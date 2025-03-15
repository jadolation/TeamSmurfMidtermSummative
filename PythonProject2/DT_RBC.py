import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

# Load the cleaned dataset
file_path = "Transformed_Bakery_Sales.xlsx"
df = pd.read_excel(file_path)

# Define sales category (Target Variable)
def classify_sales(total):
    if total > 50000:
        return "High"
    elif 20000 <= total <= 50000:
        return "Medium"
    else:
        return "Low"

df['Sales_Category'] = df['total'].apply(classify_sales)

# Convert categorical 'day of week' to numerical using Label Encoding
label_encoder = LabelEncoder()
df['Day_Num'] = label_encoder.fit_transform(df['day of week'])

# Convert item columns to presence count
for col in ['Breads', 'Beverages', 'Desserts']:
    df[col] = df[col].apply(lambda x: len(str(x).split(', ')) if pd.notna(x) else 0)

# Selecting features and target
features = ['Day_Num', 'total', 'Breads', 'Beverages', 'Desserts']
X = df[features]
y = df['Sales_Category']

# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=42)
dtree = dtree.fit(X_train, y_train)

# Plot Decision Tree
plt.figure(figsize=(15, 10))
tree.plot_tree(dtree, feature_names=features, class_names=dtree.classes_, filled=True)
plt.show()

# Rule-Based Classifier
def rule_based_classification(row):
    if row['day of week'] in ['Sat', 'Sun'] and row['total'] > 50000:
        return "High Sales"
    elif row['Breads'] > row['Beverages'] and row['Breads'] > row['Desserts']:
        return "Bread-Heavy Sales Day"
    elif row['Beverages'] > row['Breads'] and row['Beverages'] > row['Desserts']:
        return "Beverage-Heavy Sales Day"
    else:
        return "Balanced Sales"

df['Rule_Based_Category'] = df.apply(rule_based_classification, axis=1)

# Save results
df.to_excel("Classified_Bakery_Sales.xlsx", index=False)

print("Classification complete. Decision tree generated and results saved in 'Classified_Bakery_Sales.xlsx'.")

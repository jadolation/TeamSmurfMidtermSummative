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

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
file_path = "Transformed_Bakery_Sales.xlsx"
df = pd.read_excel(file_path)

# Split 90% for training & testing, 10% as unseen data
train_test_data, unseen_data = train_test_split(df, test_size=0.10, random_state=42)

# Save the datasets
unseen_data_path = "Unseen_Data.xlsx"

unseen_data.to_excel(unseen_data_path, index=False)

unseen_data_path

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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the dataset
file_path = 'Transformed_Bakery_Sales.xlsx'
bakery_data = pd.read_excel(file_path)

# Data cleaning and preprocessing
bakery_data.dropna(subset=['Breads'], inplace=True)

# Extract date and time features
bakery_data['datetime'] = pd.to_datetime(bakery_data['datetime'])
bakery_data['hour'] = bakery_data['datetime'].dt.hour
bakery_data['day_of_week'] = bakery_data['datetime'].dt.day_name()

# Simplify target columns (Breads, Beverages, Desserts) to the first item for classification
bakery_data['Breads'] = bakery_data['Breads'].apply(lambda x: x.split(',')[0].strip())
bakery_data['Beverages'] = bakery_data['Beverages'].fillna('None').apply(lambda x: x.split(',')[0].strip())
bakery_data['Desserts'] = bakery_data['Desserts'].fillna('None').apply(lambda x: x.split(',')[0].strip())

# Features and separate targets
X = bakery_data[['hour', 'day_of_week']]
y_breads = bakery_data['Breads']
y_beverages = bakery_data['Beverages']
y_desserts = bakery_data['Desserts']

# One-hot encoding for day_of_week
X = pd.get_dummies(X, columns=['day_of_week'])

# Split data
X_train, X_test, y_train_breads, y_test_breads = train_test_split(X, y_breads, test_size=0.2, random_state=42)
X_train, X_test, y_train_beverages, y_test_beverages = train_test_split(X, y_beverages, test_size=0.2, random_state=42)
X_train, X_test, y_train_desserts, y_test_desserts = train_test_split(X, y_desserts, test_size=0.2, random_state=42)

# SVM models
svm_breads = SVC(kernel='linear')
svm_breads.fit(X_train, y_train_breads)

svm_beverages = SVC(kernel='linear')
svm_beverages.fit(X_train, y_train_beverages)

svm_desserts = SVC(kernel='linear')
svm_desserts.fit(X_train, y_train_desserts)

# Predictions
y_pred_breads = svm_breads.predict(X_test)
y_pred_beverages = svm_beverages.predict(X_test)
y_pred_desserts = svm_desserts.predict(X_test)

# Classification reports
report_breads = classification_report(y_test_breads, y_pred_breads)
report_beverages = classification_report(y_test_beverages, y_pred_beverages)
report_desserts = classification_report(y_test_desserts, y_pred_desserts)

print("Breads Classification Report:\n", report_breads)
print("Beverages Classification Report:\n", report_beverages)
print("Desserts Classification Report:\n", report_desserts)

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

# Splitting data into 90% training & testing, 10% unseen data
train_test_data, unseen_data = train_test_split(df, test_size=0.10, random_state=42)

# Further split train_test_data into 80% training and 20% testing
train_data, test_data = train_test_split(train_test_data, test_size=0.20, random_state=42)

# Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=42)
dtree = dtree.fit(train_data[features], train_data['Sales_Category'])

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
train_data.to_excel("Train_Data.xlsx", index=False)
test_data.to_excel("Test_Data.xlsx", index=False)
unseen_data.to_excel("Unseen_Data.xlsx", index=False)

print("Classification complete. Decision tree generated and results saved in 'Classified_Bakery_Sales.xlsx'.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Load the cleaned dataset
file_path = "/content/Transformed_Bakery_Sales.xlsx"
df = pd.read_excel(file_path)

# Remove unintended unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Define sales category (Target Variable)
def classify_sales(total):
    if total > 50000:
        return "High"
    elif 20000 <= total <= 50000:
        return "Medium"
    else:
        return "Low"

df['Sales_Category'] = df['total'].apply(classify_sales)

# Convert categorical 'day of week' to numerical with custom mapping
day_mapping = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thur': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
df['Day_Num'] = df['day of week'].map(day_mapping)

# Convert item columns to presence count
for col in ['Breads', 'Beverages', 'Desserts']:
    df[col] = df[col].apply(lambda x: len(str(x).split(', ')) if pd.notna(x) else 0)

# Selecting features and target
features = ['Day_Num', 'total', 'Breads', 'Beverages', 'Desserts']
X = df[features]
y = df['Sales_Category']

# Splitting data into 90% training & testing, 10% unseen data
train_test_data, unseen_data = train_test_split(df, test_size=0.10, random_state=42)

# Further split train_test_data into 80% training and 20% testing
train_data, test_data = train_test_split(train_test_data, test_size=0.20, random_state=42)

# Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=42)
dtree = dtree.fit(train_data[features], train_data['Sales_Category'])

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
train_data.to_excel("Train_Data.xlsx", index=False)
test_data.to_excel("Test_Data.xlsx", index=False)
unseen_data.to_excel("Unseen_Data.xlsx", index=False)

print("Classification complete. Decision tree generated and results saved in 'Classified_Bakery_Sales.xlsx'.")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Load the cleaned dataset
file_path = "/content/Transformed_Bakery_Sales.xlsx"
df = pd.read_excel(file_path)

# Remove unintended unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Define sales category (Target Variable)
def classify_sales(total):
    if total > 50000:
        return "High"
    elif 20000 <= total <= 50000:
        return "Medium"
    else:
        return "Low"

df['Sales_Category'] = df['total'].apply(classify_sales)

# Convert categorical 'day of week' to numerical with custom mapping
day_mapping = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thur': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
df['Day_Num'] = df['day of week'].map(day_mapping)

# Convert item columns to presence count
for col in ['Breads', 'Beverages', 'Desserts']:
    df[col] = df[col].apply(lambda x: len(str(x).split(', ')) if pd.notna(x) else 0)

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

# Selecting features and target
features = ['Day_Num', 'total', 'Breads', 'Beverages', 'Desserts']
X = df[features]
y = df['Sales_Category']

# Splitting data into 90% training & testing, 10% unseen data
train_test_data, unseen_data = train_test_split(df, test_size=0.10, random_state=42)

# Further split train_test_data into 80% training and 20% testing
train_data, test_data = train_test_split(train_test_data, test_size=0.20, random_state=42)

# Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=42)
dtree = dtree.fit(train_data[features], train_data['Sales_Category'])

# Plot Decision Tree
plt.figure(figsize=(15, 10))
tree.plot_tree(dtree, feature_names=features, class_names=dtree.classes_, filled=True)
plt.show()

# Save results
df.to_excel("Classified_Bakery_Sales.xlsx", index=False)
train_data.to_excel("Train_Data.xlsx", index=False)
test_data.to_excel("Test_Data.xlsx", index=False)
unseen_data.to_excel("Unseen_Data.xlsx", index=False)

print("Classification complete. Decision tree generated and results saved in 'Classified_Bakery_Sales.xlsx'.")



# Let's evaluate multiple algorithms using 10-fold cross-validation

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocess Data (same as before)
def count_items(s):
    if pd.isnull(s):
        return 0
    return len(str(s).split(','))

# Create new count features if not already created
if 'bread_count' not in df.columns:
    df['bread_count'] = df['Breads'].apply(count_items)
if 'beverage_count' not in df.columns:
    df['beverage_count'] = df['Beverages'].apply(count_items)
if 'dessert_count' not in df.columns:
    df['dessert_count'] = df['Desserts'].apply(count_items)

# For simplicity, create a binary target: 1 if total sales > median, otherwise 0
if 'high_sale' not in df.columns:
    median_total = df['total'].median()
    df['high_sale'] = (df['total'] > median_total).astype(int)

# Use day of week as one-hot features and combine with count features
day_dummies = pd.get_dummies(df['day of week'], prefix='day')
features = pd.concat([day_dummies, df[['bread_count', 'beverage_count', 'dessert_count']]], axis=1)

X = features
Y = df['high_sale']

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Setup 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Results dictionary
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'ROC-AUC': [],
    'Confusion Matrix': []
}

# Evaluate each model
for name, model in models.items():
    print(f"Evaluating {name}...")

    # Storage for metrics
    acc_scores = []
    prec_scores = []
    recall_scores = []
    roc_auc_scores = []

    # Confusion matrix aggregate
    conf_matrix_total = np.zeros((2, 2), dtype=int)

    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # For ROC-AUC, we need probability estimates
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # For models without predict_proba, use decision_function if available
            if hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                y_prob = y_pred  # Fallback

        # Compute metrics for the fold
        acc_scores.append(accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_test, y_pred, zero_division=0))

        try:
            roc_auc_scores.append(roc_auc_score(y_test, y_prob))
        except ValueError:
            roc_auc_scores.append(0)

        # Update aggregate confusion matrix
        conf_matrix_total += confusion_matrix(y_test, y_pred, labels=[0,1])

    # Average metrics
    avg_accuracy = np.mean(acc_scores)
    avg_precision = np.mean(prec_scores)
    avg_recall = np.mean(recall_scores)
    avg_roc_auc = np.mean(roc_auc_scores)

    # Store results
    results['Model'].append(name)
    results['Accuracy'].append(avg_accuracy)
    results['Precision'].append(avg_precision)
    results['Recall'].append(avg_recall)
    results['ROC-AUC'].append(avg_roc_auc)
    results['Confusion Matrix'].append(conf_matrix_total)

# Convert results to DataFrame for easy viewing
results_df = pd.DataFrame({
    'Model': results['Model'],
    'Accuracy': results['Accuracy'],
    'Precision': results['Precision'],
    'Recall': results['Recall'],
    'ROC-AUC': results['ROC-AUC']
})

print("\
Model Performance Metrics:")
print(results_df)

# Plot confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (name, cm) in enumerate(zip(results['Model'], results['Confusion Matrix'])):
    if i < len(axes):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - {name}', fontsize=16, pad=15)
        axes[i].set_xlabel('Predicted Label', fontsize=14, labelpad=10)
        axes[i].set_ylabel('True Label', fontsize=14, labelpad=10)
        axes[i].tick_params(labelsize=12)

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()

# Plot performance metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC']
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(results['Model']))
width = 0.2
multiplier = 0

for metric in metrics:
    offset = width * multiplier
    rects = ax.bar(x + offset, results_df[metric], width, label=metric)
    multiplier += 1

ax.set_title('Model Performance Comparison', fontsize=20, pad=15)
ax.set_xticks(x + width, results['Model'], fontsize=14)
ax.set_ylabel('Score', fontsize=16, labelpad=10)
ax.set_ylim(0, 1.0)
ax.legend(loc='lower right', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

print("Done")

# This cell performs data preprocessing, creates a binary classification target, and evaluates a logistic regression model using 10-fold cross-validation.

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score

# Load the dataset
file_path = '/content/Unseen_Data.xlsx'  # Update with the correct path
try:
    df = pd.read_excel(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Preprocess Data
# Create counts for Breads, Beverages, and Desserts

def count_items(s):
    if pd.isnull(s):
        return 0
    return len(str(s).split(','))

# Create new count features
if 'Breads' in df.columns:
    df['bread_count'] = df['Breads'].apply(count_items)
if 'Beverages' in df.columns:
    df['beverage_count'] = df['Beverages'].apply(count_items)
if 'Desserts' in df.columns:
    df['dessert_count'] = df['Desserts'].apply(count_items)

# For simplicity, create a binary target: 1 if total sales > median, otherwise 0
median_total = df['total'].median()
df['high_sale'] = (df['total'] > median_total).astype(int)

# Use day of week as one-hot features and combine with count features
if 'day of week' in df.columns:
    day_dummies = pd.get_dummies(df['day of week'], prefix='day')
    features = pd.concat([day_dummies, df[['bread_count', 'beverage_count', 'dessert_count']]], axis=1)
else:
    features = df[['bread_count', 'beverage_count', 'dessert_count']]

X = features
Y = df['high_sale']

# Initialize model
model = LogisticRegression(max_iter=200)

# Setup 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Storage for metrics
acc_scores = []
prec_scores = []
recall_scores = []
roc_auc_scores = []

# Confusion matrix aggregate
conf_matrix_total = np.zeros((2, 2), dtype=int)

for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute metrics for the fold
    acc_scores.append(accuracy_score(y_test, y_pred))
    prec_scores.append(precision_score(y_test, y_pred, zero_division=0))
    recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
    try:
        roc_auc_scores.append(roc_auc_score(y_test, y_prob))
    except ValueError:
        roc_auc_scores.append(0)

    # Update aggregate confusion matrix
    conf_matrix_total += confusion_matrix(y_test, y_pred, labels=[0,1])

# Average metrics
avg_accuracy = np.mean(acc_scores)
avg_precision = np.mean(prec_scores)
avg_recall = np.mean(recall_scores)
avg_roc_auc = np.mean(roc_auc_scores)

print('Aggregated Confusion Matrix (sum over folds):')
print(conf_matrix_total)

print('\
Average Accuracy: ' + str(avg_accuracy))
print('Average Precision: ' + str(avg_precision))
print('Average Recall: ' + str(avg_recall))
print('Average ROC-AUC: ' + str(avg_roc_auc))

print('\
Done')

# This cell performs data preprocessing, creates a binary classification target, and evaluates multiple models using 10-fold cross-validation.

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score

# Load the dataset
file_path = '/content/Unseen_Data.xlsx'  # Update with the correct path
try:
    df = pd.read_excel(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Preprocess Data
# Create counts for Breads, Beverages, and Desserts

def count_items(s):
    if pd.isnull(s):
        return 0
    return len(str(s).split(','))

# Create new count features
if 'Breads' in df.columns:
    df['bread_count'] = df['Breads'].apply(count_items)
if 'Beverages' in df.columns:
    df['beverage_count'] = df['Beverages'].apply(count_items)
if 'Desserts' in df.columns:
    df['dessert_count'] = df['Desserts'].apply(count_items)

# For simplicity, create a binary target: 1 if total sales > median, otherwise 0
median_total = df['total'].median()
df['high_sale'] = (df['total'] > median_total).astype(int)

# Use day of week as one-hot features and combine with count features
if 'day of week' in df.columns:
    day_dummies = pd.get_dummies(df['day of week'], prefix='day')
    features = pd.concat([day_dummies, df[['bread_count', 'beverage_count', 'dessert_count']]], axis=1)
else:
    features = df[['bread_count', 'beverage_count', 'dessert_count']]

X = features
Y = df['high_sale']

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Setup 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Results dictionary
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'ROC-AUC': [],
    'Confusion Matrix': []
}

# Evaluate each model
for name, model in models.items():
    print(f"Evaluating {name}...")

    # Storage for metrics
    acc_scores = []
    prec_scores = []
    recall_scores = []
    roc_auc_scores = []

    # Confusion matrix aggregate
    conf_matrix_total = np.zeros((2, 2), dtype=int)

    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # For ROC-AUC, we need probability estimates
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            if hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                y_prob = y_pred  # Fallback

        # Compute metrics for the fold
        acc_scores.append(accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_test, y_pred, zero_division=0))

        try:
            roc_auc_scores.append(roc_auc_score(y_test, y_prob))
        except ValueError:
            roc_auc_scores.append(0)

        # Update aggregate confusion matrix
        conf_matrix_total += confusion_matrix(y_test, y_pred, labels=[0,1])

    # Average metrics
    avg_accuracy = np.mean(acc_scores)
    avg_precision = np.mean(prec_scores)
    avg_recall = np.mean(recall_scores)
    avg_roc_auc = np.mean(roc_auc_scores)

    # Store results
    results['Model'].append(name)
    results['Accuracy'].append(avg_accuracy)
    results['Precision'].append(avg_precision)
    results['Recall'].append(avg_recall)
    results['ROC-AUC'].append(avg_roc_auc)
    results['Confusion Matrix'].append(conf_matrix_total)

# Convert results to DataFrame for easy viewing
results_df = pd.DataFrame({
    'Model': results['Model'],
    'Accuracy': results['Accuracy'],
    'Precision': results['Precision'],
    'Recall': results['Recall'],
    'ROC-AUC': results['ROC-AUC']
})

print("Model Performance Metrics:")
print(results_df)

print("\nDone")

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a random seed between 1 and 100 for this run
RANDOM_SEED = random.randint(1, 100)
print(f"Using random seed: {RANDOM_SEED}")

# Load the cleaned dataset
file_path = "/content/Transformed_Bakery_Sales.xlsx"
df = pd.read_excel(file_path)

# Split 90% for training & testing, 10% as unseen data
train_test_data, unseen_data = train_test_split(df, test_size=0.10, random_state=RANDOM_SEED)

# Further split train_test_data into 80% training and 20% testing
train_data, test_data = train_test_split(train_test_data, test_size=0.20, random_state=RANDOM_SEED)

# Save the datasets
train_data_path = "Train_Data.xlsx"
test_data_path = "Test_Data.xlsx"
unseen_data_path = "Unseen_Data.xlsx"

# Save the datasets (uncomment if you need a copy of a randomized dataset)
train_data.to_excel(train_data_path, index=False)
test_data.to_excel(test_data_path, index=False)
unseen_data.to_excel(unseen_data_path, index=False)

print(f"Data split and saved successfully.")
print(f"Training data: {train_data.shape[0]} samples")
print(f"Testing data: {test_data.shape[0]} samples")
print(f"Unseen data: {unseen_data.shape[0]} samples")

# Preprocess function for consistent feature engineering
def preprocess_data(df):
    # Create count features
    def count_items(s):
        if pd.isnull(s):
            return 0
        return len(str(s).split(','))

    # Create new count features
    processed_df = df.copy()
    processed_df['bread_count'] = processed_df['Breads'].apply(count_items)
    processed_df['beverage_count'] = processed_df['Beverages'].apply(count_items)
    processed_df['dessert_count'] = processed_df['Desserts'].apply(count_items)

    # Create binary target
    median_total = df['total'].median()
    processed_df['high_sale'] = (processed_df['total'] > median_total).astype(int)

    # One-hot encode day of the week
    day_dummies = pd.get_dummies(processed_df['day of week'], prefix='day')

    # Combine with count features
    features = pd.concat([day_dummies, processed_df[['bread_count', 'beverage_count', 'dessert_count']]], axis=1)

    return features, processed_df['high_sale']

# Preprocess all datasets
X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)
X_unseen, y_unseen = preprocess_data(unseen_data)

# Initialize models
models = {
    'SVM': SVC(probability=True, random_state=RANDOM_SEED),
}

# Setup 10-fold cross-validation for unseen data only
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

# Results dictionary
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'ROC-AUC': [],
    'Confusion Matrix': []
}

# Evaluate each model using 10-fold CV on the unseen data
for name, model in models.items():
    print(f"Evaluating {name} on unseen data with 10-fold CV...")

    # Storage for metrics
    acc_scores = []
    prec_scores = []
    recall_scores = []
    roc_auc_scores = []

    # Confusion matrix aggregate
    conf_matrix_total = np.zeros((2, 2), dtype=int)

    # Perform 10-fold cross-validation on unseen data
    for train_index, test_index in skf.split(X_unseen, y_unseen):
        X_cv_train, X_cv_test = X_unseen.iloc[train_index], X_unseen.iloc[test_index]
        y_cv_train, y_cv_test = y_unseen.iloc[train_index], y_unseen.iloc[test_index]

        # Train the model on a fold of the unseen data
        model.fit(X_cv_train, y_cv_train)

        # Predict on the test fold
        y_pred = model.predict(X_cv_test)

        # For ROC-AUC, we need probability estimates
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_cv_test)[:, 1]
        else:
            # For models without predict_proba, use decision_function if available
            if hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_cv_test)
            else:
                y_prob = y_pred  # Fallback

        # Compute metrics for the fold
        acc_scores.append(accuracy_score(y_cv_test, y_pred))
        prec_scores.append(precision_score(y_cv_test, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_cv_test, y_pred, zero_division=0))

        try:
            roc_auc_scores.append(roc_auc_score(y_cv_test, y_prob))
        except ValueError:
            roc_auc_scores.append(0)

        # Update aggregate confusion matrix
        conf_matrix_total += confusion_matrix(y_cv_test, y_pred, labels=[0,1])

        # Print confusion matrix in text format
        print(f"\nConfusion Matrix for {name}")
        print(f"True Negatives: {conf_matrix_total[0, 0]}, False Positives: {conf_matrix_total[0, 1]}")
        print(f"False Negatives: {conf_matrix_total[1, 0]}, True Positives: {conf_matrix_total[1, 1]}\n")


    # Average metrics
    avg_accuracy = np.mean(acc_scores)
    avg_precision = np.mean(prec_scores)
    avg_recall = np.mean(recall_scores)
    avg_roc_auc = np.mean(roc_auc_scores)

    # Store results
    results['Model'].append(name)
    results['Accuracy'].append(avg_accuracy)
    results['Precision'].append(avg_precision)
    results['Recall'].append(avg_recall)
    results['ROC-AUC'].append(avg_roc_auc)
    results['Confusion Matrix'].append(conf_matrix_total)

# Convert results to DataFrame for easy viewing
results_df = pd.DataFrame({
    'Model': results['Model'],
    'Accuracy': results['Accuracy'],
    'Precision': results['Precision'],
    'Recall': results['Recall'],
    'ROC-AUC': results['ROC-AUC']
})

print("\nModel Performance Metrics on Unseen Data (10-fold Cross-Validation):")
print(results_df.round(3))

# Plot confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (name, cm) in enumerate(zip(results['Model'], results['Confusion Matrix'])):
    if i < len(axes):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - {name}', fontsize=16, pad=15)
        axes[i].set_xlabel('Predicted Label', fontsize=14, labelpad=10)
        axes[i].set_ylabel('True Label', fontsize=14, labelpad=10)
        axes[i].tick_params(labelsize=12)

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('unseen_data_confusion_matrices.png')
plt.show()

# Plot performance metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC']
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(results['Model']))
width = 0.2
multiplier = 0

for metric in metrics:
    offset = width * multiplier
    rects = ax.bar(x + offset, results_df[metric], width, label=metric)
    multiplier += 1

ax.set_title('Model Performance on Unseen Data (10-fold CV)', fontsize=20, pad=15)
ax.set_xticks(x + width, results['Model'], fontsize=14)
ax.set_ylabel('Score', fontsize=16, labelpad=10)
ax.set_ylim(0, 1.0)
ax.legend(loc='lower right', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('unseen_data_model_comparison.png')
plt.show()

# Identify best performing model based on ROC-AUC
best_model_idx = results_df['ROC-AUC'].idxmax()
best_model = results_df.iloc[best_model_idx]['Model']
best_roc_auc = results_df.iloc[best_model_idx]['ROC-AUC']

print(f"\nBest performing model on unseen data (10-fold CV): {best_model}")
print(f"ROC-AUC Score: {best_roc_auc:.3f}")

print("\nDone")



#Finalized with the Product prediction.


import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a random seed between 1 and 100 for this run
RANDOM_SEED = random.randint(1, 100)
print(f"Using random seed: {RANDOM_SEED}")

# Load the cleaned dataset
file_path = "/content/Transformed_Bakery_Sales.xlsx"
df = pd.read_excel(file_path)

# Split 90% for training & testing, 10% as unseen data
train_test_data, unseen_data = train_test_split(df, test_size=0.10, random_state=RANDOM_SEED)

# Further split train_test_data into 80% training and 20% testing
train_data, test_data = train_test_split(train_test_data, test_size=0.20, random_state=RANDOM_SEED)

# Save the datasets
train_data_path = "Train_Data.xlsx"
test_data_path = "Test_Data.xlsx"
unseen_data_path = "Unseen_Data.xlsx"

# Save the datasets (uncomment if you need a copy of a randomized dataset)
train_data.to_excel(train_data_path, index=False)
test_data.to_excel(test_data_path, index=False)
unseen_data.to_excel(unseen_data_path, index=False)

print(f"Data split and saved successfully.")
print(f"Training data: {train_data.shape[0]} samples")
print(f"Testing data: {test_data.shape[0]} samples")
print(f"Unseen data: {unseen_data.shape[0]} samples")

# Preprocess function for consistent feature engineering
def preprocess_data(df):
    # Create count features
    def count_items(s):
        if pd.isnull(s):
            return 0
        return len(str(s).split(','))

    # Create new count features
    processed_df = df.copy()
    processed_df['bread_count'] = processed_df['Breads'].apply(count_items)
    processed_df['beverage_count'] = processed_df['Beverages'].apply(count_items)
    processed_df['dessert_count'] = processed_df['Desserts'].apply(count_items)

    # Create binary target
    median_total = df['total'].median()
    processed_df['high_sale'] = (processed_df['total'] > median_total).astype(int)

    # One-hot encode day of the week
    day_dummies = pd.get_dummies(processed_df['day of week'], prefix='day')

    # Combine with count features
    features = pd.concat([day_dummies, processed_df[['bread_count', 'beverage_count', 'dessert_count']]], axis=1)

    return features, processed_df['high_sale']

# Preprocess all datasets
X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)
X_unseen, y_unseen = preprocess_data(unseen_data)

# Initialize models
models = {
    'SVM': SVC(probability=True, random_state=RANDOM_SEED),
}

# Setup 10-fold cross-validation for unseen data only
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

# Keep your original results dictionary structure
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'ROC-AUC': [],
    'Confusion Matrix': []
}

# Add a parallel dictionary for training results
train_results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'ROC-AUC': [],
    'Confusion Matrix': []
}

# Evaluate each model using 10-fold CV on the unseen data (your existing code)
for name, model in models.items():
    print(f"Evaluating {name} on unseen data with 10-fold CV...")

    # Your existing code for unseen data evaluation...
    # (Keep all your current code for unseen data evaluation)

    # Store unseen data results - keep this as is
    results['Model'].append(name)
    results['Accuracy'].append(avg_accuracy)
    results['Precision'].append(avg_precision)
    results['Recall'].append(avg_recall)
    results['ROC-AUC'].append(avg_roc_auc)
    results['Confusion Matrix'].append(conf_matrix_total)

    # Now evaluate on training data with 10-fold CV
    print(f"Evaluating {name} on training data with 10-fold CV...")

    # Storage for train metrics
    train_acc_scores = []
    train_prec_scores = []
    train_recall_scores = []
    train_roc_auc_scores = []

    # Confusion matrix aggregate for train data
    train_conf_matrix_total = np.zeros((2, 2), dtype=int)

    # Perform 10-fold cross-validation on train data
    for train_index, test_index in skf.split(X_train, y_train):
        X_cv_train, X_cv_test = X_train.iloc[train_index], X_train.iloc[test_index]
        y_cv_train, y_cv_test = y_train.iloc[train_index], y_train.iloc[test_index]

        # Train the model on a fold of the train data
        model.fit(X_cv_train, y_cv_train)

        # Predict on the test fold
        y_pred = model.predict(X_cv_test)

        # For ROC-AUC, we need probability estimates
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_cv_test)[:, 1]
        else:
            # For models without predict_proba, use decision_function if available
            if hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_cv_test)
            else:
                y_prob = y_pred  # Fallback

        # Compute metrics for the fold
        train_acc_scores.append(accuracy_score(y_cv_test, y_pred))
        train_prec_scores.append(precision_score(y_cv_test, y_pred, zero_division=0))
        train_recall_scores.append(recall_score(y_cv_test, y_pred, zero_division=0))

        try:
            train_roc_auc_scores.append(roc_auc_score(y_cv_test, y_prob))
        except ValueError:
            train_roc_auc_scores.append(0)

        # Update aggregate confusion matrix
        train_conf_matrix_total += confusion_matrix(y_cv_test, y_pred, labels=[0,1])

    # Average metrics for train data
    train_avg_accuracy = np.mean(train_acc_scores)
    train_avg_precision = np.mean(train_prec_scores)
    train_avg_recall = np.mean(train_recall_scores)
    train_avg_roc_auc = np.mean(train_roc_auc_scores)

    # Store train results
    train_results['Model'].append(name)
    train_results['Accuracy'].append(train_avg_accuracy)
    train_results['Precision'].append(train_avg_precision)
    train_results['Recall'].append(train_avg_recall)
    train_results['ROC-AUC'].append(train_avg_roc_auc)
    train_results['Confusion Matrix'].append(train_conf_matrix_total)

    # Print train confusion matrix in text format
    print(f"\nTrain Confusion Matrix for {name}")
    print(f"True Negatives: {train_conf_matrix_total[0, 0]}, False Positives: {train_conf_matrix_total[0, 1]}")
    print(f"False Negatives: {train_conf_matrix_total[1, 0]}, True Positives: {train_conf_matrix_total[1, 1]}\n")

    # Print unseen confusion matrix in text format
    print(f"\nUnseen Confusion Matrix for {name}")
    print(f"True Negatives: {conf_matrix_total[0, 0]}, False Positives: {conf_matrix_total[0, 1]}")
    print(f"False Negatives: {conf_matrix_total[1, 0]}, True Positives: {conf_matrix_total[1, 1]}\n")

# Keep your original results DataFrame
results_df = pd.DataFrame({
    'Model': results['Model'],
    'Accuracy': results['Accuracy'],
    'Precision': results['Precision'],
    'Recall': results['Recall'],
    'ROC-AUC': results['ROC-AUC']
})

# Create a DataFrame for training results
train_results_df = pd.DataFrame({
    'Model': train_results['Model'],
    'Accuracy': train_results['Accuracy'],
    'Precision': train_results['Precision'],
    'Recall': train_results['Recall'],
    'ROC-AUC': train_results['ROC-AUC']
})

# Print both results
print("\nModel Performance Metrics on Unseen Data (10-fold Cross-Validation):")
print(results_df.round(3))

print("\nModel Performance Metrics on Training Data (10-fold Cross-Validation):")
print(train_results_df.round(3))

# Plot confusion matrices for both datasets
# First for unseen data (your existing code)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (name, cm) in enumerate(zip(results['Model'], results['Confusion Matrix'])):
    if i < len(axes):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix (Unseen) - {name}', fontsize=16, pad=15)
        axes[i].set_xlabel('Predicted Label', fontsize=14, labelpad=10)
        axes[i].set_ylabel('True Label', fontsize=14, labelpad=10)
        axes[i].tick_params(labelsize=12)

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('unseen_data_confusion_matrices.png')
plt.show()

# Now for training data
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (name, cm) in enumerate(zip(train_results['Model'], train_results['Confusion Matrix'])):
    if i < len(axes):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix (Train) - {name}', fontsize=16, pad=15)
        axes[i].set_xlabel('Predicted Label', fontsize=14, labelpad=10)
        axes[i].set_ylabel('True Label', fontsize=14, labelpad=10)
        axes[i].tick_params(labelsize=12)

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('train_data_confusion_matrices.png')
plt.show()

# Plot performance metrics comparison for both datasets
metrics = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC']

# Unseen data (your existing code)
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(results['Model']))
width = 0.2
multiplier = 0

for metric in metrics:
    offset = width * multiplier
    rects = ax.bar(x + offset, results_df[metric], width, label=metric)
    multiplier += 1

ax.set_title('Model Performance on Unseen Data (10-fold CV)', fontsize=20, pad=15)
ax.set_xticks(x + width, results['Model'], fontsize=14)
ax.set_ylabel('Score', fontsize=16, labelpad=10)
ax.set_ylim(0, 1.0)
ax.legend(loc='lower right', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('unseen_data_model_comparison.png')
plt.show()

# Training data
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(train_results['Model']))
width = 0.2
multiplier = 0

for metric in metrics:
    offset = width * multiplier
    rects = ax.bar(x + offset, train_results_df[metric], width, label=metric)
    multiplier += 1

ax.set_title('Model Performance on Training Data (10-fold CV)', fontsize=20, pad=15)
ax.set_xticks(x + width, train_results['Model'], fontsize=14)
ax.set_ylabel('Score', fontsize=16, labelpad=10)
ax.set_ylim(0, 1.0)
ax.legend(loc='lower right', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('train_data_model_comparison.png')
plt.show()

# Keep your original best model identification
best_model_idx = results_df['ROC-AUC'].idxmax()
best_model = results_df.iloc[best_model_idx]['Model']
best_roc_auc = results_df.iloc[best_model_idx]['ROC-AUC']

print(f"\nBest performing model on unseen data (10-fold CV): {best_model}")
print(f"ROC-AUC Score: {best_roc_auc:.3f}")

# Add best model identification for training data
best_train_model_idx = train_results_df['ROC-AUC'].idxmax()
best_train_model = train_results_df.iloc[best_train_model_idx]['Model']
best_train_roc_auc = train_results_df.iloc[best_train_model_idx]['ROC-AUC']

print(f"\nBest performing model on training data (10-fold CV): {best_train_model}")
print(f"ROC-AUC Score: {best_train_roc_auc:.3f}")

#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
# Read the updated file with rule-based categories
classified_bakery_sales = pd.read_excel("/content/Classified_Bakery_Sales updated.xlsx")

# Count occurrences of each rule-based category
category_counts = classified_bakery_sales['Rule_Based_Category'].value_counts()
print("\nRule-Based Category Counts:")
print(category_counts)

# Preprocess the classified bakery sales data
def preprocess_classified_data(df):
    # Create count features
    def count_items(s):
        if pd.isnull(s):
            return 0
        return len(str(s).split(','))

    # Create new count features
    processed_df = df.copy()
    processed_df['bread_count'] = processed_df['Breads'].apply(count_items)
    processed_df['beverage_count'] = processed_df['Beverages'].apply(count_items)
    processed_df['dessert_count'] = processed_df['Desserts'].apply(count_items)

    # One-hot encode day of the week
    day_dummies = pd.get_dummies(processed_df['day of week'], prefix='day')

    # Combine with count features
    features = pd.concat([day_dummies, processed_df[['bread_count', 'beverage_count', 'dessert_count']]], axis=1)

    return features

# Generate features for classified data
X_classified = preprocess_classified_data(classified_bakery_sales)

# Initialize models for category prediction
print("\n" + "="*50)
print("RULE-BASED CATEGORY PREDICTION")
print("="*50)

category_models = {
    'SVM': SVC(probability=True, random_state=RANDOM_SEED),
}

# Storage for category prediction results
category_results = {
    'Category': [],
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'ROC-AUC': [],
    'Confusion Matrix': []
}

# Evaluate each category with 10-fold cross-validation
for category in category_counts.index:
    print(f"\n{'='*60}")
    print(f"Evaluating rule-based category prediction for: {category}")
    print(f"{'='*60}")

    for model_name, base_model in category_models.items():
        print(f"Evaluating {model_name} with 10-fold CV...")

        # Get targets for this category
        y_category = (classified_bakery_sales['Rule_Based_Category'] == category).astype(int)

        # Skip if there's only one class present
        if len(y_category.unique()) == 1:
            print(f"Skipping {category} - insufficient class diversity")
            continue

        acc_scores = []
        prec_scores = []
        recall_scores = []
        roc_auc_scores = []
        conf_matrix_total = np.zeros((2, 2), dtype=int)

        # Perform 10-fold cross-validation
        for train_idx, val_idx in skf.split(X_classified, y_category):
            X_fold_train, X_fold_val = X_classified.iloc[train_idx], X_classified.iloc[val_idx]
            y_fold_train, y_fold_val = y_category.iloc[train_idx], y_category.iloc[val_idx]

            model = base_model.__class__(**base_model.get_params())
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_fold_val)[:, 1]
            else:
                y_prob = model.decision_function(X_fold_val) if hasattr(model, "decision_function") else y_pred

            acc_scores.append(accuracy_score(y_fold_val, y_pred))
            prec_scores.append(precision_score(y_fold_val, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_fold_val, y_pred, zero_division=0))

            try:
                roc_auc_scores.append(roc_auc_score(y_fold_val, y_prob))
            except ValueError:
                # Handle case where there might be only one class in the fold
                roc_auc_scores.append(np.nan)

            conf_matrix_total += confusion_matrix(y_fold_val, y_pred, labels=[0,1])

        # Store results
        category_results['Category'].append(category)
        category_results['Model'].append(model_name)
        category_results['Accuracy'].append(np.mean(acc_scores))
        category_results['Precision'].append(np.nanmean(prec_scores))
        category_results['Recall'].append(np.nanmean(recall_scores))
        category_results['ROC-AUC'].append(np.nanmean(roc_auc_scores))
        category_results['Confusion Matrix'].append(conf_matrix_total)

        print(f"\nPerformance Metrics for {category} with {model_name} (10-fold CV):")
        print(f"Accuracy: {np.mean(acc_scores):.3f}")
        print(f"Precision: {np.nanmean(prec_scores):.3f}")
        print(f"Recall: {np.nanmean(recall_scores):.3f}")
        print(f"ROC-AUC: {np.nanmean(roc_auc_scores):.3f}")

# Create and display results DataFrame
category_results_df = pd.DataFrame(category_results)
print("\nRule-Based Category Prediction Summary:")
print(category_results_df)#Finalized with the Product prediction.


import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a random seed between 1 and 100 for this run
RANDOM_SEED = random.randint(1, 100)
print(f"Using random seed: {RANDOM_SEED}")

# Load the cleaned dataset
file_path = "/content/Transformed_Bakery_Sales.xlsx"
df = pd.read_excel(file_path)

# Split 90% for training & testing, 10% as unseen data
train_test_data, unseen_data = train_test_split(df, test_size=0.10, random_state=RANDOM_SEED)

# Further split train_test_data into 80% training and 20% testing
train_data, test_data = train_test_split(train_test_data, test_size=0.20, random_state=RANDOM_SEED)

# Save the datasets
train_data_path = "Train_Data.xlsx"
test_data_path = "Test_Data.xlsx"
unseen_data_path = "Unseen_Data.xlsx"

# Save the datasets (uncomment if you need a copy of a randomized dataset)
train_data.to_excel(train_data_path, index=False)
test_data.to_excel(test_data_path, index=False)
unseen_data.to_excel(unseen_data_path, index=False)

print(f"Data split and saved successfully.")
print(f"Training data: {train_data.shape[0]} samples")
print(f"Testing data: {test_data.shape[0]} samples")
print(f"Unseen data: {unseen_data.shape[0]} samples")

# Preprocess function for consistent feature engineering
def preprocess_data(df):
    # Create count features
    def count_items(s):
        if pd.isnull(s):
            return 0
        return len(str(s).split(','))

    # Create new count features
    processed_df = df.copy()
    processed_df['bread_count'] = processed_df['Breads'].apply(count_items)
    processed_df['beverage_count'] = processed_df['Beverages'].apply(count_items)
    processed_df['dessert_count'] = processed_df['Desserts'].apply(count_items)

    # Create binary target
    median_total = df['total'].median()
    processed_df['high_sale'] = (processed_df['total'] > median_total).astype(int)

    # One-hot encode day of the week
    day_dummies = pd.get_dummies(processed_df['day of week'], prefix='day')

    # Combine with count features
    features = pd.concat([day_dummies, processed_df[['bread_count', 'beverage_count', 'dessert_count']]], axis=1)

    return features, processed_df['high_sale']

# Preprocess all datasets
X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)
X_unseen, y_unseen = preprocess_data(unseen_data)

# Initialize models
models = {
    'SVM': SVC(probability=True, random_state=RANDOM_SEED),
}

# Setup 10-fold cross-validation for unseen data only
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

# Keep your original results dictionary structure
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'ROC-AUC': [],
    'Confusion Matrix': []
}

# Add a parallel dictionary for training results
train_results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'ROC-AUC': [],
    'Confusion Matrix': []
}

# Evaluate each model using 10-fold CV on the unseen data (your existing code)
for name, model in models.items():
    print(f"Evaluating {name} on unseen data with 10-fold CV...")

    # Your existing code for unseen data evaluation...
    # (Keep all your current code for unseen data evaluation)

    # Store unseen data results - keep this as is
    results['Model'].append(name)
    results['Accuracy'].append(avg_accuracy)
    results['Precision'].append(avg_precision)
    results['Recall'].append(avg_recall)
    results['ROC-AUC'].append(avg_roc_auc)
    results['Confusion Matrix'].append(conf_matrix_total)

    # Now evaluate on training data with 10-fold CV
    print(f"Evaluating {name} on training data with 10-fold CV...")

    # Storage for train metrics
    train_acc_scores = []
    train_prec_scores = []
    train_recall_scores = []
    train_roc_auc_scores = []

    # Confusion matrix aggregate for train data
    train_conf_matrix_total = np.zeros((2, 2), dtype=int)

    # Perform 10-fold cross-validation on train data
    for train_index, test_index in skf.split(X_train, y_train):
        X_cv_train, X_cv_test = X_train.iloc[train_index], X_train.iloc[test_index]
        y_cv_train, y_cv_test = y_train.iloc[train_index], y_train.iloc[test_index]

        # Train the model on a fold of the train data
        model.fit(X_cv_train, y_cv_train)

        # Predict on the test fold
        y_pred = model.predict(X_cv_test)

        # For ROC-AUC, we need probability estimates
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_cv_test)[:, 1]
        else:
            # For models without predict_proba, use decision_function if available
            if hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_cv_test)
            else:
                y_prob = y_pred  # Fallback

        # Compute metrics for the fold
        train_acc_scores.append(accuracy_score(y_cv_test, y_pred))
        train_prec_scores.append(precision_score(y_cv_test, y_pred, zero_division=0))
        train_recall_scores.append(recall_score(y_cv_test, y_pred, zero_division=0))

        try:
            train_roc_auc_scores.append(roc_auc_score(y_cv_test, y_prob))
        except ValueError:
            train_roc_auc_scores.append(0)

        # Update aggregate confusion matrix
        train_conf_matrix_total += confusion_matrix(y_cv_test, y_pred, labels=[0,1])

    # Average metrics for train data
    train_avg_accuracy = np.mean(train_acc_scores)
    train_avg_precision = np.mean(train_prec_scores)
    train_avg_recall = np.mean(train_recall_scores)
    train_avg_roc_auc = np.mean(train_roc_auc_scores)

    # Store train results
    train_results['Model'].append(name)
    train_results['Accuracy'].append(train_avg_accuracy)
    train_results['Precision'].append(train_avg_precision)
    train_results['Recall'].append(train_avg_recall)
    train_results['ROC-AUC'].append(train_avg_roc_auc)
    train_results['Confusion Matrix'].append(train_conf_matrix_total)

    # Print train confusion matrix in text format
    print(f"\nTrain Confusion Matrix for {name}")
    print(f"True Negatives: {train_conf_matrix_total[0, 0]}, False Positives: {train_conf_matrix_total[0, 1]}")
    print(f"False Negatives: {train_conf_matrix_total[1, 0]}, True Positives: {train_conf_matrix_total[1, 1]}\n")

    # Print unseen confusion matrix in text format
    print(f"\nUnseen Confusion Matrix for {name}")
    print(f"True Negatives: {conf_matrix_total[0, 0]}, False Positives: {conf_matrix_total[0, 1]}")
    print(f"False Negatives: {conf_matrix_total[1, 0]}, True Positives: {conf_matrix_total[1, 1]}\n")

# Keep your original results DataFrame
results_df = pd.DataFrame({
    'Model': results['Model'],
    'Accuracy': results['Accuracy'],
    'Precision': results['Precision'],
    'Recall': results['Recall'],
    'ROC-AUC': results['ROC-AUC']
})

# Create a DataFrame for training results
train_results_df = pd.DataFrame({
    'Model': train_results['Model'],
    'Accuracy': train_results['Accuracy'],
    'Precision': train_results['Precision'],
    'Recall': train_results['Recall'],
    'ROC-AUC': train_results['ROC-AUC']
})

# Print both results
print("\nModel Performance Metrics on Unseen Data (10-fold Cross-Validation):")
print(results_df.round(3))

print("\nModel Performance Metrics on Training Data (10-fold Cross-Validation):")
print(train_results_df.round(3))

# Plot confusion matrices for both datasets
# First for unseen data (your existing code)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (name, cm) in enumerate(zip(results['Model'], results['Confusion Matrix'])):
    if i < len(axes):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix (Unseen) - {name}', fontsize=16, pad=15)
        axes[i].set_xlabel('Predicted Label', fontsize=14, labelpad=10)
        axes[i].set_ylabel('True Label', fontsize=14, labelpad=10)
        axes[i].tick_params(labelsize=12)

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('unseen_data_confusion_matrices.png')
plt.show()

# Now for training data
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (name, cm) in enumerate(zip(train_results['Model'], train_results['Confusion Matrix'])):
    if i < len(axes):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix (Train) - {name}', fontsize=16, pad=15)
        axes[i].set_xlabel('Predicted Label', fontsize=14, labelpad=10)
        axes[i].set_ylabel('True Label', fontsize=14, labelpad=10)
        axes[i].tick_params(labelsize=12)

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('train_data_confusion_matrices.png')
plt.show()

# Plot performance metrics comparison for both datasets
metrics = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC']

# Unseen data (your existing code)
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(results['Model']))
width = 0.2
multiplier = 0

for metric in metrics:
    offset = width * multiplier
    rects = ax.bar(x + offset, results_df[metric], width, label=metric)
    multiplier += 1

ax.set_title('Model Performance on Unseen Data (10-fold CV)', fontsize=20, pad=15)
ax.set_xticks(x + width, results['Model'], fontsize=14)
ax.set_ylabel('Score', fontsize=16, labelpad=10)
ax.set_ylim(0, 1.0)
ax.legend(loc='lower right', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('unseen_data_model_comparison.png')
plt.show()

# Training data
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(train_results['Model']))
width = 0.2
multiplier = 0

for metric in metrics:
    offset = width * multiplier
    rects = ax.bar(x + offset, train_results_df[metric], width, label=metric)
    multiplier += 1

ax.set_title('Model Performance on Training Data (10-fold CV)', fontsize=20, pad=15)
ax.set_xticks(x + width, train_results['Model'], fontsize=14)
ax.set_ylabel('Score', fontsize=16, labelpad=10)
ax.set_ylim(0, 1.0)
ax.legend(loc='lower right', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('train_data_model_comparison.png')
plt.show()

# Keep your original best model identification
best_model_idx = results_df['ROC-AUC'].idxmax()
best_model = results_df.iloc[best_model_idx]['Model']
best_roc_auc = results_df.iloc[best_model_idx]['ROC-AUC']

print(f"\nBest performing model on unseen data (10-fold CV): {best_model}")
print(f"ROC-AUC Score: {best_roc_auc:.3f}")

# Add best model identification for training data
best_train_model_idx = train_results_df['ROC-AUC'].idxmax()
best_train_model = train_results_df.iloc[best_train_model_idx]['Model']
best_train_roc_auc = train_results_df.iloc[best_train_model_idx]['ROC-AUC']

print(f"\nBest performing model on training data (10-fold CV): {best_train_model}")
print(f"ROC-AUC Score: {best_train_roc_auc:.3f}")

#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
# Read the updated file with rule-based categories
classified_bakery_sales = pd.read_excel("/content/Classified_Bakery_Sales updated.xlsx")

# Count occurrences of each rule-based category
category_counts = classified_bakery_sales['Rule_Based_Category'].value_counts()
print("\nRule-Based Category Counts:")
print(category_counts)

# Preprocess the classified bakery sales data
def preprocess_classified_data(df):
    # Create count features
    def count_items(s):
        if pd.isnull(s):
            return 0
        return len(str(s).split(','))

    # Create new count features
    processed_df = df.copy()
    processed_df['bread_count'] = processed_df['Breads'].apply(count_items)
    processed_df['beverage_count'] = processed_df['Beverages'].apply(count_items)
    processed_df['dessert_count'] = processed_df['Desserts'].apply(count_items)

    # One-hot encode day of the week
    day_dummies = pd.get_dummies(processed_df['day of week'], prefix='day')

    # Combine with count features
    features = pd.concat([day_dummies, processed_df[['bread_count', 'beverage_count', 'dessert_count']]], axis=1)

    return features

# Generate features for classified data
X_classified = preprocess_classified_data(classified_bakery_sales)

# Initialize models for category prediction
print("\n" + "="*50)
print("RULE-BASED CATEGORY PREDICTION")
print("="*50)

category_models = {
    'SVM': SVC(probability=True, random_state=RANDOM_SEED),
}

# Storage for category prediction results
category_results = {
    'Category': [],
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'ROC-AUC': [],
    'Confusion Matrix': []
}

# Evaluate each category with 10-fold cross-validation
for category in category_counts.index:
    print(f"\n{'='*60}")
    print(f"Evaluating rule-based category prediction for: {category}")
    print(f"{'='*60}")

    for model_name, base_model in category_models.items():
        print(f"Evaluating {model_name} with 10-fold CV...")

        # Get targets for this category
        y_category = (classified_bakery_sales['Rule_Based_Category'] == category).astype(int)

        # Skip if there's only one class present
        if len(y_category.unique()) == 1:
            print(f"Skipping {category} - insufficient class diversity")
            continue

        acc_scores = []
        prec_scores = []
        recall_scores = []
        roc_auc_scores = []
        conf_matrix_total = np.zeros((2, 2), dtype=int)

        # Perform 10-fold cross-validation
        for train_idx, val_idx in skf.split(X_classified, y_category):
            X_fold_train, X_fold_val = X_classified.iloc[train_idx], X_classified.iloc[val_idx]
            y_fold_train, y_fold_val = y_category.iloc[train_idx], y_category.iloc[val_idx]

            model = base_model.__class__(**base_model.get_params())
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_fold_val)[:, 1]
            else:
                y_prob = model.decision_function(X_fold_val) if hasattr(model, "decision_function") else y_pred

            acc_scores.append(accuracy_score(y_fold_val, y_pred))
            prec_scores.append(precision_score(y_fold_val, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_fold_val, y_pred, zero_division=0))

            try:
                roc_auc_scores.append(roc_auc_score(y_fold_val, y_prob))
            except ValueError:
                # Handle case where there might be only one class in the fold
                roc_auc_scores.append(np.nan)

            conf_matrix_total += confusion_matrix(y_fold_val, y_pred, labels=[0,1])

        # Store results
        category_results['Category'].append(category)
        category_results['Model'].append(model_name)
        category_results['Accuracy'].append(np.mean(acc_scores))
        category_results['Precision'].append(np.nanmean(prec_scores))
        category_results['Recall'].append(np.nanmean(recall_scores))
        category_results['ROC-AUC'].append(np.nanmean(roc_auc_scores))
        category_results['Confusion Matrix'].append(conf_matrix_total)

        print(f"\nPerformance Metrics for {category} with {model_name} (10-fold CV):")
        print(f"Accuracy: {np.mean(acc_scores):.3f}")
        print(f"Precision: {np.nanmean(prec_scores):.3f}")
        print(f"Recall: {np.nanmean(recall_scores):.3f}")
        print(f"ROC-AUC: {np.nanmean(roc_auc_scores):.3f}")

# Create and display results DataFrame
category_results_df = pd.DataFrame(category_results)
print("\nRule-Based Category Prediction Summary:")
print(category_results_df)

# Initialize models for product prediction
print("\n" + "="*50)
print("PRODUCT SOLD PREDICTION")
print("="*50)

product_models = {
    'SVM': SVC(probability=True, random_state=RANDOM_SEED),  # Changed to SVM with probability=True
}

# Storage for product prediction results
product_results = {
    'Product': [],
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'ROC-AUC': [],  # Added ROC-AUC here
    'Confusion Matrix': []
}

# Evaluate each product with 10-fold cross-validation
for product in top_products:
    print(f"\n{'='*60}")
    print(f"Evaluating product prediction for: {product}")
    print(f"{'='*60}")

    for model_name, base_model in product_models.items():
        print(f"Evaluating {model_name} with 10-fold CV...")

        # Get targets for this product
        y_merged_product = y_merged_products[product]

        # Skip if there's only one class present
        if len(y_merged_product.unique()) == 1:
            print(f"Skipping {product} - insufficient class diversity")
            continue

        # Storage for metrics
        acc_scores = []
        prec_scores = []
        recall_scores = []
        roc_auc_scores = []  # Added for ROC-AUC

        # Confusion matrix aggregate
        conf_matrix_total = np.zeros((2, 2), dtype=int)

        # Perform 10-fold cross-validation
        for train_idx, val_idx in skf.split(X_merged, y_merged_product):
            X_fold_train, X_fold_val = X_merged.iloc[train_idx], X_merged.iloc[val_idx]
            y_fold_train, y_fold_val = y_merged_product.iloc[train_idx], y_merged_product.iloc[val_idx]

            # Train the model
            model = base_model.__class__(**base_model.get_params())
            model.fit(X_fold_train, y_fold_train)

            # Predict
            y_pred = model.predict(X_fold_val)

            # For ROC-AUC score
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_fold_val)[:, 1]
            else:
                # For models without predict_proba
                if hasattr(model, "decision_function"):
                    y_prob = model.decision_function(X_fold_val)
                else:
                    y_prob = y_pred

            # Compute metrics
            acc_scores.append(accuracy_score(y_fold_val, y_pred))

            try:
                prec_scores.append(precision_score(y_fold_val, y_pred, zero_division=0))
                recall_scores.append(recall_score(y_fold_val, y_pred, zero_division=0))
                # Add ROC-AUC score calculation
                if len(np.unique(y_fold_val)) > 1:  # Ensure there are 2 classes for ROC-AUC
                    roc_auc_scores.append(roc_auc_score(y_fold_val, y_prob))
                else:
                    roc_auc_scores.append(np.nan)
            except Exception as e:
                if 'prec_scores' not in locals() or len(prec_scores) < len(acc_scores):
                    prec_scores.append(np.nan)
                if 'recall_scores' not in locals() or len(recall_scores) < len(acc_scores):
                    recall_scores.append(np.nan)
                if 'roc_auc_scores' not in locals() or len(roc_auc_scores) < len(acc_scores):
                    roc_auc_scores.append(np.nan)
                print(f"Error calculating metrics for {product}: {e}")

            # Update confusion matrix
            conf_matrix_total += confusion_matrix(y_fold_val, y_pred, labels=[0,1])

        # Average metrics
        avg_accuracy = np.mean(acc_scores)
        avg_precision = np.mean([x for x in prec_scores if not np.isnan(x)]) if any(not np.isnan(x) for x in prec_scores) else np.nan
        avg_recall = np.mean([x for x in recall_scores if not np.isnan(x)]) if any(not np.isnan(x) for x in recall_scores) else np.nan
        avg_roc_auc = np.mean([x for x in roc_auc_scores if not np.isnan(x)]) if any(not np.isnan(x) for x in roc_auc_scores) else np.nan

        # Store results
        product_results['Product'].append(product)
        product_results['Model'].append(model_name)
        product_results['Accuracy'].append(avg_accuracy)
        product_results['Precision'].append(avg_precision)
        product_results['Recall'].append(avg_recall)
        product_results['ROC-AUC'].append(avg_roc_auc)  # Store ROC-AUC
        product_results['Confusion Matrix'].append(conf_matrix_total)

        # Print performance metrics
        print(f"\nPerformance Metrics for {product} with {model_name} (10-fold CV):")
        print(f"Accuracy: {avg_accuracy:.3f}")
        print(f"Precision: {avg_precision:.3f}")
        print(f"Recall: {avg_recall:.3f}")
        print(f"ROC-AUC: {avg_roc_auc:.3f}")  # Print ROC-AUC

        # Print confusion matrix
        print(f"\nConfusion Matrix for {product} with {model_name}:")
        print(f"True Negatives: {conf_matrix_total[0, 0]}, False Positives: {conf_matrix_total[0, 1]}")
        print(f"False Negatives: {conf_matrix_total[1, 0]}, True Positives: {conf_matrix_total[1, 1]}")

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_total, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Product Prediction Confusion Matrix - {product}', fontsize=16, pad=15)
        plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
        plt.ylabel('True Label', fontsize=14, labelpad=10)
        plt.show()

# Create a DataFrame for product prediction results
product_results_df = pd.DataFrame({
    'Product': product_results['Product'],
    'Model': product_results['Model'],
    'Accuracy': product_results['Accuracy'],
    'Precision': product_results['Precision'],
    'Recall': product_results['Recall'],
    'ROC-AUC': product_results['ROC-AUC']  # Include ROC-AUC in DataFrame
})

# Display results in a well-formatted text table
print("\nProduct Prediction Performance Summary (10-fold Cross-Validation):")
print("="*80)
print(f"{'Product':<20} {'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10}")
print("-"*80)

for product in top_products:
    product_data = product_results_df[product_results_df['Product'] == product]
    if not product_data.empty:
        for i, row in product_data.iterrows():
            print(f"{row['Product']:<20} {row['Model']:<15} {row['Accuracy']:.3f}{' ':>6} {row['Precision']:.3f}{' ':>6} {row['Recall']:.3f}{' ':>6} {row['ROC-AUC']:.3f}{' ':>6}")
print("="*80)

# Create a consolidated table for all products with SVM
print("\nConsolidated Product Prediction Performance (SVM):")
print("="*80)
print(f"{'Product':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10}")
print("-"*80)

svm_results = product_results_df[product_results_df['Model'] == 'SVM']
for _, row in svm_results.iterrows():
    print(f"{row['Product']:<20} {row['Accuracy']:.3f}{' ':>6} {row['Precision']:.3f}{' ':>6} {row['Recall']:.3f}{' ':>6} {row['ROC-AUC']:.3f}{' ':>6}")

print("="*80)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize

# ========================== DATA PREPROCESSING ========================== #
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(columns=['datetime'], inplace=True)
    df.rename(columns={'Sales_Category': 'Income_Category', 'Rule_Based_Category': 'Products_Sold_Category'},
              inplace=True)
    df['Day_Num'].fillna(0, inplace=True)
    return df

data = preprocess_data("Train_Data.csv")
validate = preprocess_data("Unseen_Data.csv")

encoder_products = LabelEncoder()
encoder_income = LabelEncoder()

data['Products_Sold_Category'] = encoder_products.fit_transform(data['Products_Sold_Category'])
validate['Products_Sold_Category'] = encoder_products.transform(validate['Products_Sold_Category'])

data['Income_Category'] = encoder_income.fit_transform(data['Income_Category'])
validate['Income_Category'] = encoder_income.transform(validate['Income_Category'])

data = data[data['Products_Sold_Category'] != 3].reset_index(drop=True)
validate = validate[validate['Products_Sold_Category'] != 3].reset_index(drop=True)

# ========================== TRAINING PROCESS ========================== #

# Define features and target variables
x_products = data[['total', 'Day_Num', 'Breads', 'Beverages', 'Desserts']].values  # Features for Products Sold Category
x_income = data[['total', 'Day_Num']].values  # Features for Income Category (Modified)

y_products = data['Products_Sold_Category'].values
y_income = data['Income_Category'].values

# Split data into training and test sets
x_train_products, _, y_train_products, _ = train_test_split(x_products, y_products, test_size=0.2, random_state=42, stratify=y_products)
x_train_income, _, y_train_income, _ = train_test_split(x_income, y_income, test_size=0.2, random_state=42, stratify=y_income)

# Train models
model_products = GaussianNB()
model_products.fit(x_train_products, y_train_products)

model_income = GaussianNB()
model_income.fit(x_train_income, y_train_income)

# ========================== CROSS-VALIDATION (10-FOLD) ========================== #
kf = KFold(n_splits=10, shuffle=True, random_state=42)

proba_preds_cv_products = cross_val_predict(model_products, x_train_products, y_train_products, cv=kf, method='predict_proba')
y_pred_cv_products = np.argmax(proba_preds_cv_products, axis=1)

proba_preds_cv_income = cross_val_predict(model_income, x_train_income, y_train_income, cv=kf, method='predict_proba')
y_pred_cv_income = np.argmax(proba_preds_cv_income, axis=1)

# ========================== VALIDATION PROCESS ========================== #

x_validate_products = validate[['total', 'Day_Num', 'Breads', 'Beverages', 'Desserts']].values  # Features for Products Sold Category
x_validate_income = validate[['total', 'Day_Num']].values

y_validate_products = validate['Products_Sold_Category'].values
y_validate_income = validate['Income_Category'].values

proba_preds_val_products = model_products.predict_proba(x_validate_products)
y_pred_val_products = np.argmax(proba_preds_val_products, axis=1)

proba_preds_val_income = model_income.predict_proba(x_validate_income)
y_pred_val_income = np.argmax(proba_preds_val_income, axis=1)

# ========================== EVALUATION FUNCTIONS ========================== #
def evaluate_model(y_true, y_pred, proba_preds, title):
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    roc_auc = roc_auc_score(label_binarize(y_true, classes=np.unique(y_true)), proba_preds, multi_class='ovr')

    print(f"\n===== {title} Evaluation =====")
    print("Confusion Matrix:\n", conf_matrix)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")

# Cross-validation evaluations
evaluate_model(y_train_products, y_pred_cv_products, proba_preds_cv_products, "Products Sold - 10-Fold Cross-Validation")
evaluate_model(y_train_income, y_pred_cv_income, proba_preds_cv_income, "Income Category - 10-Fold Cross-Validation")

# Validation evaluations
evaluate_model(y_validate_products, y_pred_val_products, proba_preds_val_products, "Products Sold - Validation Set")
evaluate_model(y_validate_income, y_pred_val_income, proba_preds_val_income, "Income Category - Validation Set")

# ========================== CONFUSION MATRIX PLOTS ========================== #
def plot_confusion_matrix(y_true, y_pred, title, class_labels):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))

    ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                     xticklabels=class_labels, yticklabels=class_labels,
                     linewidths=1, linecolor='black')

    # Add external border
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(2)

    plt.xlabel('Predicted Labels', fontsize=8)
    plt.ylabel('True Labels', fontsize=8)
    plt.xticks(rotation=0, fontsize=7)
    plt.yticks(fontsize=5)
    plt.title(f'{title} Confusion Matrix', fontsize=10)

    plt.show()


# Plot confusion matrices with external border
plot_confusion_matrix(y_train_products, y_pred_cv_products, "Products Sold - 10-Fold Cross-Validation",
                      encoder_products.classes_[:-1])
plot_confusion_matrix(y_train_income, y_pred_cv_income, "Income Category - 10-Fold Cross-Validation",
                      encoder_income.classes_)
plot_confusion_matrix(y_validate_products, y_pred_val_products, "Products Sold - Validation Set",
                      encoder_products.classes_[:-1])
plot_confusion_matrix(y_validate_income, y_pred_val_income, "Income Category - Validation Set",
                      encoder_income.classes_)


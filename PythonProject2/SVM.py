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

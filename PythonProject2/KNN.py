import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

# Load data
data = pd.read_csv('TEAM SMURF-MidSummativeAct - Train_Data.csv')

# Encoding categorical features
label_encoders = {}
for col in ['Income Category', 'Product Sold Category']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Encoding day of the week
le = LabelEncoder()
le.fit(data['day of week'])
data['day'] = le.transform(data['day of week'])

# Feature selection
features = ['day', 'Breads', 'Beverages', 'Desserts','total', 'Product Sold Category']
X = data[features]

# Predicting the column
y = data['Income Sold Category']

# Preparing the feature set
X = data[features].fillna(0)

# Standardizing the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prediction
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)  # Get probability estimates

# Binarize the output labels for ROC AUC
y_test_binarized = label_binarize(y_test, classes=range(len(label_encoders['Income Category'].classes_)))

# ROC AUC score (One-vs-Rest for multiclass)
roc_auc = roc_auc_score(y_test_binarized, y_proba, multi_class='ovr')

# Evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted',zero_division=1))
print("Recall:", recall_score(y_test, y_pred, average='weighted',zero_division=1))
print("ROC AUC Score:", roc_auc)

print('\nEncoded Labels Mapping:')
for class_index, label in enumerate(label_encoders['Income Category'].classes_):
    print(f"{class_index}: {label}")
print("\nClassification Report:\n", classification_report(y_test, y_pred,zero_division=1))

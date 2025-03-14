import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, roc_auc_score

data = pd.read_csv('TEAM SMURF-MidSummativeAct - Unseen_Data.csv')

# extracting hour from datetime
data['datetime'] = pd.to_datetime(data['datetime'])
data['hour'] = data['datetime'].dt.hour

# encoding day of the week
le = LabelEncoder()
le.fit(data['day of week'])  # Fit the encoder with the unique values from the data
data['day'] = le.transform(data['day of week'])

# Counting items in Breads and Beverages columns
data['bread'] = data['Breads'].apply(lambda x: len(str(x).split(',')))
data['beverage'] = data['Beverages'].apply(lambda x: len(str(x).split(',')))
data['dessert'] = data['Desserts'].apply(lambda x: len(str(x).split(',')))

# feature selection
features = ['hour', 'day', 'bread', 'beverage','dessert', 'total']
X = data[features]

# creating the target variable Y  for a specific bread type
specific_data = "americano" #ito yung palitan, pick target variable
data['target_variable'] = data['Beverages'].apply(lambda x: 1 if specific_data in str(x).lower() else 0) #palitan yung column name (feature) based sa target variable
y = data['target_variable']

# standardizing the features
scaler = StandardScaler()
X = X.fillna(0)
X = scaler.fit_transform(X)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# prediction
y_pred = knn.predict(X)
y_proba = knn.predict_proba(X)
# ROC-AUC Score
roc_auc = roc_auc_score(y, y_proba[:, 1])

# evaluation Metrics
print("Accuracy:", accuracy_score(y, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("Precision:", precision_score(y, y_pred, average='weighted', zero_division=1))
print("Recall:", recall_score(y, y_pred, average='weighted', zero_division=1))
print("\nClassification Report:\n", classification_report(y, y_pred, zero_division=1))
print("\nROC-AUC Score:", roc_auc)
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

data = pd.read_csv('TEAM SMURF-MidSummativeAct - Seen_Data.csv')

# Encoding day of the week
le = LabelEncoder()
le.fit(data['day of week'])
data['day_of_week'] = le.transform(data['day of week'])

# counting bought breads and beverages
data['bread_count'] = data['Breads'].apply(lambda x: len(str(x).split(',')))
data['beverage_count'] = data['Beverages'].apply(lambda x: len(str(x).split(',')))
data['desert_count'] = data['Desserts'].apply(lambda x: len(str(x).split(',')))

# Feature selection
features = ['day_of_week', 'bread_count', 'beverage_count']
X = data[features]

# creating the target variable Y  for a specific bread type
specific_data = "angbutter" #ito yung palitan
data['target'] = data['Breads'].apply(lambda x: 1 if specific_data in str(x).lower() else 0)
y = data['target']

# standardizing the features
scaler = StandardScaler()
X = X.fillna(0)
X = scaler.fit_transform(X)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prediction
y_pred = knn.predict(X_test)

# evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-Score:", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
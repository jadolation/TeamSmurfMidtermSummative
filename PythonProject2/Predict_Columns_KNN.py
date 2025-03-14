import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, roc_auc_score

data = pd.read_csv('TEAM SMURF-MidSummativeAct - Train_Data.csv')

# extracting hour from datetime
data['datetime'] = pd.to_datetime(data['datetime'])
data['hour'] = data['datetime'].dt.hour

# encoding day of the week
le = LabelEncoder()
le.fit(data['day of week'])  # Fit the encoder with the unique values from the data
data['day_of_week'] = le.transform(data['day of week'])

# Counting items in Breads and Beverages columns
data['bread_count'] = data['Breads'].apply(lambda x: len(str(x).split(',')))
data['beverage_count'] = data['Beverages'].apply(lambda x: len(str(x).split(',')))
data['dessert_count'] = data['Desserts'].apply(lambda x: len(str(x).split(',')))

# feature selection
features = ['hour', 'day_of_week', 'bread_count', 'beverage_count', 'total']
X = data[features]

# para makita kung may NaN value sa row o wala
print('Number of rows with NaN: ', data['day_of_week'].isna().sum(), '\n')

# predicting the column
y = data['dessert_count']  # dapat palitan based dun sa column na gusto ipredict

# Preparing the feature set
X = data[features].fillna(0)

# Standardizing the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# prediction
y_pred = knn.predict(X)
y_pred = knn.predict(X)

# evaluation Metrics
print("Accuracy:", accuracy_score(y, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("Precision:", precision_score(y, y_pred, average='weighted', zero_division=1))
print("Recall:", recall_score(y, y_pred, average='weighted', zero_division=1))
print("\nClassification Report:\n", classification_report(y, y_pred, zero_division=1))

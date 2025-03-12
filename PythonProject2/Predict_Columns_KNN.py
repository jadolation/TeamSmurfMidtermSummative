import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

data = pd.read_csv('TEAM SMURF-MidSummativeAct - Seen_Data.csv')

# encoding day of the week
le = LabelEncoder()
le.fit(data['day of week'])  # Fit the encoder with the unique values from the data
data['day_of_week'] = le.transform(data['day of week'])

# counting bought breads and beverages
data['bread_count'] = data['Breads'].apply(lambda x: len(str(x).split(',')))
data['beverage_count'] = data['Beverages'].apply(lambda x: len(str(x).split(',')))

# feature selection
features = ['day_of_week', 'bread_count', 'beverage_count']

X = data[features]

# para makita kung may NaN value sa row o wala
print('Number of rows with NaN: ', data['day_of_week'].isna().sum(), '\n')

# predicting the column
y = data['day_of_week']  # dapat palitan based dun sa column na gusto ipredict

# standardizing the features
scaler = StandardScaler()
X = data[features].fillna(0)
X = scaler.fit_transform(X)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# prediction
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)[:, 1]  # Get probability estimates for positive class

# evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

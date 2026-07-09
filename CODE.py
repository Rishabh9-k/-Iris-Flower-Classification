import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load the dataset
# Make sure 'Iris.csv' is in the same directory or provide the correct path
df = pd.read_csv('Iris.csv')

print("--- Dataset Preview ---")
print(df.head())
print("\n--- Species Distribution ---")
print(df['Species'].value_counts())

# 2. Split into Features (X) and Target (y)
# We drop 'Id' because it's just a row identifier, and 'Species' because it's our target variable
X = df.drop(columns=['Id', 'Species'])
y = df['Species']

# 3. Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Feature Scaling (Good practice, especially for distance-based algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Building (Using a Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Make Predictions
y_pred = model.predict(X_test_scaled)

# 7. Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print("\n==============================")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("==============================\n")

print("--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))


## OUTPUT: 

--- Dataset Preview ---
   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0   1            5.1           3.5            1.4           0.2  Iris-setosa
1   2            4.9           3.0            1.4           0.2  Iris-setosa
2   3            4.7           3.2            1.3           0.2  Iris-setosa
3   4            4.6           3.1            1.5           0.2  Iris-setosa
4   5            5.0           3.6            1.4           0.2  Iris-setosa

--- Species Distribution ---
Species
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
Name: count, dtype: int64

==============================
Model Accuracy: 100.00%
==============================

--- Detailed Classification Report ---
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       1.00      1.00      1.00        10
 Iris-virginica       1.00      1.00      1.00        10

       accuracy                           1.00        30
      macro avg       1.00      1.00      1.00        30
   weighted avg       1.00      1.00      1.00        30


--- Confusion Matrix ---
[[10  0  0]
 [ 0 10  0]
 [ 0  0 10]]

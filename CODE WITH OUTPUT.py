import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv('Iris.csv')

# 2. Split into Features (X) and Target (y)
X = df.drop(columns=['Id', 'Species'])
y = df['Species']

# 3. Stratified Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 6. Graph 1: Feature Scatter Plot (Petal Length vs Width)
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='Species', palette='Set1', ax=ax, s=70)
ax.set_title('Iris Species Distribution (Petal Length vs Width)')
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
plt.tight_layout()
plt.savefig('iris_features_scatter.png')
plt.close()

# 7. Graph 2: Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
labels = sorted(y.unique())

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
ax.set_title('Model Confusion Matrix')
ax.set_xlabel('Predicted Species')
ax.set_ylabel('True Species')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
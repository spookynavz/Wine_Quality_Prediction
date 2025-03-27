import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

# Load the preprocessed dataset
df = pd.read_csv("C:/Users/navya/OneDrive/Desktop/wine quality/Wine_quality_preprocessed.csv")

# Convert target variable into categorical values
df['quality'] = pd.cut(df['quality'], bins=[2, 5, 7, 9], labels=['Low', 'Medium', 'High'])
df['quality'] = df['quality'].astype('category')

# Correlation heatmap
plt.figure(figsize=(10, 6))
numeric_features = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
sns.heatmap(numeric_features.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Define features and target
X = df.drop(columns=['quality'])
y = df['quality']

# Normalize features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality Reduction with PCA
pca = PCA(n_components=5)  # Reducing to 5 principal components
X_pca = pca.fit_transform(X_scaled)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=101)

# Logistic Regression
log_model = LogisticRegression(max_iter=500, random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, y_pred_log))
print(f"Accuracy: {accuracy_score(y_test, y_pred_log)}")

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest:")
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")

# Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print("Gradient Boosting:")
print(classification_report(y_test, y_pred_gb))
print(f"Accuracy: {accuracy_score(y_test, y_pred_gb)}")

#Confusion Matrix Visualization
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=['Low', 'Medium', 'High'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plot_confusion_matrix(y_test, y_pred_log, "Logistic Regression Confusion Matrix")
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest Confusion Matrix")
plot_confusion_matrix(y_test, y_pred_gb, "Gradient Boosting Confusion Matrix")

# Feature Importance Visualization (for Random Forest)
feature_importances = rf_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances, align='center')
plt.title('Feature Importances (Random Forest)')
plt.xlabel('Feature Index')
plt.ylabel('Importance Score')
plt.show()

# Scatter Plot for PCA Components
def scatter_plot(X_pca, y):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.cat.codes, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ticks=[0, 1, 2], label='Quality (Encoded)')
    plt.title('Scatter Plot of PCA Components')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

scatter_plot(X_pca, y)
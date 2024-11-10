# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Load the CKD dataset
data = pd.read_csv("chronic_kidney_disease.csv")

# Data Preprocessing
# Impute missing values
imputer = SimpleImputer(strategy="mean")
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)  # Fill categorical features with mode
    else:
        data[col] = imputer.fit_transform(data[[col]])  # Fill numerical features with mean

# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
numeric_cols = data.select_dtypes(include=np.number).columns
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Separate features and target variable
X = data.drop("class", axis=1)
y = data["class"]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training with Ensemble
# Define individual models
rf = RandomForestClassifier(random_state=42)
dt = DecisionTreeClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)

# Ensemble model with Voting Classifier
ensemble_model = VotingClassifier(
    estimators=[('rf', rf), ('dt', dt), ('svm', svm)],
    voting='soft'
)
ensemble_model.fit(X_train, y_train)

# Model Evaluation
# Predict and evaluate on test data
y_pred = ensemble_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Optional: Hyperparameter Tuning for RandomForest
param_grid = {
    'rf__n_estimators': [50, 100, 150],
    'rf__max_depth': [None, 10, 20, 30],
}

grid_search = GridSearchCV(estimator=ensemble_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Visualize Feature Importance for RandomForest
feature_importances = rf.feature_importances_
indices = np.argsort(feature_importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.show()

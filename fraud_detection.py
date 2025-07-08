import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load Data
data = pd.read_csv('')

# Create Target
data['is_fraud'] = data['CLAIM_STATUS'].apply(lambda x: 1 if x == 'D' else 0)

# Features
features = [
    'CLAIM_AMOUNT', 'PREMIUM_AMOUNT', 'AGE', 'TENURE',
    'ANY_INJURY', 'POLICE_REPORT_AVAILABLE',
    'INSURANCE_TYPE', 'INCIDENT_SEVERITY', 'AUTHORITY_CONTACTED'
]

# Encode Categorical
encoder = LabelEncoder()
for col in ['INSURANCE_TYPE', 'INCIDENT_SEVERITY', 'AUTHORITY_CONTACTED']:
    data[col] = encoder.fit_transform(data[col])

X = data[features]
y = data['is_fraud']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train Model
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best Params: {grid.best_params_}")


model = grid.best_estimator_
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Score Example
new_claim = pd.DataFrame({
    'CLAIM_AMOUNT': [12000],
    'PREMIUM_AMOUNT': [200],
    'AGE': [45],
    'TENURE': [5],
    'ANY_INJURY': [0],
    'POLICE_REPORT_AVAILABLE': [1],
    'INSURANCE_TYPE': [encoder.transform(['Health'])[0]],
    'INCIDENT_SEVERITY': [encoder.transform(['Major Loss'])[0]],
    'AUTHORITY_CONTACTED': [encoder.transform(['Police'])[0]]
})

risk_score = model.predict_proba(new_claim)[0][1]
print(f"Fraud Risk Score: {risk_score:.2f}")

if risk_score > 0.7:
    print("Flagged for investigation.")
else:
    print("Likely genuine.")

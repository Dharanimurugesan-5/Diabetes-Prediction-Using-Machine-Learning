import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = pd.read_csv("diabetesprediction.csv")

# Encoding
le = LabelEncoder()

df['gender'] = le.fit_transform(df['gender'])
df['smoking_history'] = le.fit_transform(df['smoking_history'])

# Features and Target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest Model
rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

# Predictions
rf_pred = rf.predict(X_test)

# Accuracy
print("Random Forest Accuracy:",
      accuracy_score(y_test, rf_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

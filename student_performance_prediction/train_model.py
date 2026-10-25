
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: Load the dataset
df = pd.read_csv("student-mat.csv", sep=";")

# STEP 2: Choose input features and target
numeric_features = [
    "age", "studytime", "failures", "absences"
]

categorical_features = [
    "schoolsup", "famsup", "paid", "activities"
]

features = numeric_features + categorical_features
target = "G3"

X = df[features].copy()
y = df[target].copy()

print("Input features:", features)
print("Target:", target)
print("Records:", len(df))

# STEP 3: Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Test records:", len(X_test))

# STEP 4: Create preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# STEP 5: Define the models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        random_state=42
    )
}

# STEP 6: Train and compare using validation MAE
results = {}
best_name = None
best_pipeline = None
best_mae = float("inf")

for name, algorithm in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, val_predictions)
    rmse = mean_squared_error(
        y_val, val_predictions
    ) ** 0.5
    r2 = r2_score(y_val, val_predictions)

    results[name] = {
        "validation_MAE": float(mae),
        "validation_RMSE": float(rmse),
        "validation_R2": float(r2)
    }

    print(f"\n{name}")
    print(f"Validation MAE: {mae:.3f}")
    print(f"Validation RMSE: {rmse:.3f}")
    print(f"Validation R2: {r2:.3f}")

    if mae < best_mae:
        best_mae = mae
        best_name = name
        best_pipeline = pipeline

# STEP 7: Evaluate the selected model on the test set
test_predictions = best_pipeline.predict(X_test)

test_metrics = {
    "MAE": float(mean_absolute_error(y_test, test_predictions)),
    "RMSE": float(mean_squared_error(
        y_test, test_predictions
    ) ** 0.5),
    "R2": float(r2_score(y_test, test_predictions))
}

# STEP 8: Save the model and results
joblib.dump(best_pipeline, "student_model.joblib")

with open("metrics.json", "w") as file:
    json.dump({
        "selected_model": best_name,
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "validation_results": results,
        "test_metrics": test_metrics
    }, file, indent=4)

print("\nSelected model:", best_name)
print("Final test results:", test_metrics)
print("\nModel saved to student_model.joblib")
print("Metrics saved to metrics.json")
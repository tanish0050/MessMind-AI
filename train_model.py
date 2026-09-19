import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load dataset
df = pd.read_csv("food_waste_data.csv")


# Input features
X = df[
    [
        "day",
        "meal",
        "menu",
        "students_present"
    ]
]


# Target variable
y = df["leftover_kg"]


# Categorical columns
categorical_features = [
    "day",
    "meal",
    "menu"
]


# Numerical columns
numerical_features = [
    "students_present"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# Machine Learning model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=12
)


# Complete ML pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train
pipeline.fit(X_train, y_train)


# Predictions
predictions = pipeline.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print("--------------------------------")
print("MODEL TRAINING COMPLETED")
print("--------------------------------")

print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))


# Save model
joblib.dump(pipeline, "waste_model.pkl")

print("--------------------------------")
print("Model saved as waste_model.pkl")
print("--------------------------------")
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "notebooks" / "data" / "processed" / "modified_data.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TARGET_COLUMN = "Will_Buy_EV"
CATEGORICAL_COLUMNS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]
NUMERIC_COLUMNS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def train_and_save() -> None:
    data = pd.read_csv(DATA_PATH)
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    if target.isna().any():
        raise ValueError(f"Unknown values found in {TARGET_COLUMN}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    train_features_transformed = preprocessor.fit_transform(train_features)
    test_features_transformed = preprocessor.transform(test_features)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(train_features_transformed, train_target)

    accuracy = model.score(test_features_transformed, test_target)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")

    print(f"Saved preprocessor to {ARTIFACTS_DIR / 'preprocessor.joblib'}")
    print(f"Saved model to {ARTIFACTS_DIR / 'model.joblib'}")
    print(f"Validation accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    train_and_save()

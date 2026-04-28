from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'heart.csv'
MODEL_PATH = BASE_DIR / 'heart_disease_model.pkl'
SCALER_PATH = BASE_DIR / 'heart_disease_scaler.pkl'
TARGET_COLUMN = 'target'
RANDOM_STATE = 42


def main():
    data = pd.read_csv(DATA_PATH)

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [None, 8, 16],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': [None, 'balanced'],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train_scaled, y_train)

    model = search.best_estimator_
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print('Heart model trained successfully.')
    print(f'Best CV Accuracy: {search.best_score_:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Best Params: {search.best_params_}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == '__main__':
    main()

import joblib

rf_model = joblib.load("models/random_forest.pkl")
xgb_model = joblib.load("models/xgboost.pkl")


def predict_with_models(data):

    rf_pred = rf_model.predict(data)
    rf_prob = rf_model.predict_proba(data)[:, 1]

    xgb_pred = xgb_model.predict(data)
    xgb_prob = xgb_model.predict_proba(data)[:, 1]

    return {

        "Random Forest": {
            "Prediction": rf_pred,
            "Fraud Probability": rf_prob
        },

        "XGBoost": {
            "Prediction": xgb_pred,
            "Fraud Probability": xgb_prob
        }
    }
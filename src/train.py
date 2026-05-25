from xgboost import XGBClassifier
import joblib

def train_model(X_train, y_train):

    rf = RandomForestClassifier(
    n_estimators=80, 
    max_depth=None,
    random_state=42,
    n_jobs=-1,           
    class_weight='balanced' 
)
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    joblib.dump(rf, "models/random_forest.pkl")
    joblib.dump(xgb, "models/xgboost.pkl")

    return rf, xgb

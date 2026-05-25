from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)

    print(f"Results for {model_name}")

    print(classification_report(y_test, y_pred))
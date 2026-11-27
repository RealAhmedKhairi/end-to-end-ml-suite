from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from processor import load_data

def tune_logistic_regression(x_train, y_train, x_val, y_val):
    c_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    best_c = None
    best_val_acc = 0.0
    best_lr_model = None
    for c in c_values:
        lr_model=LogisticRegression(C=c, max_iter=1000, random_state=42)
        lr_model.fit(x_train, y_train)

        train_pred = lr_model.predict(x_train)
        val_pred = lr_model.predict(x_val)
        
        train_acc = accuracy_score(y_train,train_pred)
        val_acc = accuracy_score(y_val,val_pred)

        print(f"{c:<10} | {train_acc:<12.4f} | {val_acc:<12.4f}")

        if val_acc > best_val_acc :
            best_val_acc = val_acc
            best_c = c
            best_lr_model = lr_model

    print("-" * 40)
    print(f"Best C: {best_c} | Best Validation Accuracy: {best_val_acc:.4f}")
    return best_lr_model, best_c, best_val_acc


if  __name__ == "__main__":
    import os
    import joblib

    x_train, x_val, x_test, y_train, y_val, y_test, le, scaler = load_data()
    best_lr_model, best_c, best_val_acc = tune_logistic_regression(x_train, y_train, x_val, y_val)

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_lr_model, os.path.join(models_dir, "image_lr_model.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "image_scaler.joblib"))
    joblib.dump(le, os.path.join(models_dir, "image_label_encoder.joblib"))


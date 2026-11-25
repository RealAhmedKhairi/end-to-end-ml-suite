import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from processor import load_data
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def tune_knn_hyperparameter(
    X_train, y_train, X_val, y_val,
    k_min=1, k_max=25
):
    val_mae_scores = {}

    
    for k in range(k_min, k_max + 1):
        model = KNeighborsRegressor(n_neighbors=k)

       
        model.fit(X_train, y_train)

        
        val_preds = model.predict(X_val)

        
        val_mae_scores[k] = mean_absolute_error(
            y_val, val_preds
        )

    
    best_k = min(val_mae_scores, key=val_mae_scores.get)

    return best_k, val_mae_scores


def evaluate_regression_model(model, X_test, y_test):

    preds = model.predict(X_test)

    
    mae = mean_absolute_error(y_test, preds)

    
    rmse = np.sqrt(
        mean_squared_error(y_test, preds)
    )

    
    r2 = r2_score(y_test, preds)

    
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    return preds, metrics

def plot_actual_vs_pred(
    y_test, y_pred, model_title, output_path
):

    # Create the plots directory automatically before saving the figure.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    
    plt.figure(figsize=(7, 6))

    
    plt.scatter(
        y_test, y_pred,
        alpha=0.65,
        edgecolors="k",
        linewidth=0.4
    )

    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        label="Ideal Fit (y_pred = y_test)"
    )

    
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title(model_title)

    
    plt.legend()
    plt.tight_layout()

    
    plt.savefig(output_path, dpi=150)

    
    plt.close()


def save_artifacts(
    scaler, knn_model,
    models_dir="models"
):

    # Create the output directory if it does not exist.
    # exist_ok=True prevents an error if the directory already exists.
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(
        scaler,
        os.path.join(models_dir, "tabular_scaler.joblib")
    )

    # os.path.join combines directory and file names safely.
    joblib.dump(
        knn_model,
        os.path.join(models_dir, "knn_model.joblib")
    )



def run_knn_pipeline():

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler
    ) = load_data()


    best_k, val_mae_scores = tune_knn_hyperparameter(
        X_train, y_train,
        X_val, y_val,
        1, 25
    )

    
    print(
        # The 'f' is used to allow variables to be inserted inside curly braces {} in the string.
        f"Optimal k* selected on validation set: "
        f"{best_k} (Val MAE = {val_mae_scores[best_k]:.2f})"
    )


    best_knn = KNeighborsRegressor(
        n_neighbors=best_k
    )
    
    best_knn.fit(X_train, y_train)

    knn_preds, knn_metrics = evaluate_regression_model(
        best_knn, X_test, y_test
    )

    plot_actual_vs_pred(
        y_test,
        knn_preds,
        f"k-NN Regressor (k={best_k}): Actual vs. Predicted Charges",
        "numerical/plots/knn_actual_vs_pred.png"
    )

    save_artifacts(
        scaler,
        best_knn,
        models_dir="models"
    )
    
    summary_df = pd.DataFrame([
        {
            "Model": f"k-NN Regressor (k={best_k})",
            **knn_metrics
        }
    ])
    
    print("\nKNN Test Set Evaluation Summary:")
    print(summary_df.to_string(index=False))

    return best_k, best_knn, summary_df

if __name__ == "__main__":
    run_knn_pipeline()

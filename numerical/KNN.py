# We use the os module to interact with the operating system,
# create directories, and manage file paths.
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


# Task 2.3: This function tests different K values from 1 to 25,
# evaluates each model using Validation MAE, and returns the best K.
def tune_knn_hyperparameter(
    X_train, y_train, X_val, y_val,
    k_min=1, k_max=25
):
    val_mae_scores = {}

    # Try each K value in the specified range.
    for k in range(k_min, k_max + 1):
        model = KNeighborsRegressor(n_neighbors=k)

        # Train the KNN model using training data.
        model.fit(X_train, y_train)

        # Predict target values for validation data.
        val_preds = model.predict(X_val)

        # Calculate and store the Validation MAE for each K.
        val_mae_scores[k] = mean_absolute_error(
            y_val, val_preds
        )

    # Select the K value with the lowest Validation MAE.
    best_k = min(val_mae_scores, key=val_mae_scores.get)

    return best_k, val_mae_scores


# Task 2.4: This function evaluates the trained KNN model
# on the test set using MAE, RMSE, and R-squared metrics.
def evaluate_regression_model(model, X_test, y_test):

    # Generate predictions using the trained model.
    preds = model.predict(X_test)

    # Calculate Mean Absolute Error (MAE).
    mae = mean_absolute_error(y_test, preds)

    # Calculate Root Mean Squared Error (RMSE).
    rmse = np.sqrt(
        mean_squared_error(y_test, preds)
    )

    # Calculate R-squared to measure model performance.
    r2 = r2_score(y_test, preds)

    # Store all evaluation metrics in a dictionary.
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    return preds, metrics


# Task 2.5: This function creates and saves a scatter plot
# comparing actual target values with predicted values.
# It also adds an ideal prediction reference line.
def plot_actual_vs_pred(
    y_test, y_pred, model_title, output_path
):

    # Create the plots directory automatically before saving the figure.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a figure with a specified size.
    plt.figure(figsize=(7, 6))

    # Plot actual values against predicted values.
    plt.scatter(
        y_test, y_pred,
        alpha=0.65,
        edgecolors="k",
        linewidth=0.4
    )

    # Determine the minimum and maximum values for the reference line.
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    # Draw the ideal prediction line where predicted values equal actual values.
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        label="Ideal Fit (y_pred = y_test)"
    )

    # Add labels and title to the plot.
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title(model_title)

    # Display the legend and adjust the layout.
    plt.legend()
    plt.tight_layout()

    # Save the plot as an image with high resolution.
    plt.savefig(output_path, dpi=150)

    # Close the figure to release memory.
    plt.close()


# Task 2.6: This function saves the trained StandardScaler
# and KNN model as Joblib files so they can be loaded and reused later.
def save_artifacts(
    scaler, knn_model,
    models_dir="models"
):

    # Create the output directory if it does not exist.
    # exist_ok=True prevents an error if the directory already exists.
    os.makedirs(models_dir, exist_ok=True)

    # Save the fitted StandardScaler.
    joblib.dump(
        scaler,
        os.path.join(models_dir, "tabular_scaler.joblib")
    )

    # Save the trained KNN model.
    # os.path.join combines directory and file names safely.
    joblib.dump(
        knn_model,
        os.path.join(models_dir, "knn_model.joblib")
    )


# Main pipeline: This function runs the complete KNN workflow.
# It loads preprocessed data, tunes K, trains the model,
# evaluates its performance, generates a plot, and saves the model.
def run_knn_pipeline():

    # Task 2.1: Load the preprocessed and scaled data
    # using the shared load_data() function from the processor module.
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler
    ) = load_data()

    # Task 2.3: Find the optimal K using Validation MAE.
    best_k, val_mae_scores = tune_knn_hyperparameter(
        X_train, y_train,
        X_val, y_val,
        1, 25
    )

    # Display the best K and its validation error.
    print(
        f"Optimal k* selected on validation set: "
        f"{best_k} (Val MAE = {val_mae_scores[best_k]:.2f})"
    )

    # Task 2.2: Create the KNN Regressor using the optimal K.
    best_knn = KNeighborsRegressor(
        n_neighbors=best_k
    )

    # Train the final KNN model using the training data.
    best_knn.fit(X_train, y_train)

    # Task 2.4: Evaluate the trained KNN model on the test set.
    knn_preds, knn_metrics = evaluate_regression_model(
        best_knn, X_test, y_test
    )

    # Task 2.5: Generate and save the KNN Actual vs Predicted plot.
    plot_actual_vs_pred(
        y_test,
        knn_preds,
        f"k-NN Regressor (k={best_k}): Actual vs. Predicted Charges",
        "numerical/plots/knn_actual_vs_pred.png"
    )

    # Task 2.6: Save the trained StandardScaler and KNN model.
    save_artifacts(
        scaler,
        best_knn,
        models_dir="models"
    )

    # Create a summary table containing the KNN evaluation metrics.
    summary_df = pd.DataFrame([
        {
            "Model": f"k-NN Regressor (k={best_k})",
            **knn_metrics
        }
    ])

    # Display the final KNN evaluation results.
    print("\nKNN Test Set Evaluation Summary:")
    print(summary_df.to_string(index=False))

    return best_k, best_knn, summary_df


# Execute the main pipeline when this file is run directly.
if __name__ == "__main__":
    run_knn_pipeline()

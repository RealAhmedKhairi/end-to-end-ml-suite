# We use the os module to interact with the operating system,
# create directories, and manage file paths.
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from processor import load_data

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# This function creates the Linear Regression model and trains it on the training data.
# It has no hyperparameters to tune, so there is no tuning step before training.
def train_linear_regression(X_train, y_train):

    # Create the Linear Regression model.
    lr_model = LinearRegression()

    # Train the model: it learns one weight per feature from the training data only.
    lr_model.fit(X_train, y_train)

    return lr_model


# This function evaluates a trained model on a given set
# using MAE, RMSE, and R-squared metrics.
def evaluate_regression_model(model, X, y):

    # Generate predictions using the trained model.
    preds = model.predict(X)

    # Calculate Mean Absolute Error (MAE).
    mae = mean_absolute_error(y, preds)

    # Calculate Root Mean Squared Error (RMSE).
    rmse = np.sqrt(
        mean_squared_error(y, preds)
    )

    # Calculate R-squared to measure model performance.
    r2 = r2_score(y, preds)

    # Store all evaluation metrics in a dictionary.
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    return preds, metrics

# This function creates and saves a scatter plot
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


# This function saves the trained Linear Regression model as a Joblib file
# so it can be loaded and reused later (for example by the GUI).
def save_model(lr_model, models_dir="models"):

    # Create the output directory if it does not exist.
    os.makedirs(models_dir, exist_ok=True)

    # Save the trained Linear Regression model.
    joblib.dump(
        lr_model,
        os.path.join(models_dir, "linear_regression_model.joblib")
    )


# Main pipeline: This function runs the complete Linear Regression workflow.
# It loads preprocessed data, trains the model, evaluates its performance,
# generates a plot, and saves the model.
def run_linear_regression_pipeline():

    # Load the preprocessed and scaled data
    # using the shared load_data() function from the processor module.
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _        # scaler we don't need it
    ) = load_data()


    # Train the Linear Regression model using the training data.
    lr_model = train_linear_regression(X_train, y_train)

    # Check the model on the validation set (it is only used as a check, not for tuning).
    _, val_metrics = evaluate_regression_model(lr_model, X_val, y_val)
    print(f"Validation MAE = {val_metrics['MAE']:.2f}")

    # Evaluate the trained model on the test set (done once, at the end).
    lr_preds, lr_metrics = evaluate_regression_model(
        lr_model, X_test, y_test
    )

    # Generate and save the Linear Regression Actual vs Predicted plot.
    plot_actual_vs_pred(
        y_test,
        lr_preds,
        "Linear Regression: Actual vs. Predicted Charges",
        "numerical/plots/linear_regression_actual_vs_pred.png"
    )

    # Save the trained Linear Regression model.
    save_model(lr_model, models_dir="models")

    # Create a summary table containing the evaluation metrics.
    summary_df = pd.DataFrame([
        {
            "Model": "Linear Regression",
            **lr_metrics
        }
    ])

    # Display the final evaluation results.
    print("\nLinear Regression Test Set Evaluation Summary:")
    print(summary_df.to_string(index=False))

    return lr_model, summary_df


# Execute the main pipeline when this file is run directly.
if __name__ == "__main__":
    run_linear_regression_pipeline()

# import libraries 
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


# create and train Linear regression model
def train_linear_regression(X_train, y_train):

    lr_model = LinearRegression()

    lr_model.fit(X_train, y_train)

    return lr_model


# we did a function evaluate_regression_model by using MAE, RMSE, R2 
def evaluate_regression_model(model, X, y):

    # After finishing training, we make prediction
    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)

    rmse = np.sqrt( mean_squared_error(y, preds))
 
    r2 = r2_score(y, preds)

   # we store in Dictionary 
    
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    return preds, metrics

# create a scatter plot to compare between actual vs prediction 
def plot_actual_vs_pred(
    y_test, y_pred, model_title, output_path
):

   # Create the plots folder if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

   # size w=7 , h=7
    plt.figure(figsize=(7, 6))

    # Plot y_test on x-axis and y_pred on y_axis
    plt.scatter(
        y_test, y_pred,
        alpha=0.65,
        edgecolors="k",
        linewidth=0.4
    )

   # Get the min، max values to draw the reference line(red line)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    # Draw red dashed line represent ideal prediction
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        label="Ideal Fit (y_pred = y_test)"
    )

    # Add labels and title to the plot
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title(model_title)

    # Display the legend and adjust the layout
    plt.legend()
    plt.tight_layout()

    # Save the plot 
    plt.savefig(output_path, dpi=150)

    # to release memory
    plt.close()


# save Linear regression (trained) as a joblib file for GUI
def save_model(lr_model, models_dir="models"):

    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(
        lr_model,
        os.path.join(models_dir, "linear_regression_model.joblib")
    )


# This function runs the complete Linear Regression workflow.
# First loads preprocessed data, trains the model, evaluates its performance,
# and then generates a plot, and saves the model.
def run_linear_regression_pipeline():

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _        # scaler we don't need it
    ) = load_data()   # using load_data() function from the processor.py


    # Train the Linear Regression model 
    lr_model = train_linear_regression(X_train, y_train)

    # Evaluate the model on the validation set and display the MAE.
    _, val_metrics = evaluate_regression_model(lr_model, X_val, y_val)
    print(f"Validation MAE = {val_metrics['MAE']:.2f}")

    # Evaluate the trained model on the test set 
    lr_preds, lr_metrics = evaluate_regression_model(
        lr_model, X_test, y_test
    )

    # Generate and save the Linear Regression Actual vs Predicted plot
    plot_actual_vs_pred(
        y_test,
        lr_preds,
        "Linear Regression: Actual vs. Predicted Charges",
        "numerical/plots/linear_regression_actual_vs_pred.png"
    )

    # Save the trained Linear Regression model.
    save_model(lr_model, models_dir="models")

    # Create a summary table containing the evaluation metrics (Model,MAE,RMSE,R2)
    summary_df = pd.DataFrame([
        {
            "Model": "Linear Regression",
            **lr_metrics
        }
    ])

    # Display summary table results.
    print("\nLinear Regression Test Set Evaluation Summary:")
    print(summary_df.to_string(index=False))

    return lr_model, summary_df


# Execute the main pipeline when this file is run directly.
if __name__ == "__main__":
    run_linear_regression_pipeline()

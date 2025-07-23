# Extra Model Evaluation Script - IBM SkillBuild Internship
# Author: [Your Name]
#
# This script is for extra/advanced analysis of the salary prediction model.
# You don't need to run this for the main project, but it's here if you want to dig deeper!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = model.predict(X_test)

    def show_metrics(self):
        # Print out some detailed metrics
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        mape = np.mean(np.abs((self.y_test - self.y_pred) / self.y_test)) * 100
        print("\nDetailed Model Evaluation:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Slope: {self.model.coef_[0]:.2f}, Intercept: {self.model.intercept_:.2f}")
        return mse, rmse, mae, r2, mape

    def cross_val(self, cv=5):
        # Do cross-validation (just for fun)
        X_all = np.vstack([self.X_train, self.X_test])
        y_all = np.hstack([self.y_train, self.y_test])
        scores = cross_val_score(self.model, X_all, y_all, cv=cv, scoring='r2')
        print(f"\nCross-validation R2 scores: {scores}")
        print(f"Mean CV R2: {scores.mean():.4f}, Std: {scores.std():.4f}")
        return scores

    def plot_evaluation(self):
        # Make some extra plots
        residuals = self.y_test - self.y_pred
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(self.y_test, self.y_pred, color='blue')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        plt.xlabel('Actual Salary')
        plt.ylabel('Predicted Salary')
        plt.title('Actual vs Predicted')
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=10, color='orange', edgecolor='black')
        plt.xlabel('Residuals')
        plt.title('Residuals Distribution')
        plt.tight_layout()
        plt.savefig('results/extra_evaluation.png')
        print("Extra evaluation plots saved to results/extra_evaluation.png")

    def run_all(self):
        self.show_metrics()
        self.cross_val()
        self.plot_evaluation()
        print("\nDone with extra evaluation!")

# Example usage (not needed for main project)
def example_usage(predictor):
    if not predictor.trained:
        print("Train the model first!")
        return
    evaluator = ModelEvaluator(
        predictor.model,
        predictor.X_train,
        predictor.X_test,
        predictor.y_train,
        predictor.y_test
    )
    evaluator.run_all()

if __name__ == "__main__":
    print("This is an extra/optional script for advanced model evaluation.")
    print("Import and use example_usage(predictor) if you want to try it!")
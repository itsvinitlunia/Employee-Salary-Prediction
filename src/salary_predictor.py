# Salary Prediction Project - IBM SkillBuild Internship
# Author: [Your Name]
#
# This script predicts employee salaries based on years of experience using Linear Regression.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os

# This class handles everything: loading data, training, predicting, and showing results
class SalaryPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.trained = False
        self.data = None

    def load_data(self, file_path):
        # Try to load the CSV file
        try:
            print("Loading data from:", file_path)
            self.data = pd.read_csv(file_path)
            print("Data loaded! Shape:", self.data.shape)
            return True
        except Exception as e:
            print("Couldn't load data:", e)
            return False

    def show_data_info(self):
        # Show some basic info about the data
        if self.data is None:
            print("No data loaded yet!")
            return
        print("\nFirst 5 rows:")
        print(self.data.head())
        print("\nColumns:", list(self.data.columns))
        print("\nMissing values:")
        print(self.data.isnull().sum())
        print("\nStats:")
        print(self.data.describe())
        # Save info to a file (optional)
        os.makedirs('results', exist_ok=True)
        with open('results/data_info.txt', 'w') as f:
            f.write(str(self.data.describe()))

    def prepare(self, test_size=0.2, random_state=42):
        # Split data into features and target
        if self.data is None:
            print("No data loaded!")
            return False
        self.X = self.data[['YearsExperience']].values
        self.y = self.data['Salary'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)
        print(f"Training samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")
        return True

    def train(self):
        # Train the model
        if not hasattr(self, 'X_train'):
            print("Data not prepared!")
            return False
        self.model.fit(self.X_train, self.y_train)
        self.trained = True
        print("Model trained!")
        print(f"Slope: {self.model.coef_[0]:.2f}, Intercept: {self.model.intercept_:.2f}")
        return True

    def evaluate(self):
        # Check how good the model is
        if not self.trained:
            print("Model not trained yet!")
            return
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print("\nModel Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R2 Score: {r2:.4f}")
        # Save metrics
        with open('results/model_metrics.txt', 'w') as f:
            f.write(f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR2: {r2:.4f}\n")
        return mse, rmse, mae, r2

    def plot_results(self):
        # Show some plots to visualize the results
        if not self.trained:
            print("Model not trained yet!")
            return
        y_pred = self.model.predict(self.X_test)
        plt.figure(figsize=(12, 5))
        # Scatter plot: Actual vs Predicted
        plt.subplot(1, 2, 1)
        plt.scatter(self.X_test, self.y_test, color='blue', label='Actual')
        plt.scatter(self.X_test, y_pred, color='red', label='Predicted')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.title('Actual vs Predicted Salary')
        plt.legend()
        # Residuals
        plt.subplot(1, 2, 2)
        residuals = self.y_test - y_pred
        plt.scatter(self.X_test, residuals, color='green')
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel('Years of Experience')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.title('Residuals Plot')
        plt.tight_layout()
        plt.savefig('results/salary_plots.png')
        print("Plots saved to results/salary_plots.png")
        # plt.show()  # Uncomment if running interactively

    def predict(self, years):
        # Predict salary for a given number of years
        if not self.trained:
            print("Model not trained yet!")
            return None
        return self.model.predict([[years]])[0]

    def run_all(self, file_path):
        print("\n--- Salary Prediction Project ---")
        if not self.load_data(file_path):
            return
        self.show_data_info()
        if not self.prepare():
            return
        if not self.train():
            return
        self.evaluate()
        self.plot_results()
        # Show some sample predictions
        print("\nSample Predictions:")
        for exp in [1, 3, 5, 7, 10]:
            pred = self.predict(exp)
            print(f"Years: {exp} -> Predicted Salary: ${pred:,.2f}")
        print("\nDone! Check the 'results' folder for outputs.")

def main():
    predictor = SalaryPredictor()
    predictor.run_all('data/Salary_Data.csv')

if __name__ == "__main__":
    main()
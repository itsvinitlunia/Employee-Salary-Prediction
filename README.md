# Salary Prediction Project (IBM SkillBuild Internship)

Hey there! 👋

This is a simple machine learning project I made for the IBM SkillBuild AI Internship. The goal is to predict employee salaries based on their years of experience using Linear Regression. It's a great way to see how data science works in real life!

## What does this project do?
- Loads a small dataset of employees (years of experience + salary)
- Trains a linear regression model to find the relationship
- Shows how well the model works (with some easy-to-understand metrics)
- Makes predictions for new experience values
- Plots the results so you can see how good (or bad!) the model is

## How to Run This Project

1. **Install Python** (if you don't have it already)
   - Python 3.7 or above is best.

2. **Install the required packages**
   - Open a terminal in the project folder and run:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the main script**
   - In the terminal, type:
     ```bash
     python src/salary_predictor.py
     ```
   - This will print info, train the model, show results, and save some plots in the `results/` folder.

4. **(Optional) Try the Jupyter Notebook**
   - If you like working in notebooks, open `notebooks/salary_analysis.ipynb` with Jupyter and run the cells one by one.

## Project Structure
```
Salary Prediction/
├── data/                # The dataset (CSV file)
├── notebooks/           # Jupyter notebook for step-by-step analysis
├── src/                 # Python scripts (main code here)
├── results/             # Output files and plots (created after running)
├── requirements.txt     # List of Python packages needed
└── README.md            # This file!
```

## About the Data
- **YearsExperience**: How many years someone has worked
- **Salary**: Their annual salary (in USD)
- Only 30 rows, so it's easy to play with

## Example Output
When you run the script, you'll see something like:
```
--- Salary Prediction Project ---
Loading data from: data/Salary_Data.csv
Data loaded! Shape: (30, 2)
First 5 rows:
   YearsExperience   Salary
0             1.1  39343.0
1             1.3  46205.0
...
Training samples: 24, Test samples: 6
Model trained!
Slope: 9449.96, Intercept: 25792.20

Model Performance:
MSE: 37738872.15
RMSE: 6142.11
MAE: 5074.34
R2 Score: 0.9714

Sample Predictions:
Years: 1 -> Predicted Salary: $35,242.16
Years: 3 -> Predicted Salary: $54,142.08
...
Done! Check the 'results' folder for outputs.
```

## Why I Made This
I wanted to learn how to use Python for real-world data science, and this project was a perfect fit for the IBM SkillBuild internship. It's simple, but it covers all the basics: loading data, training a model, checking results, and making predictions.

## Author
- [Your Name]
- Third Year Computer Science Student
- IBM SkillBuild AI Intern

## Thanks!
Big thanks to IBM SkillBuild, Edunet Foundation, and all the mentors who helped out!

---

## Conclusion & Reflection
This project really helped me understand how machine learning works in practice. I learned how to load and explore data, train a model, check how well it works, and make predictions. It was cool to see how a simple linear regression can actually predict real-world things like salaries! Sometimes I got stuck with errors (like file paths or missing packages), but I figured them out by reading error messages and searching online. Overall, it was a fun and valuable experience.

## Future Work
- Try adding more features (like education, job title, or location) to make the predictions even better
- Experiment with other models (like decision trees or polynomial regression)
- Build a simple web app or interface for users to enter their experience and get a salary prediction
- Use a bigger and more diverse dataset

---
If you have any questions or want to suggest improvements, feel free to reach out or open an issue. Happy coding! 🚀
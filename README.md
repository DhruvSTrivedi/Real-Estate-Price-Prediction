# Real Estate Price Prediction

This repository contains a Python script that performs linear regression on real estate data to predict property prices based on their size and construction year. We also examine the significance of each feature in predicting the property prices.

---

## Table of Contents

- [Dependencies](#dependencies)
- [Data](#data)
- [Features](#features)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`

---

## Data

The dataset (`real_estate_price_size_year.csv`) contains property details, including their size in square feet and the year of construction.

---

## Features

1. **Loading and Describing Data**: A quick view into the first few rows of the dataset and its descriptive statistics.
2. **Scaling Features**: Uses StandardScaler from Scikit-learn to scale the features.
3. **Linear Regression**: Implementing a linear regression model and determining coefficients.
4. **Predictions**: Given the size and year of a property, the model predicts its price.
5. **Feature Significance**: Calculate p-values to understand the significance of each feature.

---

## Usage

1. Setting up the environment:
   Ensure that you have all the dependencies installed. You can do this using pip:
   ```
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

2. Dataset:
   Place the `real_estate_price_size_year.csv` in the same directory as your script.

3. Running the script:
   Execute your Python script:
   ```
   python main.py
   ```

4. Expected Output:

   - Initial data head and description.
   - Linear regression model's intercept, coefficients, R-squared, and Adjusted R-squared.
   - Predicted price for an apartment of 750 sq. ft. from the year 2015.
   - Univariate p-values of the variables.
   - Summary of the regression model's findings.
   - Observation on the significance of the 'Year' feature.

---

## Results

- Initial observations show R-squared and adjusted R-squared values.
- We've also provided a predicted price for an example input.
- P-values for each feature are calculated to understand their significance. Based on our findings, the 'Year' feature may not be significant enough to include in our predictive model.

---


---

# Real Estate Price Prediction

This repository contains a Python script that performs linear regression on real estate data to predict property prices based on their size and construction year. We also examine the significance of each feature in predicting the property prices.

## Table of Contents

- [Dependencies](#dependencies)
- [Data](#data)
- [Features](#features)
- [Usage](#usage)
- [Results](#results)
- [Contribute](#contribute)
- [License](#license)

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`

## Data

The dataset (`real_estate_price_size_year.csv`) contains property details, including their size in square feet and the year of construction.

## Features

1. **Loading and Describing Data**: A quick view into the first few rows of the dataset and its descriptive statistics.
2. **Scaling Features**: Uses StandardScaler from Scikit-learn to scale the features.
3. **Linear Regression**: Implementing a linear regression model and determining coefficients.
4. **Predictions**: Given the size and year of a property, the model predicts its price.
5. **Feature Significance**: Calculate p-values to understand the significance of each feature.

## Usage

1. Clone this repository:
   ```
   git clone <repository-link>
   ```
2. Navigate to the repository's directory and run the script:
   ```
   python <script-name>.py
   ```
3. View results and insights.

## Results

- Initial observations show R-squared and adjusted R-squared values.
- We've also provided a predicted price for an example input.
- P-values for each feature are calculated to understand their significance. Based on our findings, the 'Year' feature may not be significant enough to include in our predictive model.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

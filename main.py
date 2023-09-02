# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression

# Setting the aesthetics
sns.set()

# Loading the data
data = pd.read_csv('real_estate_price_size_year.csv')
print(data.head())
print(data.describe())

# Declaring the dependent and independent variables
x = data[['size', 'year']]
y = data['price']

# Scaling the inputs
scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

# Linear Regression
reg = LinearRegression()
reg.fit(x_scaled, y)

# Displaying Intercept, Co-efficient & R-Squared
print("Intercept:", reg.intercept_)
print("Coefficients:", reg.coef_)
print("R-Squared:", reg.score(x_scaled, y))

# Calculating and Displaying Adjusted R-Squared
def adj_r2(x, y):
    r2 = reg.score(x, y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2

print("Adjusted R-Squared:", adj_r2(x_scaled, y))

# Predicting the price of an apartment of size 750 sq.ft. from the year 2015
new_data = [[750, 2015]]
new_data_scaled = scaler.transform(new_data)
print("Predicted Price:", reg.predict(new_data_scaled))

# Calculating the univariate p-values of the variables
f_values, p_values = f_regression(x_scaled, y)
print("P-values:", p_values.round(3))

# Creating a summary of the findings
reg_summary = pd.DataFrame(data=x.columns.values, columns=['Features'])
reg_summary['Coefficients'] = reg.coef_
reg_summary['p-values'] = p_values.round(3)
print(reg_summary)

# Final observation
print("It seems that 'Year' is not even significant, therefore it would be wise to remove it from the model.")

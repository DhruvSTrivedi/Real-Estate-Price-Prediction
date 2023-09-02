import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

data = pd.read_csv('real_estate_price_size_year.csv')
data.head()

data.describe()

#Declaring the dependent and independent variables
x = data[['size','year']]
y = data['price']

#Scaling the inputs
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

#LinearRegression
reg = LinearRegression()
reg.fit(x_scaled,y)

#Intercept, Co-efficient & R-Squared
reg.intercept_
reg.coef_
reg.score(x_scaled,y)

#Adjusted R-Squared
def adj_r2(x,y):
    r2 = reg.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2
adj_r2(x_scaled,y)

#Predictions:- The predicted price of an apartment that has a size of 750 sq.ft. from year 2015.
new_data = [[750,2015]]
new_data_scaled = scaler.transform(new_data)
reg.predict(new_data_scaled)

#Calculating the univariate p-values of the variables
from sklearn.feature_selection import f_regression
f_regression(x_scaled,y)
p_values = f_regression(x,y)[1]
p_values
p_values.round(3)

#Summary of the findings:
reg_summary = pd.DataFrame(data = x.columns.values, columns=['Features'])
reg_summary ['Coefficients'] = reg.coef_
reg_summary ['p-values'] = p_values.round(3)
reg_summary

#It seems that 'Year' is not event significant, therefore it would be wise to remove it from the model.

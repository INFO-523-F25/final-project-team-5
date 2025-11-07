# Importing all the Necessary Modules
import pandas as pd

# Exploratory Data Analysis
# Reading in the csv files
stock_data = pd.read_csv('data/stock_data.csv')
unemployment = pd.read_csv('data/SeriesReport.csv')

# Returning the First Five Rows of the stock_data DataFrame
print(stock_data.head())

# Returning Column Information of the stock_data DataFrame
print(stock_data.info())

# Returning Descriptive Statistics of the stock_data DataFrame
print(stock_data.describe())

# Returning the First Five Rows of the unemployment DataFrame
print(unemployment.head())

# Returning Column Information on the unemployment DataFrame
print(unemployment.info())
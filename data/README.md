# Data
**Dataset #1: 34-year Daily Stock Data (1990-2024)**: This dataset (stock_data.csv) captures historical financial market data and macroeconomic indicators spanning over three decades, from 1990 onwards. It is designed for financial analysis, time series forecasting, and exploring relationships between market volatility, stock indices, and macroeconomic factors. This dataset is particularly relevant for researchers, data scientists, and enthusiasts interested in studying:

* Volatility forecasting (VIX)
* Stock market trends (S&P 500, DJIA, HSI)
* Macroeconomic influences on markets (joblessness, interest rates, etc.)
* The effect of geopolitical and economic uncertainty (EPU, GPRD)

Source: This data has been aggregated from a mix of historical financial records and publicly avaliable macroeconimic datasets:
* VIX (Volatility Index): Chicago Board Options Exchange (CBOE).
* Stock Indices (S&P 500, DJIA, HSI): Yahoo Finance and historical financial databases.
* Volume Data: Extracted from official exchange reports.
* Macroeconomic Indicators: Bureau of Economic Analysis (BEA), Federal Reserve, and other public records.
* Uncertainty Metrics (EPU, GPRD): Economic Policy Uncertainty Index and Global Policy Uncertainty Database.

**Dataset #2: Monthly Unemployment Data (1990-2024):** This dataset (SeriesReport.csv) captures historical unemployment percent data in the United States from 1990-2024.

# Codebook for 34-year Daily Stock Data (1990-2024) Dataset


## Variable Names and Descriptions:

-   **dt**: Date of observation in YYYY-MM-DD format.
-   **vix**: VIX (Volatility Index), a measure of expected market volatility.
-   **sp500**: S&P 500 index value, a benchmark of the U.S. stock market.
-   **sp500_volume**: Daily trading volume for the S&P 500.
-   **djia**: Dow Jones Industrial Average (DJIA), another key U.S. market index.
-   **djia_volume**: Daily trading volume for the DJIA.
-   **hsi**: Hang Seng Index, representing the Hong Kong stock market.
-   **ads**: Aruoba-Diebold-Scotti (ADS) Business Conditions Index, reflecting U.S. economic activity.
-   **us3m**: U.S. Treasury 3-month bond yield, a short-term interest rate proxy.
-   **joblessness**: U.S. unemployment rate, reported as quartiles (1 represents lowest quartile and so on).
-   **epu**: Economic Policy Uncertainty Index, quantifying policy-related economic uncertainty.
-   **GPRD**: Geopolitical Risk Index (Daily), measuring geopolitical risk levels.
-   **prev_day**: Previous day’s S&P 500 closing value, added for lag-based time series analysis.
## Data Types:

-   **dt**: String
-   **vix**: Float
-   **sp500**: Float
-   **sp500_volume**: Float
-   **djia**: Dow Jones Float
-   **djia_volume**: Float
-   **hsi**: Float
-   **ads**: Float
-   **us3m**: Float
-   **joblessness**: Float
-   **epu**: Float
-   **GPRD**: Float
-   **prev_day**: Integer

## Data Source:
https://www.kaggle.com/datasets/shiveshprakash/34-year-daily-stock-data

# Codebook for Monthly Unemployment Data (1990-2024) Dataset


## Variable Names and Descriptions:

-   **Year**: Year of Observation in YYYY format.
-   **Jan**: The Unemployment Rate in the United States in January of the respective year of the record.
-   **Feb**: The Unemployment Rate in the United States in February of the respective year of the record.
-   **Mar**: The Unemployment Rate in the United States in March of the respective year of the record.
-   **Apr**: The Unemployment Rate in the United States in April of the respective year of the record.
-   **May**: The Unemployment Rate in the United States in May of the respective year of the record.
-   **Jun**: The Unemployment Rate in the United States in June of the respective year of the record.
-   **Jul**: The Unemployment Rate in the United States in July of the respective year of the record.
-   **Aug**: The Unemployment Rate in the United States in August of the respective year of the record.
-   **Sep**: The Unemployment Rate in the United States in September of the respective year of the record.
-   **Oct**: The Unemployment Rate in the United States in October of the respective year of the record.
-   **Nov**: The Unemployment Rate in the United States in November of the respective year of the record.
-   **Dec**: The Unemployment Rate in the United States in December of the respective year of the record.
## Data Types:

-   **Year**: Integer (64)
-   **Jan**: Float
-   **Feb**: Float
-   **Mar**: Float
-   **Apr**: Float
-   **May**: Float
-   **Jun**: Float
-   **Jul**: Float
-   **Aug**: Float
-   **Sep**: Float
-   **Oct**: Float
-   **Nov**: Float
-   **Dec**: Float

## Data Source:
https://data.bls.gov/timeseries/LNS14000000


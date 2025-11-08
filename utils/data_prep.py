# Importing all the Necessary Modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Setting the Boxplot Theme
sns.set_theme(style = 'whitegrid')

# Scaling font size
sns.set(font_scale = 1.25)

# Exploratory Data Analysis
# Reading in the csv files
stock_data = pd.read_csv('data/stock_data.csv')
unemployment = pd.read_csv('data/SeriesReport.csv')

# Converting the Observation Date Variable to a Datetime Variable
stock_data['dt'] = pd.to_datetime(stock_data['dt'])

# Unpivoting the Unemployment Data
unemployment_unpivot = unemployment.melt(id_vars='Year', var_name='Month', value_name='Unemployment Percent')

# Extracting the Year and Month from the Observation Date
stock_data['Year'] = stock_data['dt'].dt.year
stock_data['Month'] = stock_data['dt'].dt.month

# Replacing the Month Words with Month Numbers
month_replacement = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

# Apply the mapping
unemployment_unpivot['Month'] = unemployment_unpivot['Month'].map(month_replacement)

# Merging the two DataFrames Together
stock_data_final = pd.merge(stock_data, unemployment_unpivot, on = ['Year', 'Month'], how = 'left')

# Returning the First Five Records
stock_data_final.head()

# Returning Column Information on the DataFrame
stock_data_final.info()

# Returning Descriptive Statistics on the Data
stock_data_final.describe()

# Creating the Boxplots

# Creating a Figure to Hold all Boxplot Subplots
fig1 = plt.figure(figsize = (12, 10))

# Creating the Subplots
ax1 = fig1.add_subplot(3, 2, 1)

ax2 = fig1.add_subplot(3, 2, 2)

ax3 = fig1.add_subplot(3, 2, 3)

ax4 = fig1.add_subplot(3, 2, 4)

ax5 = fig1.add_subplot(3, 2, 5)

# Creating the Boxplot for Volatility
ax1.boxplot(x = stock_data_final['vix'], vert = False, patch_artist = True, boxprops = dict(facecolor = 'blue'))

# Setting the Title and X-Axis Label
ax1.set_title('Volatility')
ax1.set_xlabel('Volatility Index')

# Creating the Boxplot for the U.S. Treasury 3-Month Bond Yield
ax2.boxplot(x = stock_data_final['us3m'], vert = False, patch_artist = True, boxprops = dict(facecolor = 'green'))

# Setting the Title and X-Axis Label
ax2.set_title('U.S. Treasure 3-Month Bond Yield')
ax2.set_xlabel('U.S. Treasure 3-Month Bond Yield')

# Creating the Boxplot for Economic Policy Uncertainty Index
ax3.boxplot(x = stock_data_final['epu'], vert = False, patch_artist = True, boxprops = dict(facecolor = 'red'))

# Setting the Title and X-Axis Label
ax3.set_title('Economic Policy Uncertainty')
ax3.set_xlabel('Economic Policy Uncertainty Index')

# Creating the Boxplot for Geopolitical Risk Index
ax4.boxplot(x = stock_data_final['GPRD'], vert = False, patch_artist = True, boxprops = dict(facecolor = 'purple'))

# Setting the Title and X-Axis Label
ax4.set_title('Geopolitical Risk')
ax4.set_xlabel('Geopolitical Risk Index')

# Creating the Boxplot for Unemployment Percent
ax5.boxplot(x = stock_data_final['Unemployment Percent'], vert = False, patch_artist = True, boxprops = dict(facecolor = 'orange'))

# Setting the Title and X-Axis Label
ax5.set_title('Unemployment')
ax5.set_xlabel('Unemployment Percent')

# Adding Padding to the Graphs
fig1.tight_layout(pad=1.0)

# Displaying the Plot
plt.show()

# Creating the Denisty Plots

# Setting the Density Plot Theme
sns.set_theme(style = 'white')

# Creating a Figure to Hold all Denisty Plot Subplots
fig2 = plt.figure(figsize = (12, 10))

# Creating the Subplots
ax1 = fig2.add_subplot(3, 2, 1)

ax2 = fig2.add_subplot(3, 2, 2)

ax3 = fig2.add_subplot(3, 2, 3)

ax4 = fig2.add_subplot(3, 2, 4)

ax5 = fig2.add_subplot(3, 2, 5)

# Creating the Density Plot for Volatility
sns.kdeplot(stock_data_final['vix'], fill = True, ax = ax1, color = 'blue')

# Setting the Title and X-Axis Label
ax1.set_title('Volatility')
ax1.set_xlabel('Volatility Index')

# Creating the Density Plot for the U.S. Treasury 3-Month Bond Yield
sns.kdeplot(stock_data_final['us3m'], fill = True, ax = ax2, color = 'green')

# Setting the Title and X-Axis Label
ax2.set_title('U.S. Treasure 3-Month Bond Yield')
ax2.set_xlabel('U.S. Treasure 3-Month Bond Yield')

# Creating the Density Plot for Economic Policy Uncertainty Index
sns.kdeplot(stock_data_final['epu'], fill = True, ax = ax3, color = 'red')

# Setting the Title and X-Axis Label
ax3.set_title('Economic Policy Uncertainty')
ax3.set_xlabel('Economic Policy Uncertainty Index')

# Creating the Density Plot for Geopolitical Risk Index
sns.kdeplot(stock_data_final['GPRD'], fill = True, ax = ax4, color = 'purple')

# Setting the Title and X-Axis Label
ax4.set_title('Geopolitical Risk')
ax4.set_xlabel('Geopolitical Risk Index')

# Creating the Density Plot for Unemployment Percent
sns.kdeplot(stock_data_final['Unemployment Percent'], fill = True, ax = ax5, color = 'orange')

# Setting the Title and X-Axis Label
ax5.set_title('Unemployment')
ax5.set_xlabel('Unemployment Percent')

# Adding Padding to the Graphs
fig2.tight_layout(pad=1.0)

# Displaying the Plot
plt.show()

# Creating the Pairplots

# Selecting a Subset of Relevant Features for the purpose of Creating the Pairplot
stock_data_sample = stock_data_final[['vix', 'us3m', 'epu', 'GPRD', 'Unemployment Percent']]

# Creating the plot
plt.figure(figsize = (8, 6))

# Creating the pairplot
stock_data_pairplot = sns.pairplot(stock_data_sample)

# Creating a Plot Title
stock_data_pairplot.fig.suptitle('Correlations Among Predictor Variables', y = 1)

plt.show()

# Creating the Q-Q Plots

# Extracting the Desired Columns
stock_data_sample = stock_data_final[['vix', 'us3m', 'epu', 'GPRD', 'Unemployment Percent']]

# Creating the Columns Variable for the for loop
columns_to_plot = stock_data_sample.columns

# Creating a Dictionary to Map the Variable Names to
column_name_mapping = {
    'vix': 'Volatility Index (VIX)',
    'us3m': 'U.S. Treasury 3-Month Bond Yield',
    'epu': 'Economic Policy Uncertainty Index',
    'GPRD': 'Geopolitical Risk Index',
    'Unemployment Percent': 'Unemployment Percent'
}

# Creating the Figure for the Q-Q Subplots
fig3, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (12, 10))

# Flatten the Axes Array to Iterate over all Subplots
axes_flat = axes.flatten()

# Iterate through columns and corresponding axes and Creating the Q-Q Plots
for k, ax in zip(columns_to_plot, axes_flat):
    # Mapping the Variables to their New Names
    new_name = column_name_mapping.get(k, k)

    # Creating the Q-Q Plots
    sm.qqplot(stock_data_sample[k], line='s', ax=ax)
    ax.set_title(new_name)

# Hiding any Unused Subplots
for ax in axes_flat[len(columns_to_plot):]:
    ax.axis('off')

# Setting the Plot Title
fig3.suptitle('Q-Q Plots')

# Adding Padding to the Graphs
fig3.tight_layout(pad=1.0)

# Displaying the Plots
plt.show()
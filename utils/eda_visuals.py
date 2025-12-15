import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

'''
This module contains functions to create various EDA visualizations such as boxplots, KDE plots, pairplots, and Q-Q plots.
'''

def boxplot_visuals(df, columns_to_plot, column_name_mapping, title):
    '''
    Creates boxplot visuals on the relevant numeric columns.

    Parameters
    ------------
    df (pd.DataFrame):
        The DataFrame that contain the information that the boxplots will be created from.
    columns_to_plot (list):
        The list of columns that will be plotted.
    column_name_mapping (dict):
        A dictionary containing descriptive names that will be used for the subplot titles.
    title (str):
        A string that will be used for the figure title.

    Results
    ------------
    matplotlib.figure.Figure:
        The figure that contains all the boxplot graphs.
    '''
    # Creating the Figure for the Boxplot Subplots
    fig, axes = plt.subplots(nrows = 5, ncols = 3, figsize = (12, 15))

    # Flatten the Axes Array to Iterate over all Subplots
    axes_flat = axes.flatten()

    # Initializing Columns to Plot
    cols = df[columns_to_plot].columns

    # Initializing a Variable to hold a Palette with as many colors as columns
    palette = sns.color_palette("Set2", n_colors=len(cols))

    # Iterate through columns and corresponding axes and Creating the Boxplots
    for j, ax, col_color in zip(columns_to_plot, axes_flat, palette):
        # Mapping the Variables to their New Names
        new_name = column_name_mapping.get(j, j)

        # Creating the Boxplots
        sns.boxplot(x=df[j], ax=ax, color = col_color, vert = False, patch_artist=True, width = 0.2)
        
        # Setting the Title for each Subplot
        ax.set_title(new_name)

    # Hiding any Unused Subplots
    for ax in axes_flat[len(columns_to_plot):]:
        ax.axis('off')

    # Setting the Figure Title
    fig.suptitle(title, y = 1)

    # Adding Padding to the Graphs
    fig.tight_layout(pad=1.0)

    # Displaying the Plots
    plt.show()

def kde_visuals(df, columns_to_plot, column_name_mapping, title):
    '''
    Creates KDE visuals on the relevant numeric columns.

    Parameters
    ------------
    df (pd.DataFrame):
        The DataFrame that contain the information that the KDE plots will be created from.
    columns_to_plot (list):
        The list of columns that will be plotted.
    column_name_mapping (dict):
        A dictionary containing descriptive names that will be used for the subplot titles.
    title (str):
        A string that will be used for the figure title.

    Results
    ------------
    matplotlib.figure.Figure:
        The figure that contains all the KDE graphs.
    '''
    # Creating the Figure for the KDE Subplots
    fig, axes = plt.subplots(nrows = 5, ncols = 3, figsize = (12, 15))

    # Flatten the Axes Array to Iterate over all Subplots
    axes_flat = axes.flatten()

    # Initializing Columns to Plot
    cols = df[columns_to_plot].columns

    # Initializing a Variable to hold a Palette with as many colors as columns
    palette = sns.color_palette("Set2", n_colors=len(cols))

    # Iterate through columns and corresponding axes and Creating the KDE Plots
    for j, ax, col_color in zip(columns_to_plot, axes_flat, palette):
        # Mapping the Variables to their New Names
        new_name = column_name_mapping.get(j, j)

        # Creating the KDE Plots
        sns.kdeplot(df[j], ax=ax, color = col_color, fill = True)
        
        # Setting the Title for each Subplot
        ax.set_title(new_name)

    # Hiding any Unused Subplots
    for ax in axes_flat[len(columns_to_plot):]:
        ax.axis('off')

    # Setting the Figure Title
    fig.suptitle(title, y = 1)

    # Adding Padding to the Graphs
    fig.tight_layout(pad=1.0)

    # Displaying the Plots
    plt.show()

def pairplot_visual(columns_to_plot, column_name_mapping, title):
    '''
    Creates a pairplot graph on the relevant numeric columns.

    Parameters
    ------------
    columns_to_plot (pd.DataFrame):
        The DataFrame of columns that will be plotted.
    title (str):
        A string that will be used for the figure title.

    Results
    ------------
    matplotlib.figure.Figure:
        The figure that contains the pairplot grapha.
    '''
    # Creating the plot
    plt.figure(figsize = (15, 12))

    # Mapping the Variables to their New Names
    plot_df_renamed = columns_to_plot.rename(columns = column_name_mapping)

    # Creating the pairplot
    stock_data_pairplot = sns.pairplot(plot_df_renamed)

    # Setting the Plot Title
    stock_data_pairplot.fig.suptitle(title, y = 1)

    # Adding Padding to the Plots
    plt.tight_layout(pad=1.05)


    plt.show()

def qq_plot(df, columns_to_plot, column_name_mapping, title):
    '''
    Creates Q-Q Plots on the relevant numeric columns.

    Parameters
    ------------
    df (pd.DataFrame):
        The DataFrame that contain the information that the Q-Q plots will be created from.
    columns_to_plot (list):
        The list of columns that will be plotted.
    column_name_mapping (dict):
        A dictionary containing descriptive names that will be used for the subplot titles.
    title (str):
        A string that will be used for the figure title.

    Results
    ------------
    matplotlib.figure.Figure:
        The figure that contains all the Q-Q plots.
    '''
    # Creating the Figure for the Q-Q Subplots
    fig3, axes = plt.subplots(nrows = 5, ncols = 3, figsize = (12, 15))

    # Flatten the Axes Array to Iterate over all Subplots
    axes_flat = axes.flatten()

    # Iterate through columns and corresponding axes and Creating the Q-Q Plots
    for k, ax in zip(columns_to_plot, axes_flat):
        # Mapping the Variables to their New Names
        new_name = column_name_mapping.get(k, k)

        # Creating the Q-Q Plots
        sm.qqplot(df[k], line='s', ax=ax)
        ax.set_title(new_name)

    # Hiding any Unused Subplots
    for ax in axes_flat[len(columns_to_plot):]:
        ax.axis('off')

    # Setting the Figure Title
    fig3.suptitle(title, y = 1)

    # Adding Padding to the Plots
    fig3.tight_layout(pad=1.0)

    # Displaying the Plots
    plt.show()
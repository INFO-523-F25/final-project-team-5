import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

'''
This module contains functions to create various EDA visualizations such as boxplots, KDE plots, pairplots, and Q-Q plots.
'''

def boxplot_visuals(df, columns_to_plot, column_name_mapping, title):
    # Creating the Figure for the Boxplot Subplots
    fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (12, 10))

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
        ax.set_title(new_name)

    # Hiding any Unused Subplots
    for ax in axes_flat[len(columns_to_plot):]:
        ax.axis('off')

    # Setting the Plot Title
    fig.suptitle(title)

    # Adding Padding to the Graphs
    fig.tight_layout(pad=1.0)

    # Displaying the Plots
    plt.show()

def kde_visuals(df, columns_to_plot, column_name_mapping, title):
    # Creating the Figure for the Boxplot Subplots
    fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (12, 10))

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
        sns.kdeplot(df[j], ax=ax, color = col_color, fill = True)
        ax.set_title(new_name)

    # Hiding any Unused Subplots
    for ax in axes_flat[len(columns_to_plot):]:
        ax.axis('off')

    # Setting the Plot Title
    fig.suptitle(title)

    # Adding Padding to the Graphs
    fig.tight_layout(pad=1.0)

    # Displaying the Plots
    plt.show()

def pairplot_visual(columns_to_plot, title):
    # Creating the plot
    plt.figure(figsize = (8, 6))

    # Creating the pairplot
    stock_data_pairplot = sns.pairplot(columns_to_plot)

    # Creating a Plot Title
    stock_data_pairplot.fig.suptitle('Correlations Among Predictor Variables', y = 1)

    plt.title(title)
    plt.show()

def qq_plot(df, columns_to_plot, column_name_mapping, title):
    # Creating the Figure for the Q-Q Subplots
    fig3, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (12, 10))

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

    # Setting the Plot Title
    fig3.suptitle(title)

    # Adding Padding to the Graphs
    fig3.tight_layout(pad=1.0)

    # Displaying the Plots
    plt.show()
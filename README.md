[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/wojP3-_r)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=21215296)
# Team 5 Project Final

Final project repo for INFO 523 - Fall 2025.

## Overview

This repository contains the final project for **INFO-523-F25 (Team 5)**.  
It includes the code, data, and Quarto files developed by our team to explore the relationship between key economic indicators and financial market behavior.

Our analysis focuses on two primary research questions:
1. **How do economic factors — such as unemployment rates and the U.S. Treasury 3-Month Bond Yield — influence volatility within financial markets?**
2. **What insights can historical stock market data provide to help forecast future price movements and assess market volatility?**

Through data exploration, data visualization, and machine learning, we aim to identify patterns and relationships to enhance our understanding of market dynamics and predictive trends.

## Project Structure
**Below is an overview of the repository layout and the purpose of each key file and directory.**\
\
├── .github/ # GitHub Actions and workflow automation configurations\
├── _extra/ # Supplementary or reference materials (not part of main analysis)\
├── _freeze/ # Quarto cache and build artifacts (auto-generated)\
├── data/ # Datasets used in analysis\
├── images/ # Figures and visual assets for reports or presentations\
├── utils/ # Custom Python utility modules for data prep, modeling, and visualization...\
├── .gitignore # Specifies intentionally untracked files to ignore in Git\
├── README.md # Main project overview and documentation (you are here)\
├── _quarto.yml # Quarto configuration file (controls site structure and output)\
├── about.qmd # About page describing the project’s context and objectives\
├── index.qmd # Landing page (main entry point) for the Quarto project\
├── presentation.qmd # Slide deck or presentation generated through Quarto\
├── proposal.qmd # Initial project proposal and planning document\
├── project-final.Rproj # RStudio project file for Quarto/R integration\
└── requirements.txt # Python dependencies required for running the notebooks and
## Branching Conventions
Developers should create feature branches following the format `dev_[lastName]_[issue#]`.
Once development and testing are complete, submit a pull request into main for peer review. Team members will review the changes, offer feedback, and approve the merge once all comments are resolved.
## Dependencies
Python 3.12 or Greater is required for this project. We used the following packages in this project:
- jupyter
- nbformat
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- pmdarima
- tensorflow
- scikit-learn
- prophet
## Authors
- Nicholas Tyler
- John Moran
- Zulemina Cota

#### Disclosure:
Derived from the original data viz course by Mine Çetinkaya-Rundel @ Duke University

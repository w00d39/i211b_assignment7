
## Overview
This Python script, assignment7.py, performs data analysis and visualization on two datasets: the Iris dataset and a debt dataset. It generates multiple plots to provide insights into the data, including bar charts, line plots, and feature importance visualizations using a Random Forest classifier.

The script is divided into two main functions:

 - iris_analysis(): Analyzes the Iris dataset and generates three plots.
 - debt_analysis(): Analyzes a debt dataset and generates three plots.

## Requirements
To run this script, you need the following libraries installed:

# matplotlib
# numpy
# pandas
# scikit-learn

## File Structure
Functions
# iris_analysis()

Analyzes the Iris dataset and generates three plots:
A bar chart comparing the average sepal length and width by species.
A line graph with error bars for sepal length and width by species.
A bar chart showing feature importance using a Random Forest classifier.

# debt_analysis()

Analyzes a debt dataset and generates three plots:
A bar chart showing feature importance using a Random Forest classifier.
A bar chart showing the distribution of loan intent.
A line plot showing

## Outputs
The script will generate six plots in total:

Three plots for the Iris dataset:
1. Average Sepal Length and Width by Species (Bar Chart)
2. Average Sepal Length and Width by Species (Line Plot with Error Bars)
3. Random Forest Feature Importances (Bar Chart)
Three plots for the Debt dataset:
1. Random Forest Feature Importances (Bar Chart)
2. Loan Intent Distribution (Bar Chart)
3. Loan Amount and Customer Income over Term Years (Line Plot)

## Paragraph of Analysis
In the data visualizations for the debt dataset, there are 3 conclusions to be drawn from them.
 The first plot based on a random forest classifier and the most important features it is looking at when deciding
if the loan status will be default or no default. With this trained model, the most important feature it was looking at
was historical_default far above any other feature. The model does perform at 92% accuracy in this case. A
historical default is the biggest predictor in this dataset if there will be another default from that customer.
In the second plot it focuses on the loan intent in the form of a bar graph. With this we see the biggest
reasons for loans are debt consolidation and education. In the third plot we have a line plot focusing on
loan amount and customer income over term years. And through this the pattern that can be seen is that the loan amount
is about 40,000 pounds less than the customer's income, and even with that the patterns are about parallel with each other 
visually with the loan amount not showcasing the peaks and valleys customer income does.


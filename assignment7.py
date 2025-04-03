#libraries and packages being used in this program
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn import datasets, ensemble, model_selection, metrics 


#A

def iris_analysis():
    """
    This function analyzes the iris dataset and creates 3 plots:
    1. A bar chart that compares avg speal lenfth and with by species.
    2. A line graph with error bars for sepal length and width.
    3. A bar chart showing feature importance with a Random Forest classifier.
    """
        # Load the iris dataset
    iris = datasets.load_iris() #loading the iris dataset

    #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    #print(iris.feature_names) #the names of the features in the dataset

    #['setosa', 'versicolor', 'virginica']
    #print(iris.target_names) #the names of the target in the dataset

    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names) #creating a dataframe from the iris dataset

    iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names) #adding the target to the dataframe

    iris_species = iris_df.groupby('species', observed = False) #grouping the data by species and getting the mean of each group

    fig = plt.figure(figsize=(15, 12)) 
    fig.suptitle("Iris Species Analysis", fontsize=22, y=0.98) 

    #1: Sepal Length vs Sepal Width
    colors1 = ['#88ECB6', '#888CEC', '#FF5733']

    avg_sepal_length = iris_species['sepal length (cm)'].mean() #the average sepal length of each species
    avg_sepal_width = iris_species['sepal width (cm)'].mean() #the average sepal width of each species

    sd_sepal_length = iris_species['sepal length (cm)'].std() #the standard deviation of the sepal length of each species
    sd_sepal_width = iris_species['sepal width (cm)'].std() #the standard deviation of the sepal width of each species

    avg_iris_df = pd.DataFrame({'species': avg_sepal_length.index, #gets species names
                                'avg_sepal_length': avg_sepal_length.values, #avg sepal length
                                'avg_sepal_width': avg_sepal_width.values, #avg sepal width
                                'sd_sepal_length': sd_sepal_length.values, #standard deviation of sepal length
                                'sd_sepal_width': sd_sepal_width.values #standard deviation of sepal width
                                })


    ax1 = fig.add_subplot(2, 2, 1) #first plot

    bar_width = 0.35 #bar width
    x = np.arange(len(avg_iris_df['species']))  # X positions for the bars

    # Create bar charts for sepal length and width and also formats them with error bars being the standard deviation
    ax1.bar(x - bar_width / 2, avg_iris_df['avg_sepal_length'], bar_width, 
            yerr=avg_iris_df['sd_sepal_length'], capsize=5, color='#88ECB6', label='Sepal Length')
    ax1.bar(x + bar_width / 2, avg_iris_df['avg_sepal_width'], bar_width,
            yerr=avg_iris_df['sd_sepal_width'], capsize=5, color='#888CEC', label='Sepal Width')

    # Add labels, title, and legend
    ax1.set_title('Average Sepal Length and Width by Species')
    ax1.set_xlabel('Species')
    ax1.set_ylabel('Centimeters')
    ax1.set_xticks(x)
    ax1.set_xticklabels(avg_iris_df['species'])
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)


   

    #2: Sepal Length vs  Sepall Width as line graph
    ax2 = fig.add_subplot(2, 2, 2) #second plot

    # Line plots for sepal length and width
    ax2.plot(avg_iris_df['species'], avg_iris_df['avg_sepal_length'], marker='o', color='#FF5733', label='Sepal Length')
    ax2.plot(avg_iris_df['species'], avg_iris_df['avg_sepal_width'], marker='s', color='#88ECB6', label='Sepal Width')

    # Add error bars 
    ax2.errorbar(avg_iris_df['species'], avg_iris_df['avg_sepal_length'], yerr=avg_iris_df['sd_sepal_length'],
                 fmt='o', color='#FF5733', capsize=5)
    ax2.errorbar(avg_iris_df['species'], avg_iris_df['avg_sepal_width'], yerr=avg_iris_df['sd_sepal_width'],
                    fmt='s', color='#88ECB6', capsize=5)
    

    ax2.set_ylim(0, None) # Set y-axis limit to start from 0

    # Add labels, title, and legend
    ax2.set_title('Average Sepal Length and Width by Species')
    ax2.set_xlabel('Species')
    ax2.set_ylabel('Centimeters')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)


    #3: Feature Importance with Random Forest
    ax3 = fig.add_subplot(2, 2, 3) #third plot

    colors3 = ['#F7BAE5', '#F7CCBA', '#C6F7BA','#F7EBBA'] #colors for horizontal bar chart for each feature

    #random forest classifier
    x_data = iris.data #data
    y_data = iris.target #target

    #splitting the data into training and testing sets with an 80/20 split and a random state of 301
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.2, random_state=301)

    #random forest
    iris_forest = ensemble.RandomForestClassifier(
        n_estimators=75, random_state=301
    )

    iris_forest.fit(x_train, y_train) #fitting the data to the model
    iris_forest_predictions = iris_forest.predict(x_test) #training the model

    #graph time
    feature_importances = iris_forest.feature_importances_ #feature importances
    feature_names = iris.feature_names #feature names

    ax3.barh(feature_names, feature_importances, color=colors3) #horizontal bar chart
    ax3.set_xlabel('Feature Importance')
    ax3.set_title('Random Forest Feature Importances')
    ax3.set_xlim(0, 1)
    ax3.grid(axis='x', linestyle='--', alpha=0.7)
  
    plt.tight_layout(rect=[0, 0, 1, 0.95]) #fixing the layout so we dont smush the plots together
    return plt.show() #retuning the plot to be shown

#B
def debt_analysis():
    """
    This function analyzes a debt dataset and creates 3 plots:
    1. A bar chart showing feature importance with a Random Forest classifier.
    2. A bar chart showing the distribution of loan intent.
    3. A line plot showing the average loan amount and customer income over term years.
    """
    color1 = ['#F7BAE5', '#88ECB6', '#888CEC', '#FF5733', '#F7BAE5', '#F7CCBA', '#C6F7BA','#F7EBBA'] #colors for the plots

    debt_df = pd.read_csv("LoanDataset - LoansDatasest.csv") #loading the dataset
    debt_df.columns = debt_df.columns.str.strip() #removing any whitespace from the column names

    loan_intent_labels = debt_df['loan_intent'].astype('category').cat.categories #loan intent labels preserved for later

    #This will be to clean and prep the data because there's a couple things that will mess it up and cause errors
    debt_df.dropna(inplace=True) #get rid of any rows with missing values
    cat_columns = ['home_ownership', 'loan_intent','loan_grade', 'historical_default', 'Current_loan_status'] #categorical columns
    num_columns = ['customer_age', 'customer_income', 'employment_duration', 'loan_amnt', 'loan_int_rate', 'term_years', 'cred_hist_length'] #numerical columns

    # Convert categorical variables to numerical codes in a for loop to cut down on lines
    for col in cat_columns:
        if col in debt_df.columns: # Check if the column exists in the DataFrame
            debt_df[col] = debt_df[col].astype('category').cat.codes # Convert to categorical codes
        else:
            print(f"Column '{col}' not found in the DataFrame.") # Print a message if the column is not found
    #Deal with the pound symbol and comma in the loan amount column
    if 'loan_amnt' in debt_df.columns:
        debt_df['loan_amnt'] = debt_df['loan_amnt'].str.replace('£', '', regex=False)  # Remove the £ symbol
        debt_df['loan_amnt'] = debt_df['loan_amnt'].str.replace(',', '', regex=False)  # Remove commas
    #convert any numeric columns to numeric values instead of strings in a for loop
    for col in num_columns: 
        if col in debt_df.columns:
            debt_df[col] = pd.to_numeric(debt_df[col], errors='coerce') # Convert to numeric, forcing errors to NaN

    debt_df.dropna(inplace=True) #get rid of any rows with missing values after coercing to numeric

     # Verify 'Current_loan_status' exists
    if 'Current_loan_status' not in debt_df.columns: # Check if the column exists bc it is the target
        raise KeyError("The column 'Current_loan_status' was not found in the dataset.")

    #plot 1

    x_data = debt_df.drop(columns=['Current_loan_status']) #data
    y_data = debt_df['Current_loan_status'] #target

    #splitting the data into training and testing sets with an 80/20 split and a random state of 301
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.2, random_state=301)
    #random forest
    debt_forest = ensemble.RandomForestClassifier(
        n_estimators=75, random_state=301
    )
    debt_forest.fit(x_train, y_train) #fitting the data to the model
    debt_forest_predictions = debt_forest.predict(x_test) #training the model

    feature_importances = debt_forest.feature_importances_ #feature importances
    feature_names = x_data.columns #feature names

    # Create a bar plot for feature importances
    fig = plt.figure(figsize=(15, 12)) #plot holder
    fig.suptitle("Debt Analysis", fontsize=22, y=0.98)
    ax1 = fig.add_subplot(2, 2, 1) #plot one

    ax1.barh(feature_names, feature_importances, color=color1) #horizontal bar chart
    ax1.set_title('Feature Importances') 
    ax1.set_xlabel('Importance')
    ax1.set_ylabel('Features')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)


    #plot 2 about loan intent as a bar chart
    value_counts = debt_df['loan_intent'].value_counts() #this will be the data for the bar chart counting up each loan intent

    ax2 = fig.add_subplot(2, 2, 2) #plot two
    #loan intent labels come from when the dataset was originally loaded before the data was cat coded
    ax2.bar(loan_intent_labels, value_counts.values, color=color1[:len(value_counts)], edgecolor='black') #bar chart
    ax2.set_title('Loan Intent Distribution')
    ax2.set_xlabel('Loan Intent')
    ax2.set_ylabel('Count')
    ax2.set_xticks(range(len(value_counts))) # Set x-ticks to match the number of loan intents
    ax2.set_xticklabels(loan_intent_labels, rotation=45, ha='right')  # Rotate labels and align to the right to fix the overlap
    ax2.grid(axis='y', linestyle='--', alpha=0.7)


    #plot 3
    ax3 = fig.add_subplot(2, 2, 3) #third plot
    # Creates line plot for loan amount over time
    # The second plot is for customer income over time
    grouped_data = debt_df.groupby('term_years').agg({'loan_amnt': 'mean', 'customer_income': 'mean'}).reset_index() #aggregate the data by term years and get the mean of each group
    ax3.plot(grouped_data['term_years'], grouped_data['loan_amnt'], marker='o', color=color1[0], label='Loan Amount') #plots the loan amount
    ax3.plot(grouped_data['term_years'], grouped_data['customer_income'], marker='s', color=color1[1], label='Customer Income') #plots the customer income

    # Add labels, title, and legend
    ax3.set_title('Loan Amount and Customer Income over Term Years')
    ax3.set_xlabel('Term Years')
    ax3.set_ylabel('Values')
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.7)


    return plt.show()

iris_analysis()

debt_analysis()
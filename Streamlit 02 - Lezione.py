
# The objective of this notebook is to work on a complete data science project and to create a Streamlit web application to present it interactively.
# (a) Open VSCode. Create a Python file called streamlit_app.py. Save this file in the same folder as the train.csv file.


# (b) In the streamlit_app.py file, import the Streamlit library and the necessary data exploration and DataVizualization libraries.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def wide_space_default():
    st.set_page_config(layout="wide")

wide_space_default()


# (c) Create a dataframe called df to read the file train.csv.
path = r"G:\Il mio Drive\Colab Notebooks\DS-WorldTemperature Proj\Coding\Part 5 - Streamlit\train.csv"
df=pd.read_csv(path)


# (d) Copy the following code lines to add a title and to create 3 pages called "Exploration", "DataVizualization" and "Modelling" on Streamlit.
st.title("Titanic : binary classification project")
st.sidebar.title("Table of contents")
pages=["Exploration", "DataVizualization", "Modelling"]
page=st.sidebar.radio("Go to", pages)


#(a) Write "Presentation of data" at the top of the first page using the streamlit command st.write() in the Python script.
if page == pages[0] : 
    st.write("### Presentation of data")
    st.write("Below is the first 10 lines of the dataframe")
    
    # (c) Display the first 10 lines of df on the web application Streamlit by using the method st.dataframe().
    st.dataframe(df.head(10))
    # (d) Display informations about the dataframe on the Streamlit web application using the st.write() method 
    # in the same way as a print and the st.dataframe() method for a dataframe.
    st.write("The shape of the dataframe is as follows: ")
    st.write(df.shape)
    st.write("The describe method applied to the df gives the following information")
    st.dataframe(df.describe())

    #(e) Create a checkbox to choose whether to display the number of missing values or not, using the st.checkbox() method.
    if st.checkbox("Show NA") :
        st.write("In the DataFrame the following NaN Values are present in the following columns: ")
        st.dataframe(df.isna().sum())


# --------------------------------- DataVizualization PART
# We focus now on the second page of the Streamlit. The objective is to graphically analyse the data with DataVizualization, according to different axes of study.
# (a) Write "DataVizualization" at the top of the second page using the st.write() command in the Python script.
if page == pages[1] : 
    st.write("### DataVizualization")

    # We are interested in the target variable "Survived". This variable takes 2 modalities: 0 if the individual did not survive and 1 if the individual survived.
    #  (b) Display in a plot the distribution of the target variable.
    fig = plt.figure(figsize=(10,10))
    sns.countplot(x = 'Survived', data = df, hue="Survived")
    plt.title("Number of Survied People vs Not Survived")
    st.pyplot(fig)
    # (c) Display plots to describe the Titanic passengers. Add titles to the plots.
    fig = plt.figure()
    sns.countplot(x = 'Sex', data = df, hue="Sex")
    plt.title("Distribution of the passengers gender")
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x = 'Pclass', data = df, hue="Pclass");
    plt.title("Distribution of the passengers class")
    st.pyplot(fig)

    fig = sns.displot(x = 'Age', data = df, hue="Age")
    plt.title("Distribution of the passengers age")
    st.pyplot(fig)

    # Then we analyse the impact of the different factors on the survival or not of passengers.

    # (d) Display a countplot of the target variable according to the gender.

    fig = plt.figure()
    plt.title("Count of Target Variable related to the gender")
    sns.countplot(x = 'Survived', hue='Sex', data = df)
    st.pyplot(fig)

    # (e) Display a plot of the target variable according to the classes.
    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    plt.title("Category Plot based on the Pclass")
    st.pyplot(fig)

    # (f) Display a plot of the target variable according to the age.
    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    plt.title("Line plot of passenger survived based on their age and Travelling class")
    st.pyplot(fig)

    # Finally we conclude the multivariate analysis by looking at the correlations between the variables.
    # (g) Display the correlation matrix of the explanatory variables.
    fig, ax = plt.subplots()
    dfnumeric = df.select_dtypes(include=["int", "float"]) 
    corr_matrix = dfnumeric.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Matrix between numeric only variables")
    st.write(fig)


# --------------------------------- MODELLING PART
# Finally, we move on to the Modelling step. We do binary classification to predict whether a passenger survives to the Titanic or not.
if page == pages[2] : 
    # (a) Write "Modelling" at the top of the third page using the st.write() command in the Python script.
    st.write("### Modelling")


# (b) In the Python script streamlit_app.py, remove the irrelevant variables (PassengerID, Name, Ticket, Cabin).
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# (c) In the Python script, create a variable y containing the target variable. 
# Create a dataframe X_cat containing the categorical explanatory variables and a dataframe 
# X_num containing the numerical explanatory variables.
y = df['Survived']
X_cat = df[['Pclass', 'Sex',  'Embarked']]
X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

#  (d) In the Python script, replace the missing values for categorical variables by the mode and 
# replace the missing values for numerical variables by the median.
for col in X_cat.columns:
    X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
for col in X_num.columns:
    X_num[col] = X_num[col].fillna(X_num[col].median())

# (e) In the Python script, encode the categorical variables.
X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)

# (f) In the Python script, concatenate the encoded explanatory variables without missing values to obtain a clean X dataframe.
X = pd.concat([X_cat_scaled, X_num], axis = 1)

# (g) In the Python script, separate the data into a train set and a test set using the train_test_split function from the Scikit-Learn model_selection package.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# (h) In the Python script, standardize the numerical values using the StandardScaler function from the Preprocessing package of Scikit-Learn.
scaler = StandardScaler()
X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

# (i) In the Python script, create a function called "prediction" which takes 
# the name of a classifier as an argument and which returns the trained classifier.

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC # Another calssification method?
from sklearn.metrics import confusion_matrix


def prediction(classifier):
    if classifier == 'Random Forest':
        clf = RandomForestClassifier()
    elif classifier == 'SVC':
        clf = SVC()
    elif classifier == 'Logistic Regression':
        clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf

# It creates a function which returns either the accuracy or the confusion matrix.
def scores(clf, choice):
    if choice == 'Accuracy':
        return clf.score(X_test, y_test)
    elif choice == 'Confusion matrix':
        return confusion_matrix(y_test, clf.predict(X_test))

# We create a "select box" to choose which classifier to train.
# (j) In the Python script, use the st.selectbox() method to choose between the RandomForest classifier, the SVM classifier and the LogisticRegression classifier. 
# Then return to the Streamlit web application to view the select box.

if page == pages[2] : 
    choice = ['Random Forest', 'SVC', 'Logistic Regression']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)

    # The st.radio() method displays checkboxes to choose between many options. 
    # You can play with the selectbox and the checkboxes to see the classification results of the different models interactively. 
    clf = prediction(option)
    display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(clf, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(clf, display))

    import pickle
    pickle.dump(clf, open("model", 'wb'))
# Detecting attempted credit card fraud

## Introduction
In this project I take a look at credit card information and create a model to predict fraud. The dataset can be found <a href=https://www.kaggle.com/mlg-ulb/creditcardfraud>here</a> on kaggle.com. 

## How to navigate this project
Here I will show you how to navigate this project and follow me through my project workflow.
<img src='Images/Data_Science_Process.png' height=200 width=200>



#### In this directory
1) Table_of_Contents.ipynb is a file giving an overview of this project.
2) creditcard.csv is the original dataset and can be found in this directory.
3) Data_Cleaning.ipynb is a notebook where I go through cleaning the data.
4) Cleaned_data.csv is the product of my data cleaning.
5) Exploritory_Data_Analysis.ipynb is a notebook where I go through exploritory data analysis.
6) Feature_Engineering.ipynb is a notebook where I do Feature Engineering.
7) Modeling is a folder containing notebooks going over machine learning models as well as data from the feature engineering stage and functions that I made.
8) Executive_Summary.pdf is a pdf of a google slides presentation summarizing this project.
9) Images is a folder containing images I used throughout this project.

#### In Modeling
The folowing notebooks are from the modeling stage of my project. The names of the files show what machine learning algorithm was used
1) Logistic_Regression.ipynb
2) Support_Vector_Machine.ipynb
3) K_Nearest_Neighbors.ipynb
4) Decision_Tree.ipynb
5) Random_Forest.ipynb

#### Tools used
##### From sklearn
MinMaxScaler <br>
StandardScaler<br>
RobustScaler<br>
LogisticRegression<br>
SVC<br>
KNeighborsClassifier<br>
DecisionTreeClassifier<br>
RandomForestClassifier<br>
PCA<br>
classification_report<br>

##### From imblearn
SMOTE<br>
NearMiss<br>

##### Others
numpy<br>
pandas<br>
time<br>
pickle<br>
matplotlib<br>
seaborn<br>
scipy.stats<br>

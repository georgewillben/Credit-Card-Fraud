# Detecting attempted credit card fraud

## Introduction
In this project I take a look at credit card information and create a model to predict fraud. The dataset can be found <a href=https://www.kaggle.com/mlg-ulb/creditcardfraud>here</a> on kaggle.com. 

## How to navigate this project
Here I will show you how to navigate this project and follow me through my project workflow.
<br>
<img src='Images/Data_Science_Process.png' height=200 width=200>
<br><br>



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
2) K_Nearest_Neighbors.ipynb
3) Random_Forest.ipynb
4) XGBoost.ipynb

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
xgboost <br>

## Data Source
I found this dataset on <a href=https://www.kaggle.com/mlg-ulb/creditcardfraud>kaggle</a>

## Exploratory Data Analysis
During the data analysis stage I used critical thinking and data visualizations to come up with some insights in the data. I found if a transaction occurs more than once it is much more likely to be fraud, and that the ratio of fraud to non-fraud transactions raises during some hours of the day.

## Feature Engineering
My approach to feature engineering was to first try four different scaling techniques. Then I resampled the data with NearMiss to make it balanced. After that I had four versions of the data, I tried removing outliers so that gave me eight versions of the data. Finally I took those eight versions of the data, duplicated them, and performed principal component analysis on the duplicates, giving me sixteen versions of the data.

## Modeling
I tried several machine learning models listed above. I created a function to perform hyper parameter tuning it takes in the desired scaler based on a string as a parameter as well as parameters that tell it whether to remove outliers and/or implement principal component analysis. The function prints out every combination of parameters with their cross validated recall and precision scores. The reason I did it this way is because I wanted to have a high recall and precision, but recall was much more important than precision.

## Final Model
After trying out several classifiers and hyper parameter combinations I created a support vector machine with %79 recall and %63 precision when cross validated on the data.

# Sources 
1) Kaggle https://www.kaggle.com/mlg-ulb/creditcardfraud

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective, Expert systems with applications,41,10,4915-4928,2014, Pergamon

Dal Pozzolo, Andrea; Boracchi, Giacomo; Caelen, Olivier; Alippi, Cesare; Bontempi, Gianluca. Credit card fraud detection: a realistic modeling and a novel learning strategy, IEEE transactions on neural networks and learning systems,29,8,3784-3797,2018,IEEE

Dal Pozzolo, Andrea Adaptive Machine learning for credit card fraud detection ULB MLG PhD thesis (supervised by G. Bontempi)

Carcillo, Fabrizio; Dal Pozzolo, Andrea; Le Borgne, Yann-Aël; Caelen, Olivier; Mazzer, Yannis; Bontempi, Gianluca. Scarff: a scalable framework for streaming credit card fraud detection with Spark, Information fusion,41, 182-194,2018,Elsevier

Carcillo, Fabrizio; Le Borgne, Yann-Aël; Caelen, Olivier; Bontempi, Gianluca. Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization, International Journal of Data Science and Analytics, 5,4,285-300,2018,Springer International Publishing

Bertrand Lebichot, Yann-Aël Le Borgne, Liyun He, Frederic Oblé, Gianluca Bontempi Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection, INNSBDDL 2019: Recent Advances in Big Data and Deep Learning, pp 78-88, 2019

Fabrizio Carcillo, Yann-Aël Le Borgne, Olivier Caelen, Frederic Oblé, Gianluca Bontempi Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection Information Sciences, 2019

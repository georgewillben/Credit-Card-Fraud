# Detecting attempted credit card fraud

## Introduction
In this project I take a look at credit card information and create a model to predict fraud. The dataset can be found <a href=https://www.kaggle.com/mlg-ulb/creditcardfraud>here</a> on kaggle.com. 

## How to navigate this project
Here I will show you how to navigate this project and follow me through my project workflow. The project Folders are numbered.<br>
* 1 Business Understanding.pdf I go through why this project matters from a business perspective <br>
* 2 Data Understanding.pdf I talk about the nature of the data and where to find it
* 3 Data Cleaning.ipynb I clean the data up and make sure it is suitable for further analysis
* 4 Explorartory Data Analysis.ipynb I use visualization and statistical tests to learn from the data 
* 5 Modeling.ipynb I develope machine learning models capable of predicting fraud

<br><br>


#### Tools used
* numpy <br>
* pandas <br>
* matplotlib <br>
* seaborn <br>
* scipy <br>
* imblearn <br>
* sklearn

## Data Cleaning
In this stage I inspect the data for missing values, adnormal values, and unnecessary data. I found the data mostly clean and ready to use. The "Time" feature would be innapporpriate to use because I only had data take over 58 hours so I removed it. I also removed some duplicate rows and used a stratified split to seperate training, validation, and test sets.

## Exploratory Data Analysis
During the data analysis stage I used critical thinking and data visualizations to come up with some insights in the data. I found that the amount of a transaction does not indicate whether or not it is fraudlent. Also there where several columns that were highly indicative of fraud.

## Modeling
In the modeling stage I used machine learning techniques to create a model to predict fraud. I experimented with logistic regression and random forest models. I tried several techniques such as recursive feature elimitation, principal component analysis, resampling with SMOTE, and hyper-parameter tuning. The final model was a random forest that achieved %79 recall and %34 precision. Meaning that %79 of frauds were detected and that the model issued a false flag for fraudulent activity twice for every real fraud.

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

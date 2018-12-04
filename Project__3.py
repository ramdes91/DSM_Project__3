
# coding: utf-8

# Problem Statement
# 
# This data was extracted from the census bureau database found at
# http://www.census.gov/ftp/pub/DES/www/welcome.html
# 
# Donor: Ronny Kohavi and Barry Becker,
# 
# Data Mining and Visualization
# Silicon Graphics.
# e-mail: ronnyk@sgi.com for questions.
# 
# Split into train-test using MLC++ GenCVFiles (2/3, 1/3 random).
# 48842 instances, mix of continuous and discrete (train=32561, test=16281)
# 45222 if instances with unknown values are removed (train=30162, test=15060)
# 
# Duplicate or conflicting instances : 6
# 
# Class probabilities for adult.all file
# 
# Probability for the label '>50K' : 23.93% / 24.78% (without unknowns)
# 
# Probability for the label '<=50K' : 76.07% / 75.22% (without unknowns)
# 
# Extraction was done by Barry Becker from the 1994 Census database. A set of
# reasonably clean records was extracted using the following conditions:
# ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0)) 
# 
# Prediction task is to determine whether a person makes over 50K a year. Conversion of original data as
# follows:
# 1. Discretized a gross income into two ranges with threshold 50,000.
# 2. Convert U.S. to US to avoid periods.
# 3. Convert Unknown to "?"
# 4. Run MLC++ GenCVFiles to generate data,test.
# 
# Description of fnlwgt (final weight)
# The weights on the CPS files are controlled to independent estimates of the civilian
# noninstitutional population of the US. These are prepared monthly for us by Population
# Division here at the Census Bureau. We use 3 sets of controls.
# These are:
# 1. A single cell estimate of the population 16+ for each state.
# 2. Controls for Hispanic Origin by age and sex.
# 3. Controls by Race, age and sex.
# 
# We use all three sets of controls in our weighting program and "rake" through them 6
# times so that by the end we come back to all the controls we used.
# The term estimate refers to population totals derived from CPS by creating "weighted
# tallies" of any specified socio-economic characteristics of the population. People with
# similar demographic characteristics should have similar weights. There is one important
# caveat to remember about this statement. That is that since the CPS sample is actually a
# collection of 51 state samples, each with its own probability of selection, the statement
# only applies within state.
# 
# Dataset Link
# https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
# 
# Problem 1:
# Prediction task is to determine whether a person makes over 50K a year.
# 
# Problem 2:
# Which factors are important\n
# 
# Problem 3:
# Which algorithms are best for this dataset
# 

# In[1]:


#Fetching Data
#Import Package and Data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


income_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data3."
names =  ['age', 'workclass', 'fnlwgt', 
           'education', 'education-num', 
           'marital-status', 'occupation', 
           'relationship', 'race', 'sex', 
           'capital-gain', 'capital-loss', 
           'hours-per-week', 'native-country', 
           'predclass']
income_df = pd.read_csv(income_data, na_values=[" ?"],
                         header=None, 
                         names = names)
income_df.head()


# In[3]:


income_df.describe()


# In[4]:


#Data Cleaning
#Dealing with Missing Value
income_df.isnull().sum()


# In[5]:


#Attributes workclass, occupation, and native-country most NAs. Let's drop these NA.
income_df.age = income_df.age.astype(float)
income_df['hours-per-week'] = income_df['hours-per-week'].astype(float)
my_df = income_df.dropna()
my_df.info()


# In[6]:


my_df.isnull().sum()


# In[7]:


print('workclass',my_df.workclass.unique())
print('education',my_df.education.unique())
print('marital-status',my_df['marital-status'].unique())
print('occupation',my_df.occupation.unique())
print('relationship',my_df.relationship.unique())
print('race',my_df.race.unique())
print('sex',my_df.sex.unique())
print('native-country',my_df['native-country'].unique())
print('predclass',my_df.predclass.unique())


# In[8]:


fig = plt.figure(figsize=(20,1))
plt.style.use('seaborn-ticks')
sns.countplot(y="predclass", data=my_df, order=my_df['predclass'].value_counts().index)


# Income level less than 50K is more than 3 times of those above 50K, indicating that the the dataset is somewhat skewed. However, since there is no data on the upper limit of adult's income above 50K, it's premature to conclude that the total amount of wealth are skewed towards high income group.

# In[9]:


#Education
my_df['education'].replace(' Preschool', 'dropout',inplace=True)
my_df['education'].replace(' 10th', 'dropout',inplace=True)
my_df['education'].replace(' 11th', 'dropout',inplace=True)
my_df['education'].replace(' 12th', 'dropout',inplace=True)
my_df['education'].replace(' 1st-4th', 'dropout',inplace=True)
my_df['education'].replace(' 5th-6th', 'dropout',inplace=True)
my_df['education'].replace(' 7th-8th', 'dropout',inplace=True)
my_df['education'].replace(' 9th', 'dropout',inplace=True)
my_df['education'].replace(' HS-Grad', 'HighGrad',inplace=True)
my_df['education'].replace(' HS-grad', 'HighGrad',inplace=True)
my_df['education'].replace(' Some-college', 'CommunityCollege',inplace=True)
my_df['education'].replace(' Assoc-acdm', 'CommunityCollege',inplace=True)
my_df['education'].replace(' Assoc-voc', 'CommunityCollege',inplace=True)
my_df['education'].replace(' Bachelors', 'Bachelors',inplace=True)
my_df['education'].replace(' Masters', 'Masters',inplace=True)
my_df['education'].replace(' Prof-school', 'Masters',inplace=True)
my_df['education'].replace(' Doctorate', 'Doctorate',inplace=True)


# In[10]:


my_df[['education', 'education-num']].groupby(['education'], as_index=False).mean().sort_values(by='education-num', ascending=False)


# In[11]:


fig = plt.figure(figsize=(20,3))
plt.style.use('seaborn-ticks')
sns.countplot(y="education", data=my_df, order=my_df['education'].value_counts().index)


# In[12]:


#Marital-status¶
my_df['marital-status'].replace(' Never-married', 'NotMarried',inplace=True)
my_df['marital-status'].replace([' Married-AF-spouse'], 'Married',inplace=True)
my_df['marital-status'].replace([' Married-civ-spouse'], 'Married',inplace=True)
my_df['marital-status'].replace([' Married-spouse-absent'], 'NotMarried',inplace=True)
my_df['marital-status'].replace([' Separated'], 'Separated',inplace=True)
my_df['marital-status'].replace([' Divorced'], 'Separated',inplace=True)
my_df['marital-status'].replace([' Widowed'], 'Widowed',inplace=True)


# In[13]:


fig = plt.figure(figsize=(20,2))
plt.style.use('seaborn-ticks')
sns.countplot(y="marital-status", data=my_df, order=my_df['marital-status'].value_counts().index)


# In[14]:


#Occupation
plt.style.use('seaborn-ticks')
plt.figure(figsize=(20,3)) 
sns.countplot(y="workclass", data=my_df, order=my_df['workclass'].value_counts().index)


# In[15]:


# make the age variable discretized 
my_df['age_bin'] = pd.cut(my_df['age'], 20)


# In[16]:


my_df[['predclass', 'age']].groupby(['predclass'], as_index=False).mean().sort_values(by='age', ascending=False)


# In[17]:


#Race

plt.style.use('seaborn-whitegrid')
x, y, hue = "race", "prop", "sex"
#hue_order = ["Male", "Female"]
plt.figure(figsize=(20,5)) 
f, axes = plt.subplots(1, 2)
sns.countplot(x=x, hue=hue, data=my_df, ax=axes[0])

prop_df = (my_df[x]
           .groupby(my_df[hue])
           .value_counts(normalize=True)
           .rename(y)
           .reset_index())

sns.barplot(x=x, y=y, hue=hue, data=prop_df, ax=axes[1])


# In[18]:


#Hours of work
# Let's use the Pandas Cut function to bin the data in equal buckets
my_df['hours-per-week_bin'] = pd.cut(my_df['hours-per-week'], 10)
my_df['hours-per-week'] = my_df['hours-per-week']


# In[19]:


#Create a crossing feature: Age + hour of work
g = sns.jointplot(x = 'age', 
              y = 'hours-per-week',
              data = my_df, 
              kind = 'hex', 
              cmap= 'hot', 
              size=10)

sns.regplot(my_df.age, my_df['hours-per-week'], ax=g.ax_joint, scatter=False, color='grey')


# In[20]:


# Crossing Numerical Features
my_df['age-hours'] = my_df['age']*my_df['hours-per-week']
my_df['age-hours_bin'] = pd.cut(my_df['age-hours'], 10)

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
sns.distplot(my_df[my_df['predclass'] == ' >50K']['age-hours'], kde_kws={"label": ">$50K"})
sns.distplot(my_df[my_df['predclass'] == ' <=50K']['age-hours'], kde_kws={"label": "<$50K"})


# In[21]:


#age vs. income level
plt.style.use('seaborn-ticks')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="age_bin", data=my_df)
plt.subplot(1, 2, 2)
sns.distplot(my_df[my_df['predclass'] == ' >50K']['age'], kde_kws={"label": ">$50K"})
sns.distplot(my_df[my_df['predclass'] == ' <=50K']['age'], kde_kws={"label": "<=$50K"})


# In[22]:


#Working hour vs. income level
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
plt.subplot(1, 2, 1)
sns.countplot(y="hours-per-week_bin", data=my_df, order=my_df['hours-per-week_bin'].value_counts().index);
plt.subplot(1, 2, 2)
sns.distplot(my_df['hours-per-week']);
sns.distplot(my_df[my_df['predclass'] == ' >50K']['hours-per-week'], kde_kws={"label": ">$50K"})
sns.distplot(my_df[my_df['predclass'] == ' <=50K']['hours-per-week'], kde_kws={"label": "<$50K"})
plt.ylim(0, None)
plt.xlim(20, 60)


# In[23]:


from matplotlib import pyplot
a4_dims = (20, 5)
fig, ax = pyplot.subplots(figsize=a4_dims)
ax = sns.violinplot(x="occupation", y="age", hue="predclass",
                    data=my_df, gridsize=100, palette="muted", split=True, saturation=0.75)
ax


# The general trend is in sync with common sense: more senior workers have higher salaries. Armed-forces don't have a high job salaries.
# 
# Interestingly, private house sevice has the widest range of age variation, however, the payment is no higher than 50K, indicating that senority doesn't give rise to a higher payment comparing to other jobs.

# In[24]:


#Bivariate Analysis
my_df.tail()


# In[25]:


import math

def plot_bivariate_bar(dataset, hue, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    dataset = dataset.select_dtypes(include=[np.object])
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataset.shape[1]) / cols)
    for i, column in enumerate(dataset.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if dataset.dtypes[column] == np.object:
            g = sns.countplot(y=column, hue=hue, data=dataset)
            substrings = [s.get_text()[:10] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            
bivariate_df = my_df.loc[:, ['workclass', 'education', 
           'marital-status', 'occupation', 
           'relationship', 'race', 'sex','predclass']]  

plot_bivariate_bar(bivariate_df, hue='predclass', cols=2, width=20, height=15, hspace=0.4, wspace=0.5)


# The dataset was created in 1996, a large number of jobs fall into the category of mannual labor, e.g., Handlers cleaners, craft repairers, etc. Executive managerial role and some one with a professional speciality has a high level payment

# In[26]:


#Building Machine Learning Models

from sklearn.cluster import KMeans
from matplotlib import cm
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score


from sklearn.model_selection import GridSearchCV


#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
import xgboost as xgb
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
#train,test=train_test_split(train_df,test_size=0.2,random_state=0,stratify=abalone_data['Sex'])


# In[27]:


#Feature Encoding
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split #training and testing data split


# In[28]:


my_df = my_df.apply(LabelEncoder().fit_transform)
my_df.head()


# In[29]:


#Train-test split
drop_elements = ['education', 'native-country', 'predclass', 'age_bin', 'age-hours_bin','hours-per-week_bin']
y = my_df["predclass"]
X = my_df.drop(drop_elements, axis=1)
X.head()


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[31]:


#Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
# y_pred = gaussian.predict(X_test)
score_gaussian = gaussian.score(X_test,y_test)
print('The accuracy of Gaussian Naive Bayes is', score_gaussian)


# In[32]:


#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
#y_pred = logreg.predict(X_test)
score_logreg = logreg.score(X_test,y_test)
print('The accuracy of the Logistic Regression is', score_logreg)


# In[33]:


# Random Forest Classifier
randomforest = RandomForestClassifier()
randomforest.fit(X_train, y_train)
#y_pred = randomforest.predict(X_test)
score_randomforest = randomforest.score(X_test,y_test)
print('The accuracy of the Random Forest Model is', score_randomforest)


# In[34]:


# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
#y_pred = knn.predict(X_test)
score_knn = knn.score(X_test,y_test)
print('The accuracy of the KNN Model is',score_knn)


# In[35]:


### cross validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Naive Bayes','Logistic Regression','Decision Tree','KNN','Random Forest']
models=[GaussianNB(),LogisticRegression(),DecisionTreeClassifier(),
        KNeighborsClassifier(n_neighbors=9),RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    cv_result = cross_val_score(model,X,y, cv = kfold,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
models_dataframe=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
models_dataframe


# Random Forest is the most accurate model by now.

# END OF PROJECT

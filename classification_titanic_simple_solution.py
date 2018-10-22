
# coding: utf-8

# ## Classification
# ## Example: Predict survival on Titanic

# In[5]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#import matplotlib.pyplot as plt
#%matplotlib inline


# ## Working with data

# In[6]:


train = pd.read_csv(r'C:\Users\lizac\Downloads\homework-master (8)\homework-master\lecture_2\data\train.csv')
test = pd.read_csv(r'C:\Users\lizac\Downloads\homework-master (8)\homework-master\lecture_2\data\test.csv')


# In[7]:


train.info()


# In[8]:


test.info()


# In[9]:


train.head(3)


# ### We'll need test_pas_id for submission dataframe

# In[10]:


test_pas_id = test['PassengerId']


# ### Make a list from train and test

# In[11]:


full_data=[train, test]


# ### Impute missing values

# #### Embarked

# In[12]:


train[train['Embarked'].isnull()]


# In[13]:


train[(train['Fare']>79) & (train['Fare']<81) & (train['Pclass']==1)].groupby('Embarked').size()


# In[14]:


for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('C')


# #### Fare

# In[15]:


for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())


# ####  Age

# In[16]:


# We have plenty of missing values in this feature. 
# Generate random numbers between (mean - std) and (mean + std). 


# In[17]:


np.random.seed(0)
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list 


# ### Data preprocessing

# In[18]:


np.unique(full_data[0]['Sex'], return_counts = True)


# In[19]:


for dataset in full_data:
  # Mapping Sex
  dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} )


# In[20]:


for dataset in full_data:    
    #Discrete variables
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 


# In[21]:


np.unique(full_data[0]['Embarked'], return_counts = True)


# In[22]:


for dataset in full_data:
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)


# In[23]:


for dataset in full_data:
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


# In[24]:


for dataset in full_data:
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)


# In[25]:


for dataset in full_data:
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)


# In[26]:


#title
stat_min=10
data=pd.concat([train,test])
title_names = (data['Title'].value_counts() < stat_min)
title_names=pd.DataFrame(title_names).reset_index()
title_names=title_names[title_names['Title']==False]['index'].values
for dataset in full_data:
 
    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if x not in title_names else x)


# In[27]:


for dataset in full_data:
    # Mapping Title
    dataset['Title'] = dataset['Title'].map( {'Master': 0, 'Misc': 1, 'Miss': 2,'Mr':3,'Mrs':4} ).astype(int)


# In[28]:


np.unique(full_data[0]['AgeBin'], return_counts = True)


# In[29]:


#encoding interval values
label = LabelEncoder()
for dataset in full_data:
    dataset['FareBin'] = label.fit_transform(dataset['FareBin'])
    dataset['AgeBin'] = label.fit_transform(dataset['AgeBin'])


# In[30]:


np.unique(full_data[0]['Title'], return_counts = True)


# In[31]:


factors_train = train[['Pclass','Embarked','Title','FareBin','AgeBin']]
factors_test = test[['Pclass','Embarked','Title','FareBin','AgeBin']]


# In[32]:



enc = preprocessing.OneHotEncoder(dtype = 'int32')
enc.fit(factors_train)


# In[33]:


tfactors_train = enc.transform(factors_train).toarray()
tfactors_train


# In[34]:


tfactors_test = enc.transform(factors_test).toarray()
tfactors_test[:5,]


# In[35]:


train_pclass_emb = pd.DataFrame(tfactors_train, columns = ('Pclass_1', 'Pclass_2', 'Pclass_3', 'Emb_C', 'Emb_Q', 'Emb_S','Master', 'Misc', 'Miss','Mr','Mrs',
                                                          'Fare_1','Fare_2','Fare_2','Fare_4','Age_1','Age_2','Age_3','Age_4','Age_5'))


# In[36]:


train_pclass_emb.head()


# In[37]:


test_pclass_emb = pd.DataFrame(tfactors_test, columns =('Pclass_1', 'Pclass_2', 'Pclass_3', 'Emb_C', 'Emb_Q', 'Emb_S','Master', 'Misc', 'Miss','Mr','Mrs',
                                                          'Fare_1','Fare_2','Fare_2','Fare_4','Age_1','Age_2','Age_3','Age_4','Age_5'))


# In[38]:


test_pclass_emb.head()


# ### Feature Selection

# In[39]:


drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Pclass', 'Embarked','Age','Fare','Title']
train = train.drop(drop_elements, axis = 1)
test = test.drop(drop_elements, axis = 1)


# In[40]:


train = pd.concat([train,train_pclass_emb], axis=1)
test = pd.concat([test,test_pclass_emb], axis=1)


# In[41]:


train.head(5)


# In[42]:


trainv = train.values


# In[43]:


trainv.shape


# In[44]:


X = trainv[0:, 1:]
y = trainv[0:, 0]


# In[45]:


# Standardize features by removing the mean and scaling to unit variance
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# ### Data prepared to predict for submission

# In[46]:


Xnew = test.values
Xnew.shape


# ## Modeling

# In[47]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, classification_report


# In[48]:


# Split to train and test
# 75% and 25% by default
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=2)
print(Xtrain.shape, Xtest.shape)


# In[49]:


np.unique(ytrain, return_counts = True)


# In[50]:


np.unique(ytest, return_counts = True)


# In[51]:


# http://scikit-learn.org


# ## LogisticRegression

# In[52]:


# http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


# In[53]:


# Model
model_lgr = LogisticRegression(random_state = 1)
print(model_lgr)


# In[54]:


# C - Inverse of regularization strength; must be a positive float.
# Smaller values specify stronger regularization.


# ### Fit the model

# In[55]:


model_lgr.fit(Xtrain, ytrain)


# ### Model fit parameters

# In[56]:


model_lgr.coef_


# In[57]:


model_lgr.coef_.reshape(11,)


# In[ ]:


params = pd.Series(model_lgr.coef_.reshape(11,), index=train.columns[1:])
params


# In[ ]:


model_lgr.intercept_


# ### Model validation

# In[ ]:


# Predict on train

ypred_train = model_lgr.predict(Xtrain)
ypred_train_proba = model_lgr.predict_proba(Xtrain)


# In[ ]:


# Predict on test

ypred = model_lgr.predict(Xtest)
print(ypred[:10])

ypred_proba = model_lgr.predict_proba(Xtest)
print(ypred_proba[:5,:])

# ypred_proba[:,0] - probability for class zero (not survived), 
# ypred_proba[:,1] - probability for class one - survived


# #### Metrics: accuracy, confusion matrix, classification report, AUC
# #### http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics

# In[ ]:


# We can check our classification accuracy by comparing 
# the true values of the test set to the predictions:


# In[ ]:


# Accuracy on train
accuracy_score(ytrain, ypred_train)


# In[ ]:


# Accuracy on test
accuracy_score(ytest, ypred)


# In[ ]:


# Score for classification models is accuracy
model_lgr.score(Xtest, ytest)


# In[ ]:


# Accuracy doesn't tell us where we've gone wrong: 
# one nice way to do this is to use the confusion matrix


# In[ ]:


print(confusion_matrix(ytest, ypred))


# In[ ]:


target_names = ['not survived', 'survived']
print(classification_report(ytest, ypred, target_names=target_names))


# In[ ]:


# AUC
# y_scores -  probability estimates of the positive class

print("AUC on traint =", roc_auc_score(ytrain, ypred_train_proba[:, 1]))
print("AUC on test =", roc_auc_score(ytest, ypred_proba[:, 1]))


# #### <span style="color:red">Submission to kaggle a prediction for Xnew with model_lgr was given a score (accuracy) 0.7799</span>

# ### K-fold Cross-Validation

# In[ ]:


# http://scikit-learn.org/stable/modules/cross_validation.html

from sklearn.model_selection import cross_val_score


# In[ ]:


lgr = LogisticRegression(random_state = 1)

# Split to train and test: 80% and 20% 
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=2, test_size=0.2)

# Train, valid, test
scores = cross_val_score(lgr, Xtrain, ytrain, cv=5)
scores


# In[ ]:


print("Mean cv accuracy : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:


lgr.fit(Xtrain,ytrain)
print("Accuracy on train",lgr.score(Xtrain,ytrain))
print("Accuracy on test", lgr.score(Xtest, ytest))


# ### Hyperparameters Grid Search

# In[58]:


# http://scikit-learn.org/stable/modules/grid_search.html#grid-search

# GridSearchCV exhaustively considers all parameter combinations

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [.001, .01, 1, 10],
              'penalty': ['l1', 'l2']}
lgr = LogisticRegression(random_state=1)
grid = GridSearchCV(lgr, param_grid, cv=5)


# In[59]:


grid.fit(Xtrain, ytrain)


# In[60]:


# Mean cross-validated score of the best_estimator
grid.best_score_


# In[61]:


grid.best_params_


# In[62]:


model = grid.best_estimator_


# In[63]:


print(model)


# In[ ]:


model.score(Xtest,ytest)


# In[ ]:


model.score(Xtrain,ytrain)


# ### Save / load a model

# In[ ]:


from sklearn.externals import joblib
joblib.dump(model, 'model.pkl') 


# In[ ]:


model1 = joblib.load('model.pkl') 


# In[ ]:


print(model1)


# In[ ]:


ypred = model1.predict(Xtest)
ypred[:10]


# ### RandomForestClassifier

# In[64]:


from sklearn.ensemble import RandomForestClassifier


# In[65]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=2)
print(Xtrain.shape, Xtest.shape)


# In[66]:


rfc = RandomForestClassifier(random_state = 2)
rfc.fit(Xtrain, ytrain)


# In[67]:


rfc.score(Xtrain, ytrain)


# In[68]:


rfc.score(Xtest, ytest)


# In[69]:


from sklearn.tree import DecisionTreeClassifier


# In[70]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(Xtrain, ytrain)
Y_pred = decision_tree.predict(Xtest)
acc_decision_tree = round(decision_tree.score(Xtrain, ytrain) * 100, 2)
acc_decision_tree


# In[71]:


decision_tree.score(Xtest, ytest)


# In[ ]:


featires_imp = pd.Series(rfc.feature_importances_, index=train.columns[1:])
featires_imp


# In[ ]:


ypred_test = rfc.predict(Xtest)


# In[ ]:


target_names = ['not survived', 'survived']
print(classification_report(ytest, ypred_test, target_names=target_names))


# #### Hyperparameters Grid Search

# In[72]:


from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [5, 10, 15, 20,200],
             'max_depth': [2, 3, 4, 5, 6, 7, 8]}
grid = GridSearchCV(RandomForestClassifier(random_state = 2), param_grid, cv=3)


# In[73]:


grid.fit(Xtrain, ytrain)


# In[74]:


# Mean cross-validated score of the best_estimator
grid.best_score_


# In[75]:


grid.best_params_


# In[76]:


best_rfc = grid.best_estimator_


# In[77]:


best_rfc.score(Xtest,ytest)


# In[78]:


best_rfc.score(Xtrain,ytrain)


# ### GradientBoostingClassifier

# In[79]:


from sklearn.ensemble import GradientBoostingClassifier


# In[80]:


gbc = GradientBoostingClassifier(random_state = 2)


# In[81]:


gbc.fit(Xtrain,ytrain)


# In[82]:


gbc.score(Xtrain,ytrain)


# In[83]:


gbc.score(Xtest,ytest)


# In[ ]:


# learning_rate, n_estimators, max_depth


# #### Hyperparameters Grid Search

# In[91]:


from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [10, 20, 30, 40, 50, 75, 100],
             'max_depth': [2, 3, 4, 5, 6]}
grid = GridSearchCV(GradientBoostingClassifier(random_state = 2), param_grid, cv=4)


# In[92]:


grid.fit(Xtrain, ytrain)


# In[93]:


# Mean cross-validated score of the best_estimator
grid.best_score_


# In[94]:


grid.best_params_


# In[95]:


best_gbc = grid.best_estimator_


# In[96]:


best_gbc.score(Xtest,ytest)


# In[97]:


best_gbc.score(Xtrain,ytrain)


# ### XGBoost
# #### http://xgboost.readthedocs.io/en/latest/python/python_intro.html

# In[98]:


grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]


# In[102]:


import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
'learning_rate': grid_learn, #default: .3
            'max_depth': [1,2,4,6,8,10], #default 2
            'n_estimators': grid_n_estimator, 
            'seed': grid_seed  
}

# Instantiate the regressor: gbm
gbm = XGBClassifier(n_estimators=10)

# Perform random search: grid_mse
xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 
                                    estimator = gbm, scoring = "accuracy", 
                                    verbose = 4, n_iter = 100, cv = 4)


# Fit randomized_mse to the data
xgb_random.fit(Xtrain, ytrain)


# In[103]:


xgb_random.score(Xtest,ytest)


# ### Submission

# In[ ]:


ypred_Xnew = model_lgr.predict(Xnew).astype(int)


# In[ ]:


# Generate Submission File 

# Use model with the best accuracy on test to predict on Xnew (ypred_Xnew should be int)

# Example: ypred_Xnew = model_lgr.predict(Xnew).astype(int)

submission = pd.DataFrame({ 'PassengerId': test_pas_id,
                            'Survived': ypred_Xnew })
submission.to_csv("submission.csv", index=False)


# 1) Register on https://www.kaggle.com
# 2) Go to https://www.kaggle.com/c/titanic/submit
# 3) Submit your csv file and get the score (accuracy)

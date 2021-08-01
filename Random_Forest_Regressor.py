# -*- coding: utf-8 -*-
"""Random forest regressor.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/SatyamRaj1/Machine-Learning-using-Scikit-Toolkit/blob/main/Random_forest_regressor.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl

"""Assigning and Declaring"""

import seaborn as seb
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets 
from sklearn.model_selection import train_test_split as tts
import seaborn
from sklearn import metrics

iris=datasets.load_iris()

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
#print(df)
print("columns",df.columns)
#filling none values by mean
df['sepal length (cm)']=df['sepal length (cm)'].fillna(df['sepal length (cm)'].mean())
df['sepal width (cm)']=df['sepal width (cm)'].fillna(df['sepal width (cm)'].mean())
df['petal length (cm)']=df['petal length (cm)'].fillna(df['petal length (cm)'].mean())
df['petal width (cm)']=df['petal width (cm)'].fillna(df['petal width (cm)'].mean())
df['target']=df['target'].fillna(df['target'].mean())
x=df.iloc[:,0:4]
y=df.iloc[:,4]
'''x=iris.data
y=iris.target'''
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2,random_state=4)
print("X",'\n',x.head(),'\n')
print("y",'\n',y.head(),'\n')
print(x.shape,y.shape)
classes=['Iris Setosa','Iris Versicolour','Iris Virginica']
print(df.info())

"""EDA.  
SEABORN CORRELATION - used for feature selection. M1 for feature selection
"""

seaborn.heatmap(df.corr(),annot= True,fmt='.1g',cmap='RdYlGn')

"""from this we can observe that target is least related to sepal width (with 0.4, -ve show that incresing it will decrese target so are reversaly related by 0.4) and most to petal width  
**Note 0.4 is a significant number so acc may decrease**

Scattering Graph b/w sepal width (cm) and target
"""

mpl.scatter(df['sepal width (cm)'],y)
mpl.xlabel('sepal width (cm)')
mpl.ylabel('target')

"""b/w petal width (cm) and target"""

mpl.scatter(df['petal width (cm)'],y)
mpl.xlabel('petal width (cm)')
mpl.ylabel('target')

"""SelectKBest feature engineering method using chi - squared statical test"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestf=SelectKBest(score_func=chi2, k=3)  #selecting best 3 using chi2
fit=bestf.fit(x,y)
a=pd.DataFrame({"score":fit.scores_,
                "columns":x.columns})
print(a)

"""Model- Random Forest Regressor"""

model=RandomForestRegressor()
model.fit(x_train,y_train)
print(model)
pred=model.predict(x_test)  #predictions
#print('predictions:',pred)
#print(np. array(y_test))  #to compare with predictions
#print('accuracy:',acc)
mse=metrics.mean_squared_error(y_test,pred)
print("RMSE: ",np.sqrt(mse))
sc=model.score(x_test,y_test)
print('R squared score',sc)
a=y_test.values
fra=pd.DataFrame(data=[pred,a]).T
fra.columns=['predictions','original']
print(fra)

"""Feature Engineering using feature inportance"""

imp=model.feature_importances_
df_imp=pd.DataFrame(imp,index=x.columns,columns=['Importance'])
print(df_imp)

"""**Note** there is an error coming in feature importance (as it is deviating from heatmap and SelectKbest and also it's value is changing if we change random state while dividing data during Test Train Split (type any other no. in random state of test train split and find out other two are same but it is not and also every time gives different when running model).   
**Till now issue didn't resolved**
"""

df_imp.plot(kind="bar")

"""removing unimp features"""

'''
necc=df_imp[((df_imp["Importance"]>0.0085)  | (df_imp["Importance"]<0.008))].index
df_necc=x[necc].copy()          ##would be using if importance feature going good
#print(df_necc)
'''
df_necc=x[['sepal length (cm)','petal length (cm)','petal width (cm)']]
x1_train,x1_test,y1_train,y1_test=tts(df_necc,y,test_size=0.2,random_state=2)
model=RandomForestRegressor()
model.fit(x1_train,y1_train)
print("score after feature eng",model.score(x1_test,y1_test))
pred1=model.predict(x1_test)
print(pred1)
print(np.array(y1_test))

"""Hyperprameter Tuning

declaring all individual hyperparameters.  can add or delete parameters
"""

from sklearn.model_selection import GridSearchCV

#number of trees in random forest
n_est=[x for x in np.arange(start=10,stop=500,step=10)]
#print(ar)
#number of features to be consider at every split
max_f=['auto','sqrt']
#maximum number oflevels
max_d=[2,4]
#min no. of sample required to split a node
min_ss=[2,5]
#min no. of sample required at each leaf node
min_sl=[0,1,2]
#method of selecting samplesfor each training tree
bootstrap=[True,False]

"""putting them in a grid using dictionary"""

rand_grid={'n_estimators':n_est,
           "max_features": max_f,
           'max_depth':max_d,
           #'min_sample_split':min_ss,       ##didn't undersood these both's parameters
           #'min_sample_leaf':min_sl,
           'bootstrap':bootstrap
           }
print(rand_grid)

"""now we will build a model and run all the combination from _est to find best  
gridsearchcv - to run all combinations. can also use RandomizedSearchCV
"""

model=RandomForestRegressor()
grd=GridSearchCV(estimator=model,param_grid=rand_grid,cv=3,verbose=2,n_jobs=3)
grd.fit(x1_train,y1_train)
grd.best_params_

print("R square after hypertunning. ",grd.score(x1_test,y1_test))

model
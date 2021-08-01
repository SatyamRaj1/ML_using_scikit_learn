from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as acs
import pandas as pd
from sklearn.preprocessing import scale
bc=datasets.load_iris()
#print(bc)
x=scale(bc.data)  #big difference between numbers so scaling  #making somewhat data easy
y=bc.target
print(x)
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2)
model=KMeans(n_clusters=3,random_state=0)  #to fix randomization use random  cluster==label/group of data
model.fit(x_train)
pred=model.predict(x_test)
labels=model.labels_
print('labels:',labels)
print('predictions:',pred)
acc=acs(y_test,pred)
print('accuracy:',acc)
print(y_test)
print(pd.crosstab(y_train,labels))  #if in output order of 0,1 is opposite in both then flip 0,1
#make 1--0, 0--1

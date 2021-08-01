#from sklearn.externals import joblib   #to not to train model again and again

#saving
filename='model.sav'  #filename
joblib.dump(clf,filename)   #here clf is a model
#opening
clf=joblib.load(filename)

#traintestsplit
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn import metrics,neighbors
from sklearn.preprocessing import LabelEncoder 
#to use iris dataset of sklearn
iris=datasets.load_iris()
#splitting in features and labels
x=iris.data #features
y=iris.target  #labels
print(x.shape,y.shape)  #dimensions  y=150*1

#For ML Project we need to use some data to train and then some data to test
#for ex we have data of 10 differebt students and we are putting their hours of study vs grade in model
#so we use data of 8 students to trai and 2 to test (predict)
#more data in trai make good model and more data in train part takes proper test (level of accuracy)
#training data is always more than testing data
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2)  #be in order  #test size for 20% of data in train
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)   #y are labels

#knn classifier
#to check a point(P) it calculate 15(k) nearest(default or specified) and check majority are in which region
#more k will result in better model (using odd number is better)
#all points have same amount of role if weight is uniform(by default)
#if weights=distance then more importance to closer

#using random dataset
le=LabelEncoder() 
data=pd.read_csv('car.data',header=None,    #to name index col use index_col='ID'
      names=['buying','maint','doors','persons','lug_boots','safety','class'])
#to put heading M1 put front line in that file as headings
#M2
print(data.head())
#we are using only 3 attributes as labels in our model
x=data[['buying','maint','safety']].values  #attributes or features   #values for LabelEncoder
y=data[['class']]  #labels
#print(x,y)
#here problem is we are having strings like vhigh,high,med,low..
#so convert into numbers
#converting the data  #M1
for i in range(len(x[0])):  #x[0] for no. of columns as in first row iterate column wise
    x[:, i]=le.fit_transform(x[:, i])     #x[:,i] will take all rows and one by one columns
#    it will transform each row
#print(x,y)

#Converting M2 of Y(label)
con={'unacc':0,
     'acc':1,
     'good':2,
     'vgood':3}
y['class']=y['class'].map(con)
y=np.array(y)
print(x,y)

#creating knn model
knn=neighbors.KNeighborsClassifier(n_neighbors=25,weights='uniform')
#we need to separate data into training and testing data to maintain accuracy
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2)
knn.fit(x_train,y_train)    #training
pred=knn.predict(x_test)   #testing / predicting
acc=metrics.accuracy_score(y_test,pred)    #to check accuracy
print('predictions :',pred)
print('accuracy :',acc)
print('actual value',y[1727])
print('predicted value',knn.predict(x)[1727])

#SVM Support vestor machine
'''effective high dimension spaces
many kernel functin
#classification and regresion
use hyper plane to separate labels snd caslculate distance
for 2dimensions hyper plane=line,3d- a 2dplane
kernals--functions to increase dimensions
'''
from sklearn import svm  #importing model 
iris=datasets.load_iris()
#splitting in features and labels
x=iris.data #features
y=iris.target  #labels
print(x.shape,y.shape)  #dimensions  y=150*1
classes=['Iris Setosa','Iris Versicolour','Iris Virginica']
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2)
model=svm.SVC()
model.fit(x_train,y_train)
print(model)
pred=model.predict(x_test)  #predictions
acc=metrics.accuracy_score(y_test,pred)  #accuracy
print('predictions:',pred)
print(y_test)  #to compare with predictions
print('accuracy:',acc)
for i in range(len(pred)):
    print(classes[pred[i]])  #to print name



     









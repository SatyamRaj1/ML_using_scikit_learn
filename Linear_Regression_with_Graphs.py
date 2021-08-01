from sklearn import datasets as dt
from sklearn import linear_model as lnd
from sklearn.model_selection import train_test_split as tts
from matplotlib import pyplot as pt
boston=dt.load_boston()
#features and labels
x=boston.data
y=boston.target
#print(x)
#print(y)
print(x.shape)
print(y.shape)


#algorithm
#linear regression model
lreg=lnd.LinearRegression()
pt.scatter(x.T[5],y)   # as y is 506*1 so need 506 feature of same type (here total 13 type)
#so one type and then transpose else it will be 1*506
pt.show()  #to view graph

x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2)

#training
model=lreg.fit(x_train,y_train)
pred=model.predict(x_test)
rscore=lreg.score(x,y)
print('predictions:',pred)
print('r^2 value:',rscore)
coeff=lreg.coef_     #like slope of linear regression model
print('coefficient',coeff)  
intercept=lreg.intercept_
print('intercept:',intercept)

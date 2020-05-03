import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13]
y=dataset.iloc[:,-1]

geography=pd.get_dummies(x['Geography'],drop_first=True)
gender=pd.get_dummies(x['Gender'],drop_first=True)

x=pd.concat([x,gender,geography],axis=1)

x=x.drop(['Gender','Geography'],axis='columns')

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


classifier=Sequential()

classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=11))

classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))

classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model=classifier.fit(x_train,y_train,validation_split=0.33,batch_size=10,nb_epoch=100)

print(model.history.keys())

plt.plot(model.history(['loss']))
plt.plot(model.history(['val_loss']))
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','test'],loc='upper left')
plt.show()
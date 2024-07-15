import numpy as n
import pandas as p
import seaborn as sns
d=p.read_csv("Churn_Modelling.csv")
d.drop(['RowNumber','CustomerId','Surname'], axis=1,inplace=True)
geo = p.get_dummies(d['Geography'], drop_first=True)
gen = p.get_dummies(d['Gender'], drop_first=True)
d=p.concat([d,geo,gen], axis=1)
d.drop(['Geography','Gender'], axis=1, inplace=True)
x=d.drop('Exited',axis=1)
y=d['Exited']
from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts = train_test_split(x,y,test_size=0.2,random_state=0)
print("X's train size: {},X's test size: {}".format(xtr.shape,xts.shape))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtr = sc.fit_transform(xtr)
xts = sc.transform(xts)
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu', input_dim=11))
classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_history = classifier.fit(xtr, ytr, batch_size=10, validation_split=0.33, epochs=100)
yprd = classifier.predict(xts)
yprd = (yprd > 0.5)




def predict_exit(sample_value):

  # Convert list to numpy array
  sample_value = n.array(sample_value)

  # Reshape because sample_value contains only 1 record
  sample_value = sample_value.reshape(1, -1)

  # Feature Scaling
  sample_value = sc.transform(sample_value)

  return classifier.predict(sample_value)

sample_value = [738, 62, 10, 83008.31, 1, 1, 1, 42766.03, 1, 0, 1]
if predict_exit(sample_value)>0.5:
  print('Prediction: High change of exit!')
else:
  print('Prediction: Low change of exit.')

sample_value = [805, 45, 9, 116585.97, 1, 1, 0, 189428.75, 1, 0, 0]
if predict_exit(sample_value)>0.5:
  print('Prediction: High change of exit!')
else:
  print('Prediction: Low change of exit.')


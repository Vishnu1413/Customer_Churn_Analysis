import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
data=pd.read_csv("naive.csv")
print("Before Transformation")
print(data)
number = LabelEncoder()
data['age'] = number.fit_transform(data['age'])
data['income'] = number.fit_transform(data['income'])
data['student'] = number.fit_transform(data['student'])
data['credit'] = number.fit_transform(data['credit'])
data['buy'] = number.fit_transform(data['buy'])
print("After Transformatiobn")
print(data)
feature = ['age','income','student','credit']
x = data[feature]
y = data.buy
xtr, xte,ytr,yte = train_test_split(x,y,test_size=0.3,random_state =0)
model = GaussianNB()
model.fit(xtr,ytr)
pred = model.predict(xte)
acc = accuracy_score(yte,pred)
cm=confusion_matrix(y,model.predict(x))
print(classification_report(y,model.predict(x)))
fig,ax=plt.subplots(figsize=(8,8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0,1),ticklabels=('Predicted 0s','Predicted 1s'))
ax.yaxis.set(ticks=(0,1),ticklabels=('Actual 0s','Actual 1s'))
for i in range(2):
    for j in range(2):
        ax.text(j,i,cm[i,j],ha='center',va='center',color='red')
plt.show()


import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix

# import neptune
# run = neptune.init_run(
#     project="vishnumass/Churn-Analysis",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3NWM5MDg1Zi0yMjMwLTQ4NWYtOGYyMC00NDQ3NTYyMWM3OTEifQ==",
# ) 

data = pd.read_csv("Churn.csv")

# Define the features and target variable
features = ['Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
x = data[features]
y = data['Exited']
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

# # Upload confusion matrix visualization to Neptune
# run["Confusion-Matrix_Navie"].upload(neptune.types.File.as_image(fig))

# # Stop Neptune run
# run.stop()

#packages for logistics regression 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report,confusion_matrix 
#input file  
data=pd.read_csv('Logisticsdataset.csv') 
x= data.iloc[:, :-1].values  
y= data.iloc[:, 1].values
 
#model creation using liblinear 
model=LogisticRegression(solver='liblinear',random_state=0) 
model.fit(x,y) 
# confusion matrix 
cm=confusion_matrix(y,model.predict(x)) 
#classification report  
print(classification_report(y,model.predict(x))) 
#visulaization of heatmap 
fig,ax=plt.subplots(figsize=(8,8)) 
ax.imshow(cm) 
ax.grid(False) 
ax.xaxis.set(ticks=(0,1),ticklabels=('Predicted 0s','Predicted 1s')) 
ax.yaxis.set(ticks=(0,1),ticklabels=('Actual 0s','Actual 1s')) 
ax.set_ylim(1.5,-0.5) 
for i in range(2): 

    for j in range(2): 

        ax.text(j,i,cm[i,j],ha='center',va='center',color='red') 
plt.show() 


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

# import neptune
# run = neptune.init_run(
#     project="vishnumass/Churn-Analysis",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3NWM5MDg1Zi0yMjMwLTQ4NWYtOGYyMC00NDQ3NTYyMWM3OTEifQ==",
# )  

# Load your dataset
data = pd.read_csv("Churn.csv")

# Define the features and target variable
features = ['Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
X = data[features]
y = data['Exited']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

# Create a Random Forest classifier with 'max_features' set to 'None'
rf = RandomForestClassifier(n_estimators=3, max_depth=2, max_features=None, bootstrap=True)
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Compute and print confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)
print("Classification Report")

cr=classification_report(y_test, y_pred)
print(cr)

# Calculate specificity and sensitivity
TN, FP, FN, TP = cm.ravel()
specificity = TN / (TN + FP)
sensitivity = TP / (TP + FN)

print("Specificity: %.2f" %round(specificity,3))
print("Sensitivity: %.2f" %round(sensitivity,3))



# run["Confusion"].append(cm)
# run["Report"].append(cr)
# run["Specificity"].append(specificity)
# run["sensitivity"].append(sensitivity)


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

figs= plt.figure(figsize=(12, 12), facecolor='w')
tree.plot_tree(rf.estimators_[0], feature_names=features, class_names=["0", "1"], filled=True, fontsize=9)
plt.show()

# run["Confusion-Matrix_RandomForest"].upload(neptune.types.File.as_image(fig))
# run["Tree_RandomForest"].upload(neptune.types.File.as_image(figs))

# run.stop()




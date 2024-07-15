import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# Load the churn dataset
data = pd.read_csv("Churn.csv")

# Define the features and target variable
features = ['Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
X = data[features]
y = data['Exited']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Define class names
class_names = ['Not Exited', 'Exited']

# Plot and visualize the decision tree with class_names
plt.figure(figsize=(12, 6))
plot_tree(clf, filled=True, class_names=class_names)
plt.title("Decision Tree Visualization")
plt.show()

# Make predictions on the test data
y_pred = clf.predict(X_test)

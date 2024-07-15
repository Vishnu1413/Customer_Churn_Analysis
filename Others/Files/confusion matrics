import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# generate some sample data
X, y = make_classification(n_samples=1000,
n_features=10,
n_informative=6,
n_redundant = 2,
n_repeated = 2,
n_classes = 6,
n_clusters_per_class=1,
random_state = 42
)


# split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)


# initialize and train a classifier
clf = SVC(random_state=0)
clf.fit(X_train, y_train)


# get the model’s prediction for the test set
predictions = clf.predict(X_test)


# using the model’s prediction and the true value,
# create a confusion matrix
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)


# use the built-in visualization function to generate a plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()

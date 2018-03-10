from sklearn import tree
from sklearn import preprocessing

file  = open("training.txt")
data = []
target = []

print "segregating data and splitting the test and train data..."
for row in open("training.txt"):
    rowdata = row.split()
    labelEncoder = preprocessing.LabelEncoder() #using labelEncoder because the DecisionTreeClassifier doesnot accept strings. because we have some strings in the data, we need to encode it.
    data.append(labelEncoder.fit_transform(rowdata[:17])) # here i am separating the data from target because thats how the classifier takes inputs
    target.append(rowdata[17:])  # separating target

from sklearn.cross_validation import train_test_split
train_data, test_data, train_label, test_label = train_test_split(data, target, test_size = .33)
# the above step will divide data set as 2/3 to training and 1/3 to testing so the test_size = .33 which is 1/3
clf = tree.DecisionTreeClassifier()
print "training the model...."
clf.fit(train_data, train_label) # training model happens here
predictions = clf.predict(test_data) # prediction is happening here

from sklearn.metrics import accuracy_score
print " Accuracy : %s " % (accuracy_score(test_label, predictions)) # accuracy_score is displayed here.

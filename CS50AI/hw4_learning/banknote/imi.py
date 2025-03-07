import csv
import random

from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# model = KNeighborsClassifier(n_neighbors=1)
# model = Perceptron()
# model = svm.SVC()
model = GaussianNB()

# Read data in from file

with open("banknotes.csv") as f:   ## new to zzt
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:  ## new to zzt
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": "Authentic" if row[4] == "0" else "Counterfeit"
        })

evidence = [dict["evidence"] for dict in data]
labels = [dict["label"] for dict in data]

# Separate data into training and testing groups
 ## new to zzt
X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.4
)

# Fit model
model.fit(X_training, y_training)

# Make predictions on the testing set
predictions = model.predict(X_testing)

# Compute how well we performed
correct = (y_testing == predictions).sum()     ## new to zzt
incorrect = (y_testing != predictions).sum()   
total = len(predictions)
accuracy = correct / total

print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {accuracy:.2%}")   ## new to zzt

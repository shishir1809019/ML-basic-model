from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

digits = load_digits()
print(digits.data.size)

# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()

X = digits.data
Y = digits.target

# split test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# target and predict value
print(digits.target[900])
result = model.predict([digits.data[900]])
# print(result)

# test accuracy
accuracy = model.score(X_test, y_test)
# print(accuracy)

# confusion metrics
y_predicted = model.predict(X_test)
confusion = confusion_matrix(y_test, y_predicted)
print(confusion)
# plot_confusion_matrix(model, X_test, y_test)
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()
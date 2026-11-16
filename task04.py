
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

x = data.data
y = data.target

print("Number of Samples:", x.shape[0])
print("Number of Features:", x.shape[1])

classes, counts = np.unique(y, return_counts=True)

print("\nClass Counts:")
for i in range(len(classes)):
    print("Class", classes[i], ":", counts[i])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=1, stratify=y
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression(max_iter=5000)
model.fit(x_train, y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)

print("\nTest Accuracy:", accuracy)

probabilities = model.predict_proba(x_test)[:, 1]

thresholds = [0.3, 0.5, 0.7]

print("\nThreshold Results:")

for threshold in thresholds:
    predicted_classes = (probabilities >= threshold).astype(int)

    total_class1 = np.sum(predicted_classes == 1)
    accuracy = accuracy_score(y_test, predicted_classes)

    print("\nThreshold:", threshold)
    print("Predicted Class 1:", total_class1)
    print("Accuracy:", accuracy)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


decision_values = model.decision_function(x_test)

sigmoid_values = sigmoid(decision_values)

probability_values = model.predict_proba(x_test)[:, 1]

print("\nSigmoid Verification:")
print("Sigmoid Values:")
print(sigmoid_values[:10])

print("\nPredict Proba Values:")
print(probability_values[:10])

print("\nAre Both Values Equal?")
print(np.allclose(sigmoid_values, probability_values))

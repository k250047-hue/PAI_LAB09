
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

hours = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0])
marks = np.array([32, 40, 41, 50, 54, 60, 62, 70, 78, 88])

def gradient_descent(x, y, lr, epochs):
    m = 0
    c = 0
    n = len(x)
    loss = []

    for i in range(epochs):
        predicted = m * x + c

        error = predicted - y

        dm = (2/n) * np.sum(error * x)
        dc = (2/n) * np.sum(error)

        m = m - lr * dm
        c = c - lr * dc

        mse = np.mean((y - (m * x + c)) ** 2)
        loss.append(mse)

        if (i + 1) % 100 == 0:
            print("Epoch:", i + 1, "MSE:", mse)

    return m, c, loss


learning_rates = [0.001, 0.01, 0.05]
epochs = 1000

results = {}

for lr in learning_rates:
    print("\nLearning Rate:", lr)

    m, c, loss = gradient_descent(hours, marks, lr, epochs)

    results[lr] = (m, c, loss)

    print("Final Slope:", m)
    print("Final Intercept:", c)

    plt.plot(range(1, epochs + 1), loss, label=f"LR = {lr}")

plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("MSE vs Epochs - Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()

model = LinearRegression()
model.fit(hours.reshape(-1, 1), marks)

print("\nScikit-learn Results")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

print("\nComparison with Scikit-learn")

for lr in learning_rates:
    m, c, loss = results[lr]

    print("\nLearning Rate:", lr)
    print("Gradient Descent Slope:", m)
    print("Scikit-learn Slope:", model.coef_[0])
    print("Gradient Descent Intercept:", c)
    print("Scikit-learn Intercept:", model.intercept_)

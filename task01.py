
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

hours = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]
marks = [32, 40, 41, 50, 54, 60, 62, 70, 78, 88]

x = np.array(hours).reshape(-1, 1)
y = np.array(marks)

model = LinearRegression()
model.fit(x, y)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

new_hours = np.array([[5.5], [8.0]])
predicted_marks = model.predict(new_hours)

print("Predicted marks for 5.5 hours:", predicted_marks[0])
print("Predicted marks for 8 hours:", predicted_marks[1])

predictions = model.predict(x)

mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

plt.scatter(hours, marks, color='blue', label='Actual Data')
plt.plot(hours, predictions, color='red', label='Regression Line')

plt.xlabel("Hours Studied")
plt.ylabel("Marks Obtained")
plt.title("Study Hours vs Marks")
plt.legend()
plt.grid(True)
plt.show()

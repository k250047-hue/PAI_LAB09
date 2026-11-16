
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

experience = [1, 2, 3, None, 5, 6, 7, 8, 9, 10, 3, 5]
education = ['BS', 'BS', 'MS', 'BS', 'MS', 'PhD', 'MS', 'PhD', 'BS', 'PhD', 'BS', 'MS']
salary_k = [45, 50, 70, 58, 85, 120, 95, 140, 90, 160, 55, 88]

df = pd.DataFrame({
    'Experience': experience,
    'Education': education,
    'Salary': salary_k
})

print("Original Dataset:")
print(df)

df['Experience'] = df['Experience'].fillna(df['Experience'].median())

df = pd.get_dummies(df, columns=['Education'], dtype=int)

print("\nDataset After Preprocessing:")
print(df)

x = df.drop('Salary', axis=1)
y = df['Salary']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation:")
print("MAE:", mae)
print("R2 Score:", r2)

new_data = pd.DataFrame({
    'Experience': [4],
    'Education_BS': [0],
    'Education_MS': [1],
    'Education_PhD': [0]
})

new_data = scaler.transform(new_data)

salary_prediction = model.predict(new_data)

print("\nPredicted Salary for MS Graduate with 4 Years Experience:")
print(salary_prediction[0], "K")

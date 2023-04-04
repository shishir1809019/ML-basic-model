import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("Module\M-30\HR_comma_sep.csv")
# print(data )
# print(data.isnull().any())
# print(data.dtypes)
# print(data.salary.unique())

clean_up_values = {
    "salary": {
        'low': 1,
        'medium': 2,
        'high': 3
    }
}

data.replace(clean_up_values, inplace=True)
# print(data)

dummies = pd.get_dummies(data.Department)
# print(dummies)
merged = pd.concat([data, dummies], axis='columns')

final_data = merged.drop(['Department', 'technical'], axis='columns')
# print(final_data)

# plt.scatter(x = final_data.salary, y = final_data.left)
# plt.show()

X = final_data.drop('left', axis = 'columns')
Y = final_data.left
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = LogisticRegression()
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
# print(accuracy)
# print(X.head())

result = model.predict([[.85,.87,6,232,5,0,0,3,0,0,0,0,1,0,0,0,0]])
# [[.85,.87,6,232,5,0,0,3,0,0,0,0,1,0,0,0,0]]
# [[0,0.38,0.53,2,157,0,0,1,0, 0, 3, 0,0,1,0,0,0]]
print(result)
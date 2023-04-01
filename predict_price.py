import pandas
from sklearn.linear_model import LinearRegression
data = pandas.read_csv('Module\M-28\iphone_price.csv')
model = LinearRegression()
model.fit(data[['version']], data[['price']])
predicted_price = model.predict([[15]])
print(predicted_price)

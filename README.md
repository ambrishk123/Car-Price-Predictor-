# Car-Price-Predictor-

import pandas as pd

car_dataset = pd.read_csv('car data.csv')
car_dataset.head()

car_dataset.shape

car_dataset.columns

car_dataset.info()

car_dataset.isna().sum()

car_dataset['Fuel_Type'].value_counts()

car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}}, inplace=True)

car_dataset.head()

car_dataset['Seller_Type'].value_counts()

car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}}, inplace=True)

car_dataset.head()

car_dataset['Transmission'].value_counts()

car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
car_dataset.head()

X = car_dataset.drop(['Car_Name', 'Selling_Price'],axis='columns')
X.head()

y = car_dataset['Selling_Price']
y.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)

from sklearn.linear_model import LinearRegression

linear_regression_model = LinearRegression()

linear_regression_model.fit(X_train, y_train)

y_predicted = linear_regression_model.predict(X_test)
y_predicted

linear_regression_model.score(X_test, y_test)

from sklearn.tree import DecisionTreeRegressor

decision_tree_regressor_model = DecisionTreeRegressor()

decision_tree_regressor_model.fit(X_train, y_train)

y_predicted = decision_tree_regressor_model.predict(X_test)
y_predicted

decision_tree_regressor_model.score(X_test, y_test)

from sklearn.ensemble import RandomForestRegressor

random_forst_regressor_model = RandomForestRegressor()

random_forst_regressor_model.fit(X_train, y_train)

y_predicted = random_forst_regressor_model.predict(X_test)
y_predicted

random_forst_regressor_model.score(X_test, y_test)

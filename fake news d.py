import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# Load dataset
df = pd.read_csv('owid-covid-data.csv')

# Select features and target
features = ['stringency_index', 'people_fully_vaccinated_per_hundred',
            'population_density', 'reproduction_rate',
            'hospital_beds_per_thousand', 'gdp_per_capita']
target = 'new_cases_smoothed_per_million'

df = df[features + [target]].dropna()
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f'MSE: {metrics.mean_squared_error(y_test, pred)}')
accuracy = model.score(X_test, y_test)
print(accuracy)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# load dataset
df= pd.read_csv('concrete.csv')

# remove outliers/noises
df=df[df['Blast Furnace Slag'] < 350]
df=df[(df['Water'] > 125) & (df['Water'] < 230)]
df=df[df['Superplasticizer'] < 25]
df=df[df['Fine Aggregate'] < 960]
df=df[df['Age'] < 150]

X = df.drop('Strength', axis=1)
Y = df['Strength']

# print(X)
# print(Y)
standard = StandardScaler()

# split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2024)
standard = standard.fit(X_train)
X_train = standard.fit_transform(X_train)
X_test = standard.fit_transform(X_test)


param_grid = {
    'n_estimators': [100, 125, 130, 135],
    'max_depth': range(5, 16),
}

# model = GridSearchCV(RandomForestRegressor(n_jobs=-1), param_grid, cv=8, verbose=3, scoring='r2')
model = RandomForestRegressor(max_depth=13, verbose=1)
model.fit(X_train, Y_train)
# print("best params:", model.best_params_)

Y_train_pred = model.predict(X_train)
print(f'R-squared score for training: {r2_score(Y_train, Y_train_pred)}')

Y_test_pred = model.predict(X_test)
print(f'R-squared score for testing: {r2_score(Y_test, Y_test_pred)}')

importance = model.feature_importances_.argsort()
plt.barh(df.columns[importance], model.feature_importances_[importance])
plt.xlabel("Feature Importance")
plt.show()

# best params: {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
# without normalisation
# R-squared score for training: 0.9831060404477343
# R-squared score for testing: 0.9321149627789226

# best params: {'max_depth': 13, 'n_estimators': 100}
# with normalisation
# R-squared score for training: 0.9809033638178694
# R-squared score for testing: 0.9021821492502472
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import warnings
warnings.filterwarnings(action='ignore')

data = pd.read_excel('Data_Train.xlsx')
data

def preprocess_inputs(df):

    df = df.copy()
    
    # Drop Route
    df = df.drop(['Route'], axis=1)
    
    # Map airlines by their importance
    airlines = list(data['Airline'].value_counts().index)
    airlines_map = {k: v for v, k in enumerate(airlines)}
    df['Airline'] = df['Airline'].map(airlines_map)

    print(airlines)

    
    # Extract Days and Months from Date of journey
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'])
    df['Day of Journey'] = df['Date_of_Journey'].apply(lambda x: x.day)
    df['Month of Journey'] = df['Date_of_Journey'].apply(lambda x: x.month)
    df = df.drop('Date_of_Journey', axis=1)
    
    # Convert departure and arrival times to minute of the day
    df['Dep_Time'] = pd.to_datetime(df['Dep_Time'])
    df['Dep_Time'] = df['Dep_Time'].apply(lambda x: (int)(x.hour/6))
    df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'])
    df['Arrival_Time'] = df['Arrival_Time'].apply(lambda x: (int)(x.hour/6))
    
    # Duration converting to minutes
    df['H'] = df['Duration'].str.extract(r'(\d*(?=h))')
    df['M'] = df['Duration'].str.extract(r'(\d*(?=m))')
    df[['H', 'M']] = df[['H', 'M']].fillna(0)
    df[['H', 'M']] = df[['H', 'M']].astype(int)
    df['Duration'] = df['H'] * 60 + df['M']
    df = df.drop(['H', 'M'], axis=1)
    
    # Convert nb of stops into integer
    df['Total_Stops'] = df['Total_Stops'].replace({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})
    
    # Features from Additional info
    df['Included Meal'] = df['Additional_Info'].apply(lambda x: 0 if x == 'In-flight meal not included' else 1)
    df['Included Baggage'] = df['Additional_Info'].apply(lambda x: 0 if x == 'No check-in baggage included' else 1)
    df = df.drop('Additional_Info', axis=1)
    
    # Source and Destination encoding (One-hot)
    df = pd.concat([df, pd.get_dummies(df['Source'], prefix='S')], axis=1)
    df = pd.concat([df, pd.get_dummies(df['Destination'], prefix='D')], axis=1)
    df = df.drop(['Source', 'Destination'], axis=1)
    
    # Fill in the single remaining missing value (Total Stops) with 0
    df = df.fillna(0)
    
    # Split X and y
    X = df.drop(['Price'], axis=1)
    y = df['Price']
    
    # Apply log function to target price
    y = np.log(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=0)
    
    # Scale X
    #scaler = StandardScaler()
    #X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    #X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_inputs(data)
print(X_train)

models = {
    "Random Forest" : RandomForestRegressor(),
    "Linear ": LinearRegression() 
}

print("-------- Training --------")
for name, model in models.items():
    model.fit(X_train, y_train)
    print(name + " trained!")
print("---------- Done ----------")

y_test = np.exp(y_test)

for name, model in models.items():
    y_pred = np.exp(model.predict(X_test))

    print(name + ": R2 Score = {:.4f}".format(r2_score(y_test, y_pred)))
    print("                   RMSE = {:.0f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
    print("                    MAE = {:.0f}".format(mean_absolute_error(y_test, y_pred)))
    print("                   MAPE = {:.2f} %\n".format(mean_absolute_percentage_error(y_test, y_pred) * 100))

# Random Forest predictions
y_pred = np.exp(models['Linear '].predict(X_test))

# Display
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, y_test, edgecolor='k')
plt.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()], c='r', lw=2)
plt.title("Actual VS Predicted Prices")
plt.xlabel("Predicted Price")
plt.ylabel("Actual Price")
plt.show()
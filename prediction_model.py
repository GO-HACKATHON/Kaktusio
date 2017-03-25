import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

print 'Step 1 of 3: import datasets...'
#import processed_data
df_train = pd.read_csv('processed_data/training_data.csv')
df_test = pd.read_csv('processed_data/test_data.csv')

#remove data that has no supply and demand data for faster processing
df_train = df_train[(df_train.DEMAND != 0) & (df_train.SUPPLY != 0)]

#create X, Y for model input
X_train = df_train.drop(['GAP_FCST'], axis = 1)
Y_train = df_train['GAP_FCST']

X_test = df_test.drop(['GAP_FCST'], axis = 1)
Y_test = df_test['GAP_FCST']

#scale data
scaler = StandardScaler()
scaler.fit(X_train)  

#transform data based on scaler
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

print 'Step 2 of 3: build random forest model...'
#create predictive model
model = RandomForestRegressor(n_jobs=4, oob_score=True, min_samples_leaf=5, n_estimators=30)
model.fit(X_train, Y_train)

#determine accuracy
Y_test_pred = model.predict(X_test)
MSE = mean_squared_error(Y_test, Y_test_pred)**0.5

print 'Step 3 of 3: save model...'
#serialize model to pickle
joblib.dump(model, 'prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print 'Model performance:'
print 'Root Mean Squared Error = ', RMSE
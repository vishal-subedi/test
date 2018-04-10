import numpy as np
import scipy as sp
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
pf = PolynomialFeatures()
ss = StandardScaler()
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.ensemble import RandomForestRegressor as RFR, GradientBoostingRegressor as GBR
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 10, kernel = 'poly', degree = 3)

#read data
df = pd.read_csv('C:/Users/visha/Anaconda3/Lib/site-packages/sklearn/datasets/data/boston_house_prices.csv', header = 1, index_col = None)

#feature & target
X = df.iloc[:,:13]
y = df.iloc[:,13]

#deleting lower correlation features
del X['CHAS']
del X['B']


#split
X_train, X_test, y_train, y_test = tts(X, y, test_size = .33, random_state = 1)

#feature scaling
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

#polynomial Features
#X_train = pf.fit_transform(X_train)
#X_test = pf.fit_transform(X_test)

#lda
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

#Linear Regression
lr = LinearRegression()

#training
lr.fit(X_train, y_train)

#testing
y_pred_lr = lr.predict(X_test)

#r_sqrare
r_2_lr = r2_score(y_test, y_pred_lr)



#Linear regression with L2 penalty
ls = Lasso(random_state = 1)

#training
ls.fit(X_train, y_train)

#testing
y_pred_ls = ls.predict(X_test)

#r_square
r_2_ls = r2_score(y_test, y_pred_ls)



#Linear regression with L1 penalty
rg = Ridge(solver = 'saga')

#training
rg.fit(X_train, y_train)

#testing
y_pred_rg = rg.predict(X_test)

#r_square
r_2_rg = r2_score(y_test, y_pred_rg)



#elastic net
en = ElasticNet(random_state = 1, alpha = 2)

#training
en.fit(X_train, y_train)

#testing
y_pred_en = en.predict(X_test)

#r_square
r_2_en = r2_score(y_test, y_pred_en)



#support vector regression
svr = SVR(kernel = 'poly', degree = 3, C = 1)

#training
svr.fit(X_train, y_train)

#testing
y_pred_svr = svr.predict(X_test)

#r_square
r_2_svr = r2_score(y_test, y_pred_svr)



#KNeighbors Regression
knr = KNR(n_neighbors = 4, weights = 'distance', p = 4)

#training
knr.fit(X_train, y_train)

#testing
y_pred_knr = knr.predict(X_test)

#r_square
r_2_knr = r2_score(y_test, y_pred_knr)



#Random Forest Regression
rfr = RFR(n_estimators = 100, max_features = 'auto', random_state = 1)

#training
rfr.fit(X_train, y_train)

#testing
y_pred_rfr = rfr.predict(X_test)

#r_square
r_2_rfr = r2_score(y_test, y_pred_rfr)

#feature importance
fet = rfr.feature_importances_



#gradient boosting
gbr = GBR(n_estimators = 200, loss = 'ls', random_state = 1)

#training
gbr.fit(X_train, y_train)

#testing
y_pred_gbr = gbr.predict(X_test)

#r_square
r_2_gbr = r2_score(y_test, y_pred_gbr)
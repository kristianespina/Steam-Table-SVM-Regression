import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

df = pd.read_csv('data/steam.csv')
test_df = pd.read_csv('data/test.csv')

# Train model
features = df[['temperature','pressure']].values
labels = df[['enthalpy_liquid','vaporization']].values
svr = SVR(kernel = 'rbf', C=1e5, gamma=0.001)
regressor = MultiOutputRegressor(svr)
regressor.fit(features, labels)

# Predict based on model
#y_rbf = regressor.predict(features)
prediction = regressor.predict(test_df[['temperature','pressure']].values)
df_y = pd.DataFrame(prediction)
predicted_enthalpy_plot = df_y.iloc[:,0].values
predicted_vaporization_plot = df_y.iloc[:,1].values


plot_X1 = test_df[['temperature']].values
plot_Y1 = test_df[['enthalpy_liquid']].values
plot_Y2 = test_df[['vaporization']].values

plt.scatter(plot_X1, plot_Y1, color='darkorange', label='Enthalpy (data)')
plt.scatter(plot_X1, plot_Y2, color='green', label='Vaporization (data)')
plt.plot(plot_X1, predicted_enthalpy_plot, color='navy', lw=2, label='Enthalpy')
plt.plot(plot_X1, predicted_vaporization_plot, color='red', lw=2, label='Vaporization')
plt.xlabel('Temperature')
plt.ylabel('Enthalpy')
plt.legend()
#plt.show()


###
### REPORT~
###
# Coefficients
# Formula = https://scikit-learn.org/stable/modules/svm.html#svr
# SUM(ai -ai*)K(xi,x)+rho
# ai-ai* = dual_coeff_
# rho = intercept_
'''
print("[Dual Coefficients]")
print(regressor.estimators_[0].dual_coef_)
print("[Support Vector]")
print(regressor.estimators_[0].support_vectors_)
print("[Intercept]")
print(regressor.estimators_[0].intercept_)
'''

plt.show()
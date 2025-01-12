import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stats





def RISK_MLE (X,mu):
    inside = X - mu
    squared_norms = np.linalg.norm(inside, axis=1)**2
    # Compute the average of the squared norms
    average_squared_norm = np.mean(squared_norms)
    return average_squared_norm

def PLUS_estimator(X):
    p = X.shape[1]
    # Calculate the norm squared for each row independently(for each p features)
    row_norms_squared = np.linalg.norm(X, axis=1)**2
    factors = 1 - (p - 2) / row_norms_squared
    # Apply the factor to each row
    X_JS = X * factors[:, np.newaxis]
    X_JS[X_JS < 0] = 0
    return X_JS

def JS_estimator(X):
    p = X.shape[1]
    # Calculate the norm squared for each row independently(for each p features)
    row_norms_squared = np.linalg.norm(X, axis=1)**2
    factors = 1 - (p - 2) / row_norms_squared
    # Apply the factor to each row
    X_JS = X * factors[:, np.newaxis]
    return X_JS
def RISK_PLUS(X,mu):
    X_JS_PLUS = PLUS_estimator(X)
    inside = X_JS_PLUS - mu
    squared_norms = np.linalg.norm(inside, axis=1)**2
    # Compute the average of the squared norms
    average_squared_norm = np.mean(squared_norms)
    return average_squared_norm
def RISK_JS0(X,x_0,mu):
    p = X.shape[1]
    X_shifted = X-x_0
    X_norm_squared_shifted = np.linalg.norm(X_shifted,axis = 1)**2
    factor = (1-(p-2)/X_norm_squared_shifted)
    X_JS_0 = X_shifted * factor[:,np.newaxis] + x_0
    inside = X_JS_0 - mu
    squared_norms = np.linalg.norm(inside, axis=1)**2
    # Compute the average of the squared norms
    average_squared_norm = np.mean(squared_norms)
    return average_squared_norm
    
def RISK_JS(X_JS,mu):
    inside = X_JS - mu
    squared_norms = np.linalg.norm(inside, axis=1)**2
    # Compute the average of the squared norms
    average_squared_norm = np.mean(squared_norms)
    return average_squared_norm
RISK_MLE_ALL = []
RISK_JS_ALL = []
RISK_JS0_ALL = []
RISK_PLUS_ALL = []
def simulation(theta,n,p):
    mu = np.zeros(p)
    mu[0] = theta
    cov = np.identity(p)
    X = np.random.multivariate_normal(mu,cov,n)
    JS_ESTIMATORS = JS_estimator(X)
    Risk_1 = RISK_MLE(X,mu)
    RISK_MLE_ALL.append(Risk_1)
    Risk_3 = RISK_JS(JS_ESTIMATORS,mu)
    RISK_JS_ALL.append(Risk_3)
    x_0 = [0] * p
    x_0[0] = theta - 5 
    x_0 = np.array(x_0)
    Risk_4 = RISK_JS0(X,x_0,mu) 
    RISK_JS0_ALL.append(Risk_4)
    Risk_5 = RISK_PLUS(X,mu)
    RISK_PLUS_ALL.append(Risk_5)
    
P = [3,10,100, 150, 200, 250, 300, 350, 400, 450, 500]
for p in P:
    simulation(theta = 10,n = 250, p =p)

plt.figure(figsize=(12, 8))
plt.plot(P, RISK_MLE_ALL, label = 'Risk MLE', color = 'black')
plt.plot(P, RISK_JS_ALL, label = 'Risk JS', color = 'red')
plt.plot(P, RISK_JS0_ALL, label = 'Risk JS0', color = 'blue')
plt.plot(P, RISK_PLUS_ALL, label = 'Risk PLUS', color = 'green')
plt.title('Risk of MLE VS Risk JS vs Risk of JS0 vs Risk Plus estimator ', fontsize=25)
plt.legend()
plt.ylabel('Risk', fontsize=25)
plt.xlabel('p', fontsize=25)
plt.show()
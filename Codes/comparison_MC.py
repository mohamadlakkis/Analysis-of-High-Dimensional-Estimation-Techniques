import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stats


def JS_estimator(X):
    p = X.shape[1]
    # Calculate the norm squared for each row independently(for each p features)
    row_norms_squared = np.linalg.norm(X, axis=1)**2
    factors = 1 - (p - 2) / row_norms_squared
    # Apply the factor to each row
    X_JS = X * factors[:, np.newaxis]
    
    return X_JS


def first_term (X,mu):
    inside = X - mu
    squared_norms = np.linalg.norm(inside, axis=1)**2
    # Compute the average of the squared norms
    average_squared_norm = np.mean(squared_norms)
    return average_squared_norm

def second_term(X,X_JS):
    first = X - X_JS
    squared_norms = np.linalg.norm(first, axis=1)**2
    # Compute the average of the squared norms
    average_squared_norm = np.mean(squared_norms)
    return average_squared_norm
def third_term(X_JS,mu):
    inside = X_JS - mu
    squared_norms = np.linalg.norm(inside, axis=1)**2
    # Compute the average of the squared norms
    average_squared_norm = np.mean(squared_norms)
    return average_squared_norm
RISK_MLE = []
RISK_JS = []

def simulation(theta,n,p):
    mu = np.zeros(p)
    mu[0] = theta
    cov = np.identity(p)
    X = np.random.multivariate_normal(mu,cov,n)
    JS_ESTIMATORS = JS_estimator(X)
    Risk_1 = first_term(X,mu)
    RISK_MLE.append(Risk_1)
    
    Risk_2 = second_term(X,JS_ESTIMATORS)
    
    Risk_3 = third_term(JS_ESTIMATORS,mu)
    RISK_JS.append(Risk_3)
    print(f"################################## p = {p} ##################################\n")
    print(f" p = {p} -> {Risk_1} = {Risk_2} + {Risk_3} = {Risk_2 + Risk_3}")
    print("##################################\n\n")
P = [3,10,100, 150, 200, 250, 300, 350, 400, 450, 500]
for p in P:
    simulation(theta = 10,n = 250, p =p)

plt.figure(figsize=(12, 8))
plt.plot(P, RISK_MLE, label = 'Risk MLE', color = 'black')
plt.plot(P, RISK_JS, label = 'Risk JS', color = 'red')
plt.title('Risk of MLE VS Risk JS estimator', fontsize=25)
plt.legend()
plt.ylabel('Risk', fontsize=25)
plt.xlabel('p', fontsize=25)
plt.show()
from scipy.special import gamma, gammaln
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stats

def expected_mu(k):
    return np.exp(np.log(np.sqrt(2)) + gammaln((p)/2) - gammaln((p-1)/2))
def simulation(p,theta):
    n = 15
    plt.figure(figsize=(12, 8))
    for _ in range(n):
        mu = np.zeros(p)
        mu[0] = theta
        cov = np.identity(p)
        X = np.random.multivariate_normal(mu,cov,n)
        # E(||u||), which is helpful in getting (E{X_1}, E{||U||}), the center of the distribution in the z-sapce
        
        expected_mu_u = expected_mu(p)
        '''MLE estimate'''
        X_1 = X[0,0]
        U_norm = np.linalg.norm(X[0,1:])
        ''''''
        
        ''''James-Stein estimate'''
        # norm of X squared
        X_shaped = X[0]
        X_norm_squared = np.linalg.norm(X_shaped)**2
        factor = (1-(p-2)/X_norm_squared)
        X_JS = X * factor
        X_JS_1 = X_JS[0,0]
        U_JS_norm = np.linalg.norm(X_JS[0,1:])
        ''''''
        '''Please ignore why I did this if statement, just for the legend of the plot to be clean'''
        if _ == n-1:
            # plot the point sampled from the distribution (which is the MLE estimate)
            plt.plot(X_1,U_norm, 'yellow', marker='o', label='MLE estimate')
            # plotting JS estimate
            plt.plot(X_JS_1, U_JS_norm, 'black', marker='o', label='JS estimate')
            plt.plot([0,X_1,X_JS_1],[0,U_norm,U_JS_norm], label = 'Proof on same line', color = 'green', alpha = 0.5)
        else:
            # plotting JS estimate
            plt.plot(X_JS_1, U_JS_norm, 'black', marker='o')
            # draw the point sampled from the distribution (which is the MLE estimate)
            plt.plot(X_1,U_norm, 'yellow', marker='o')
            '''Plotting this line just to show that indeed (0,0), the vector X(i.e. [X_1, U_norm]) and the JS estimate are on the same line'''
            plt.plot([0,X_1,X_JS_1],[0,U_norm,U_JS_norm],color = 'green', alpha = 0.5)
    # plot the line from the origin to the point (theta,expected_mu_u)
    plt.plot([0,theta],[0,expected_mu_u],'r-', label = 'Line from origin to the mean of the distribution in z-space')
    # plot the perpendicular 
    X_, y_ = get_perp_coord(theta, expected_mu_u)
    plt.plot([theta,X_],[0,y_],'b-', label = 'Perpendicular Line')
    # plot the point (theta, expected_mu_u)
    plt.plot(theta,expected_mu_u,'purple',marker='x', label = 'Mean of the dirsribution in z-space',markersize=20)
    # plot the point (theta,0)
    plt.plot(theta,0,'ro', label = 'True mean')
    # plot the hyptenuse of the triangle from (theta,0) to (theta,expected_mu_u)
    plt.plot([theta,theta],[0,expected_mu_u],color = 'black', label = 'Hypotenuse')
    plt.gca().set_aspect('equal', adjustable='box')
    
    if p == 1000:
        plt.xlim(-1,expected_mu_u+10)
    else:
        plt.xlim(-1, theta + 5)  # Adjust x-axis limits
    plt.ylim(-1, expected_mu_u + 5)  # Adjust y-axis limits
    plt.legend(loc='best')
    plt.title(f'p = {p}, theta = {theta}')
    plt.show()

def get_perp_coord(theta, expected_mu_u):
    x = (theta**3)/(theta**2 + expected_mu_u**2)
    y = ((theta**2)*expected_mu_u) / (theta**2 + expected_mu_u**2)
    return x,y
for p in [3,10,100,1000]:
    simulation(p,10)
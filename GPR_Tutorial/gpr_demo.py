# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 21:57:17 2021

@author: sqin34
"""

import argparse
import numpy as np
from numpy.linalg import cholesky, lstsq
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Reference https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote15.html
# Reference https://github.com/TimoFlesch/GaussianProcess/blob/master/gaussian_process.ipynb

parser = argparse.ArgumentParser(description='GPR 1D DEMO')
parser.add_argument('--npoint',default=10,type=int,
                    help='number of training points')
parser.add_argument('--nsample',default=1,type=int,
                    help='number of samples')
def plot_gp(ax, mu,covmat,X,X_train=None,y_train=None,samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(covmat))
    ax_fill = ax.fill_between(X,mu+uncertainty,mu-uncertainty,
                              facecolor='C0',alpha=0.2)
    ax_test, = ax.plot(X, mu, label='Mean')
    ax_samples = []
    ax.set_xlim([-2,2])
    ax.set_ylim([-5,5])
    for i, sample in enumerate(samples):
        ax_sample, = ax.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
        ax_samples.append(ax_sample)
    if X_train is not None:
        ax.scatter(X_train, y_train, s=8)
    plt.legend()
    return ax_test,ax_fill,ax_samples

def update_gp(ax_test,ax_fill,mu,covmat,X,
              ax_samples=None,
              samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(covmat))
    ax_fill.axes.collections.clear()
    ax_fill.axes.fill_between(X,mu+uncertainty,mu-uncertainty,
                              facecolor='C0',alpha=0.2)
    ax_test.set_ydata(mu)
    for i,sample in enumerate(samples):
        ax_samples[i].set_ydata(sample)

    return

def kernel(X1, X2, k_type='RBF', len_scale=1, sig_var=1):
       
    return sig_var*np.exp(-0.5/len_scale**2*(np.sum(X1**2,1).reshape(-1,1)+np.sum(X2**2,1)-2*np.dot(X1,X2.T)))

def sufficient_statistics(X_test,X_train,y_train, 
                          len_scale=1,sig_var=1,noise_var=1e-5, prior="RBF"):
    K = kernel(X_train,X_train,prior,len_scale,sig_var)+noise_var*np.eye(len(X_train))
    Ks = kernel(X_train,X_test,prior,len_scale,sig_var)
    Kss = kernel(X_test,X_test,prior,len_scale,sig_var)+noise_var*np.eye(len(X_test))

    # mean 
    mu_test = Ks.T.dot(np.linalg.inv(K)).dot(y_train)
    # covariance 
    covmat_test = Kss-Ks.T.dot(np.linalg.inv(K)).dot(Ks)
    return mu_test, covmat_test

def objective_function(X_train, y_train, noise_var):
    
    def loss(theta):        
        K = kernel(X_train,X_train,len_scale=theta[0],sig_var=theta[1])+noise_var*np.eye(len(X_train))
        L = cholesky(K)
        loss_value = np.sum(np.log(np.diagonal(L)))+0.5*y_train.T.dot(lstsq(L.T,lstsq(L,y_train,rcond=None)[0],rcond=None)[0])+0.5*len(X_train)*np.log(2*np.pi)
        return loss_value[0][0]
    
    return loss

def main(n_points,n_sample):
    
    X_train = []
    y_train = []
    
    X_test = np.linspace(-2, 2, 41)
    X_test = np.expand_dims(X_test,1)
    
    noise_var = 0.04
    theta = [1,1]
    mu_test_initial = np.zeros(X_test.shape)
    covmat_test_initial = np.eye(X_test.shape[0])*theta[1]
    samples = np.random.multivariate_normal(mu_test_initial.ravel(),
                                            covmat_test_initial,n_sample)
    
    fig = plt.figure('demo')
    ax = fig.add_subplot(111)
    ax_test,ax_fill,ax_samples = plot_gp(ax,mu_test_initial,covmat_test_initial,X_test,
                                         samples = samples)
    
    
    for n_iter in range(n_points):
            
        (x_train_input,y_train_input) = plt.ginput(1)[0]
    
        X_train.append(x_train_input)
        y_train.append(y_train_input)
    
        res = minimize(objective_function(np.array(X_train).reshape(-1,1),np.array(y_train).reshape(-1,1),noise_var), 
                        [1, 1], bounds=((1e-3,1e3), (1e-3,1e3)), method='L-BFGS-B')
    
        len_scale_hat, sig_var_hat = res.x
        # len_scale_hat,sig_var_hat = [1,1]
    
        mu_test,covmat_test = sufficient_statistics(X_test,
                                                    np.array(X_train).reshape(-1,1),
                                                    np.array(y_train).reshape(-1,1),
                                                    len_scale=len_scale_hat,
                                                    sig_var=sig_var_hat,
                                                    noise_var=noise_var)
        samples = np.random.multivariate_normal(mu_test.ravel(),
                                                covmat_test,n_sample)
        update_gp(ax_test,ax_fill,
                  mu_test,covmat_test,X_test,
                  ax_samples,samples)
        ax.scatter(X_train,y_train,c='black',s=8,label='Train')
        ax.set_title('iter{},l={:.3g},sigma={:.3g}'.format(n_iter+1,len_scale_hat,sig_var_hat))
        fig.canvas.draw()
        fig.canvas.flush_events()
    plt.close()
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args.npoint,args.nsample)
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm as Norm
import matplotlib.pyplot as plt
import os


def Ridge(A,b,lam):
    if lam != 0: return np.linalg.solve(A.T.dot(A)+lam*np.eye(A.shape[1]), A.T.dot(b))
    else: return np.linalg.lstsq(A, b ,rcond=None)[0]
    
def SGTRidge(Xs, ys, tol, lam = 10**-4, maxit = 1, verbose = True):

    if len(Xs) != len(ys): raise Exception('Number of Xs and ys mismatch')
    if len(set([X.shape[1] for X in Xs])) != 1: 
        raise Exception('Number of coefficients inconsistent across timesteps')
        
    d = Xs[0].shape[1]
    m = len(Xs)
    
    W = np.hstack([Ridge(X,y,lam) for [X,y] in zip(Xs,ys)])
    
    num_relevant = d
    biginds = [i for i in range(d) if np.linalg.norm(W[i,:]) > tol]
    
    for j in range(maxit):
        
        smallinds = [i for i in range(d) if np.linalg.norm(W[i,:]) < tol]
        new_biginds = [i for i in range(d) if i not in smallinds]
            
        if num_relevant == len(new_biginds): j = maxit-1
        else: num_relevant = len(new_biginds)
            
        if len(new_biginds) == 0:
            if j == 0 and verbose: 
                print("Tolerance too high - all coefficients set below tolerance")
            break
        biginds = new_biginds
        
        for i in smallinds:
            W[i,:] = np.zeros(m)
        if j != maxit -1:
            for i in range(m):
                W[biginds,i] = Ridge(Xs[i][:, biginds], ys[i], lam).reshape(len(biginds))
        else: 
            for i in range(m):
                W[biginds,i] = np.linalg.lstsq(Xs[i][:, biginds],ys[i],rcond=None)[0].reshape(len(biginds))
                
    return W


def Loss(As,bs,x,epsilon=10**-2):

    D,m = x.shape
    n,_ = As[0].shape
    N = n*m
    rss = np.sum([np.linalg.norm(bs[j] - As[j].dot(x[:,j].reshape(D,1)))**2 for j in range(m)])  
    k = np.count_nonzero(x)/m

    return N * np.log(rss/N+epsilon) + 2 * (k+1) * (k+2)/(N-k-2) + 2 * k


def TrainSGTRidge(As, bs, num_tols, lam =  10**-4, normalize = 0):
    np.random.seed(0)
    m = len(As)
    n,D = As[0].shape
    
    if normalize != 0:
        candidate_norms = np.zeros(D)
        for i in range(D):
            candidate_norms[i] = Norm(np.vstack(A[:,i] for A in As), normalize)

        norm_bs = [m*Norm(b, normalize) for b in bs]

        for i in range(m):
            As[i] = As[i].dot(np.diag(candidate_norms**-1))
            bs[i] = bs[i]/norm_bs[i]
    
    x_ridge = np.hstack([Ridge(A,b,lam) for (A,b) in zip(As, bs)])
    max_tol = np.max([Norm(x_ridge[j,:]) for j in range(x_ridge.shape[0])])
    min_tol = np.min([Norm(x_ridge[j,:]) for j in range(x_ridge.shape[0])])
    Tol = [0]+ [np.exp(alpha) for alpha in np.linspace(np.log(min_tol), np.log(max_tol), num_tols)][:-1]
    X = []
    Losses = []

    for tol in Tol:
        x = SGTRidge(As,bs,tol)
        X.append(x)
        Losses.append(Loss(As, bs, x))

    if normalize != 0:
        for x in X:
            for i in range(D):
                for j in range(m):
                    x[i,j] = x[i,j]/candidate_norms[i]*norm_bs[j]
        for i in range(m):
            As[i] = As[i].dot(np.diag(candidate_norms))
            bs[i] = bs[i]*norm_bs[i]
            
    return X,Tol,Losses

if __name__ == '__main__':
    
        if not os.path.exists('./plot_data'):
            os.makedirs('./plot_data')

        x = np.load('data/candi_data.npy')[:,:-1]
        c = np.ones((x.shape[0],1),dtype=np.complex128)
        x = np.column_stack((c,np.real(x)))
  
        n,d = x.shape
        _x = np.zeros((n,d), dtype=np.complex128)
        Mreg = np.zeros((d,1), dtype=np.complex128)
        for j in range(0,d):
                Mreg[j] = 1.0/(np.linalg.norm(x[:,j],2))
                _x[:,j] = Mreg[j]*x[:,j]
        x = np.real(_x)

        y = np.load('data/candi_data.npy')[:,-1][:,np.newaxis]
        Mreg = 1.0/(np.linalg.norm(y,2))
        y = Mreg*y
        
        u = np.load('data/theta.npy')
        idx = np.argsort(np.squeeze(u))
        u = u[idx]
        x = x[idx]
        y = y[idx]
        
        upper = np.max(u)
        lower = np.min(u)
        win_size = 0.001
        min_lim = 200
        idx=[]
        
        _lower = lower
        _upper = lower + win_size
        
        while True:
            _idx = (np.array(np.where((u>=_lower) & (u<=_upper)))).T[:,0]
            if _idx[-1] + min_lim < (np.array(np.where(u==upper))).T[0,0]:
                if _idx.shape[0] > min_lim:
                    idx.append(_idx)
                else:
                    _idx = np.arange(_idx[0],_idx[0]+min_lim)
                    idx.append(_idx)
            else:
                _idx = np.arange(_idx[0],(np.array(np.where(u==upper))).T[0,0])
                idx.append(_idx)
                break
            
            _lower = u[_idx[-1],0]
            _upper = _lower + win_size
        
        
        Theta_grouped=[]
        Ut_grouped = []
        
        for i , _idx in enumerate(idx):
            _x = x[_idx,:]
            _y = y[_idx]
            Theta_grouped.append(_x)
            Ut_grouped.append(_y)
    
    num_tols = 200
    Xi,Tol,Losses = TrainSGTRidge(Theta_grouped, Ut_grouped, num_tols)
    xi = Xi[np.argmin(Losses)]
    
    np.save('data/Theta_grouped',Theta_grouped)
    np.save('data/Ut_grouped',Ut_grouped)
    np.save('plot_data/Xi',Xi)
    np.save('plot_data/Tol',Tol)
    np.save('plot_data/Losses',Losses)
    
    plt.plot(np.linspace(0,num_tols-1,num_tols),Losses,linewidth=3.0,linestyle='--')
    plt.show()
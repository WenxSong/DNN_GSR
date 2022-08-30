# -*- coding: utf-8 -*-
"""

@author: Wenxiang Song

"""

import numpy as np
from numpy.linalg import norm as Norm
import matplotlib.pyplot as plt

plt.rc('font',family='Times New Roman')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'stix'


class van_G_para:
    def __init__(self,texture):
        
        if texture == 'loam':
            self.Ks = 0.2496
            self.theta_s= 0.43
            self.theta_r= 0.078
            self.alpha= 3.6
            self.n= 1.56
        
        elif texture == 'sandy_loam':
            self.Ks =0.108
            self.theta_s= 0.45
            self.theta_r= 0.067
            self.alpha=2
            self.n= 1.41
            
        elif texture == 'silt_loam':
            self.Ks =1.061
            self.theta_s= 0.41
            self.theta_r= 0.065
            self.alpha=7.5
            self.n= 1.89
        
    def se(self,theta):
        se = (theta - self.theta_r)/ (self.theta_s-self.theta_r)
        return se
    
    def VG_dz (self,theta):
        se =  self.se(theta)
        m = 1 -1/self.n
        a = 1- np.power(se,1/m)
        coef = - 0.5 * self.Ks * np.power(se,-0.5) * np.square(1 - np.power(a,m))- 2 * self.Ks * np.power(se,-0.5+1/m) * (1-np.power(a,m)) * np.power(a,m-1)
        coef = (1/(self.theta_s - self.theta_r)) * coef
        return coef
    
    def VG_d2z (self,theta):
        se =  self.se(theta)
        m = 1 -1/self.n
        a = 1- np.power(se,1/m)

        coef = np.power(se,0.5-1/m-1) * np.square(1-np.power(a,m)) * np.power(np.power(se,-1/m),1/self.n-1)
        coef = (self.Ks/((self.theta_s - self.theta_r)*(self.alpha * self.n *m))) * coef
        return coef
    
    def VG_dz2 (self,theta):
        se =  self.se(theta)
        m = 1 -1/self.n
        a = 1-np.power(se,1/m)
        D = self.VG_d2z(theta)
        coef = (0.5-1/m-1)*np.power(se,-1)+2*np.power(se,1/m-1)*np.power(1-np.power(a,m),-1)*np.power(a,m-1)+(1/m)*(1-1/self.n)*np.power(se,-1/m-1)*np.power(np.power(se,-1/m)-1,-1)
        coef  = (D/(self.theta_s - self.theta_r))* coef
        return coef
    
    
    
    
def Ridge(A,b,lam):
    if lam != 0: return np.linalg.solve(A.T.dot(A)+lam*np.eye(A.shape[1]), A.T.dot(b))
    else: return np.linalg.lstsq(A, b ,rcond=None)[0]


def SGTRidge(Xs, ys, tol, lam, maxit = 1, verbose = True):

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


def AIC(As,bs,x,epsilon= 10**-2):

    D,m = x.shape
    n,_ = As[0].shape
    N = n*m
    rss = np.sum([np.linalg.norm(bs[j] - As[j].dot(x[:,j].reshape(D,1)))**2 for j in range(m)])  
    k = np.count_nonzero(x)/m
    
    return N * np.log(rss/N + epsilon ) + 2 * k + 2 * (k+1) * (k+2)/(N-k-2) 


def TrainSGTRidge(As, bs, num_tols, norm_x, norm_y, lam = 10**-4):

    np.random.seed(0)

    n,D = As[0].shape
    
    x_ridge = np.hstack([Ridge(A,b,lam) for (A,b) in zip(As, bs)])
    max_tol = np.max([Norm(x_ridge[j,:]) for j in range(x_ridge.shape[0])])
    min_tol = np.min([Norm(x_ridge[j,:]) for j in range(x_ridge.shape[0])])
    Tol = [0]+ [np.exp(alpha) for alpha in np.linspace(np.log(min_tol), np.log(max_tol), num_tols)][:-1]

    X = []
    Losses = []

    for tol in Tol:
        x = SGTRidge(As,bs,tol,lam)
        X.append(x)
        Losses.append(AIC(As,bs,x))
        
    for x in X:
        for i in range(D):
                x[i,:] = x[i,:]/norm_x[i]*norm_y
                
                
    return X,Tol,Losses


def data_group(x,y,u,win_size = 0.001, min_lim = 200):

    c = np.ones((x.shape[0],1),dtype=np.float64)
    x = np.column_stack((c,np.real(x))) #

    n,d = x.shape
    
    norm_x = np.ones((d,1))
    for j in range(0,d):
        norm_x[j] = (np.linalg.norm(x[:,j],2))
        x[:,j] = x[:,j]/norm_x[j]
    
    norm_y = np.linalg.norm(y,2)
    y = y/norm_y
    
    idx = np.argsort(np.squeeze(u))
    u = u[idx]
    x = x[idx]
    y = y[idx]
        
    upper = np.max(u)
    lower = np.min(u)
    idx=[]
        
    _lower = lower
    _upper = lower + win_size
    
    iter = 0
    while True:
        
        _idx = np.where((u>=_lower) & (u<=_upper))[0]
        if _idx[-1] + min_lim < np.where(u==upper)[0][0]:
            if  _idx.shape[0] > min_lim:
                idx.append(_idx)
                
            else:
                _idx = np.arange(_idx[0],_idx[0]+min_lim)
                idx.append(_idx)
                
            iter +=1
        else:
            _idx = np.arange(_idx[0],np.where(u==upper)[0][0])
            idx.append(_idx)
            
            break
        
        _lower = u[_idx[-1],0]
        _upper = _lower + win_size
    
    
    lower_control = int(np.ceil(0.01*len(idx)))
    upper_control = int(np.floor(0.99*len(idx)))
    
    
    lower_idx = np.hstack(idx[:lower_control])
    upper_idx = np.hstack(idx[upper_control:])

    idx = [lower_idx] + idx[lower_control:upper_control] + [upper_idx]
        
    Theta_grouped=[]
    Ut_grouped = []
    Uz_grouped = []
        
    for i , _idx in enumerate(idx):
            _x = x[_idx,:]
            _y = y[_idx]
            
            Theta_grouped.append(np.mean(u[_idx]))
            Uz_grouped.append(_x)
            Ut_grouped.append(_y)
    
    return Theta_grouped,Uz_grouped, Ut_grouped,norm_x,norm_y


if __name__ == '__main__':
    
    data_path = 'data/loam_S1'
    true_texture = 'loam'
    
    
    x = np.load(data_path+'/candidates.npy')[:,:-1]
    y = np.load(data_path+'/candidates.npy')[:,-1][:,np.newaxis]
    u = np.load(data_path+'/collected_theta.npy')
    
    Theta_grouped,Uz_grouped,Ut_grouped,norm_x,norm_y = data_group(x, y, u, win_size=0.001)
    
    num_tols = 200
    Xi,Tol,AIC_loss = TrainSGTRidge(Uz_grouped, Ut_grouped, num_tols,norm_x,norm_y)
    
    best_term = Xi[np.argmin(AIC_loss)]
    

    class_coef=van_G_para(true_texture)
    
    
    if np.nonzero(best_term[:,0])[0].all() != np.array([1,2,4]).all():
        print('The Dicovered Euquation is Not RRE')
        
    else:
        coef_theta_z = best_term[1,:]
        coef_theta_2z = best_term[2,:]
        coef_theta_z2 = best_term[4,:]
        
        fig,ax = plt.subplots(3,1,sharex=True,figsize=(6,10))
        
        
        ax[0].plot(Theta_grouped,class_coef.VG_dz(np.array(Theta_grouped)),lw=2,label='True')
        ax[0].scatter(Theta_grouped,coef_theta_z,color='r',s=15,alpha=0.5,label='Discovered')
        ax[0].legend(frameon=False)
        ax[1].plot(Theta_grouped,class_coef.VG_d2z(np.array(Theta_grouped)),lw=2)
        ax[1].scatter(Theta_grouped,coef_theta_2z,color='r',s=15,alpha=0.5)
        
        ax[2].plot(Theta_grouped,class_coef.VG_dz2(np.array(Theta_grouped)),lw=2)
        ax[2].scatter(Theta_grouped,coef_theta_z2,color='r',s=15,alpha=0.5)
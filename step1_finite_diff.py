# -*- coding: utf-8 -*-
"""
@author: Wenxiang Song

"""


import pandas as pd
import numpy as np

class Finite:
    def __init__(self, data_u, delta_h, data_t):
        self.data_u = data_u
        self.data_t = data_t
        
        self.delta_h = delta_h[1] - delta_h[0]
        
        for i in range(1,self.data_u.shape[1]):
            delta = (self.data_u[:,i]-self.data_u[:,i-1])/(self.data_t[i]-self.data_t[i-1])
            if i==1:
                ut = delta
            elif i !=1:
                ut = np.vstack((ut,delta))
        
        for i in range(1,self.data_u.shape[0]-1):
            delta = 0.5*(self.data_u[i+1,:]-self.data_u[i-1,:])/self.delta_h
            if i==1:
                ux = delta
            elif i !=1:
                ux = np.vstack((ux,delta))
        
        for i in range(1,self.data_u.shape[0]-1):
            delta = (self.data_u[i+1,:]-2*self.data_u[i,:]+self.data_u[i-1,:])/np.square(self.delta_h)
            if i==1:
                u2x = delta
            elif i !=1:
                u2x = np.vstack((u2x,delta))
        
        for i in range(2,self.data_u.shape[0]-2):
            delta = (0.5*self.data_u[i+2,:]-self.data_u[i+1,:]+self.data_u[i-1,:]-0.5*self.data_u[i-2,:])/np.power(self.delta_h,3)
            if i==2:
                u3x = delta
            elif i !=2:
                u3x = np.vstack((u3x,delta))
        
        self.u = self.data_u[2:-2,1:]
        self.ut = ut.T[2:-2,0:]
        self.ux = ux[1:-1,1:]
        self.u2x = u2x[1:-1,1:]
        
        
        if delta_h.shape[0] == 5 :            
            self.u3x = u3x[1:][np.newaxis,:]
            
        elif delta_h.shape[0] > 5  :
            self.u3x = u3x[0:,1:]
        
        self.ux_2 = np.power(self.ux ,2)
        self.u2x_2 = np.power(self.u2x ,2)
        self.u3x_2 = np.power(self.u3x ,2)
        
        self.ux_u2x = self.ux * self.u2x
        self.ux_u3x = self.ux * self.u3x
        self.u2x_u3x = self.u2x * self.u3x
        
    def generate_lib(self, N_u=None):

        candidate_lib = [self.u, self.ux,  self.u2x, self.u3x,self.ux_2,  self.u2x_2,
                         self.u3x_2,  self.ux_u2x,  self.ux_u3x,  self.u2x_u3x, self.ut]
        
        data_list = []
        for i in range(len(candidate_lib)):
            _u = candidate_lib[i]
            _u = np.vstack((np.full([2,_u.shape[1]], np.nan),np.vstack((_u,np.full([2,_u.shape[1]], np.nan)))))
            _u = np.concatenate((np.full([_u.shape[0],1], np.nan),_u), axis=1)
            __u = _u[2:-2,1:]
            data_list.append(__u.flatten()[:,None])
            
            
        theta = data_list[0]

        for i in range(len(data_list)):
            
            if i ==0:
                pass
            elif i  == 1:
                candi_lib = data_list[i]
            else:
                _candi_lib = data_list[i]
                candi_lib = np.hstack((candi_lib,_candi_lib))
                
                
        return theta, candi_lib
    
if __name__ == "__main__": 
    data_path = 'data/loam_S1'

    data = pd.read_csv(data_path+'/th.txt',delim_whitespace=True,header=None)


    measured_data_points = [10,12,14,16,18]

    depth = np.array(measured_data_points)*0.01
    time_series = np.array(data.iloc[:,0])
    measured_data_points = [points+1 for points in measured_data_points]
    measured_data = np.array(data.iloc[:,measured_data_points])



    diff_er = Finite(measured_data.T,depth,time_series)

    theta, candidates = diff_er.generate_lib()

    np.save(data_path+'/collected_theta',theta)
    np.save(data_path+'/candidates',candidates)
# -*- coding: utf-8 -*-
"""

@author: Wenxiang Song

"""


import tensorflow as tf
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



class DNN:
    def __init__(self, train_points, theta, layers_theta):

        self.t = train_points[:,1:2]
        self.z = train_points[:,0:1]
        self.theta = theta
        
        self.upper = np.max(theta)
        self.lower = np.min(theta)
        
        self.layers_theta = layers_theta
        
        self.weights_theta, self.biases_theta = self.initialize_NN(layers_theta)

        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)

        self.t_tf = tf.placeholder(tf.float32, shape = [None, 1])
        self.z_tf = tf.placeholder(tf.float32, shape = [None, 1])
        self.theta_tf = tf.placeholder(tf.float32, shape = [None, 1])
        
        self.loss =  tf.reduce_mean(tf.square(self.net_theta(self.t_tf, self.z_tf) - self.theta_tf))
        
        self.theta_pred = self.net_theta(self.t_tf, self.z_tf)
        
        self.z_pred_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_pred_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.meta_data_pred = self.net_meta_data(self.t_pred_tf, self.z_pred_tf)
        
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                        method = 'L-BFGS-B',
                                                        options = {'maxiter': 10000,
                                                                    'maxfun': 50000,
                                                                    'maxcor': 50,
                                                                    'maxls': 50,
                                                                    'ftol' : 1.0 * np.finfo(float).eps})
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
    

        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    def xavier_init(self, size):
        
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev = xavier_stddev), dtype = tf.float32)
    
    def initialize_NN(self, layers):
        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
    
    
    def net_theta(self, t, z):
        t = 2.0*(t - np.min(self.t))/(np.max(self.t) - np.min(self.t)) - 1.0
        z = 2.0*(z - np.min(self.z))/(np.max(self.z) - np.min(self.z)) - 1.0
        X = tf.concat([t, z],1)
        weights = self.weights_theta
        biases = self.biases_theta
        num_layers = len(weights) + 1
        
        H = X 
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        
        return Y

    def net_meta_data(self, t, z):
        
        u = self.net_theta(t, z)
        ut = tf.gradients(u, t)[0]
        ux = tf.gradients(u, z)[0]
        u2x = tf.gradients(ux, z)[0]
        u3x = tf.gradients(u2x, z)[0]
        ux_2 = tf.math.square(ux) 
        ux_u2x = tf.multiply(ux, u2x)
        ux_u3x = tf.multiply(ux, u3x)
        u2x_2 = tf.math.square(u2x)
        u2x_u3x = tf.multiply(u2x, u3x)
        u3x_2 = tf.math.square(u3x)
        
        meta_data =[u, ux, u2x, u3x, ux_2, u2x_2, u3x_2, ux_u2x, ux_u3x, u2x_u3x, ut]
        return meta_data
    
    def train(self, max_epoch, batch_size):
        
        def get_batch(z,t,theta,batch_size,it_one_epcoh):
            idx = np.random.choice(theta.shape[0], theta.shape[0], replace=False)
            z = z[idx]
            t = t[idx]
            theta = theta[idx]
            z_batch = []
            t_batch = []
            theta_batch = []
            for it in range(it_one_epcoh):
                if it == it_one_epcoh-1:
                    z_batch.append(z[-batch_size:,:])
                    t_batch.append(t[-batch_size:,:])
                    theta_batch.append(theta[-batch_size:,:])
                else:
                    z_batch.append(z[it*batch_size:(it+1)*batch_size,:])
                    t_batch.append(t[it*batch_size:(it+1)*batch_size,:])
                    theta_batch.append(theta[it*batch_size:(it+1)*batch_size,:])
            
            return z_batch, t_batch, theta_batch
        
        it_one_epcoh = int(np.ceil(self.theta.shape[0]/batch_size))
        
        for it in range(max_epoch):
            z_batch , t_batch , theta_batch = get_batch(self.z, self.t, self.theta, batch_size, it_one_epcoh)
            start_time = time.time()
            
            for i in range(it_one_epcoh):
                
                train_u = theta_batch[i]
                train_t = t_batch[i]
                train_x = z_batch[i]
    
                tf_dict = {self.z_tf: train_x, self.t_tf: train_t, self.theta_tf: train_u}
                
                self.sess.run(self.train_op_Adam, tf_dict)
                
                loss_value = self.sess.run(self.loss, {self.z_tf: self.z, self.t_tf: self.t, self.theta_tf: self.theta})

                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' %(it, loss_value, elapsed))
                
        self.optimizer.minimize(self.sess,
                                          {self.z_tf: self.z, self.t_tf: self.t, self.theta_tf: self.theta},
                                          fetches = [self.loss],
                                          loss_callback = self.callback)
    
    def callback(self, loss):
        print('Loss: ', loss)

    def predict(self, X_star):
        f_star = self.sess.run(self.meta_data_pred, {self.z_pred_tf: X_star[:,0:1], self.t_pred_tf: X_star[:,1:2]})
        return  f_star
    
def plot_theta(T,X,theta):
    fig, ax = plt.subplots(1,1,figsize=(8,4))
    mapped = ax.pcolormesh(T,X,theta,cmap='RdBu_r')
    
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.tick_params(direction='in',labelsize=12,width=1,which='major')
    ax.tick_params(direction='in',labelsize=12,width=0.5,which='minor')

    ax.set_ylabel("Depth [cm]",labelpad=10,fontsize=16)
    ax.set_xlabel("Simulate Days [day]",labelpad=14,fontsize=16)
    
    plt.subplots_adjust(wspace=0, hspace=0.4,top=0.95,bottom=0.1)
    cb = fig.colorbar(mapped,ax=ax,fraction=0.05,aspect=30,pad=0.05)
    
    cb.set_label(label = 'Volumetric Water Content ' + r'$ \theta $' + ' [-]', fontsize=16)
    cb.ax.tick_params(axis='y', direction='in',labelsize=12)


if __name__ == "__main__": 
    
    layers_theta = [2, 50, 50, 50, 50, 50, 1]

    data_path = 'data/loam_S1'
    measured_data_points = [5,10,15,20,25]
    noise = 0.01
    repeat = 5
    
    data = pd.read_csv(data_path+'/th.txt',delim_whitespace=True,header=None)
    depth = np.array(measured_data_points)*0.01
    day = np.array(data)[:,0].T
    time_series = np.array(data.iloc[:,0])
    measured_data_points = [points+1 for points in measured_data_points]
    measured_data = np.array(data.iloc[:,measured_data_points])
    
            
    X,T = np.meshgrid(depth,day)
    train_points = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

    idx = np.random.choice(train_points.shape[0], train_points.shape[0], replace=False)
    X_train = train_points[idx,:]
    theta_train = measured_data.flatten()[idx][:,np.newaxis]
    theta_train = theta_train + noise * np.std(theta_train) * np.random.randn(theta_train.shape[0], theta_train.shape[1])

    epoch = 1000
    batch_size= 512
    seeds = [1,2,3,4,5]
    for i in range(repeat):
        np.random.seed(seeds[i])
        tf.set_random_seed(seeds[i])
        DNN_model = DNN(X_train, theta_train, layers_theta)

        DNN_model.train(max_epoch = epoch, batch_size= batch_size)
        
        X,T =np.meshgrid(np.linspace(np.min(depth),np.max(depth),20),day)
        predict_points = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        f_pred = DNN_model.predict(predict_points)
        
        if i ==0:
            candi_lib = (np.array(f_pred).squeeze()).T
        else:
            candi_lib = ((np.array(f_pred).squeeze()).T + i*candi_lib)/(i+1)
            
np.save(data_path+'/collected_theta',candi_lib[:,0:1])
np.save(data_path+'/candidates',candi_lib[:,1:])
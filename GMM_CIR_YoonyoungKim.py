# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 20:09:55 2018

@author: Yoonyoung Kim

마지막 부분의 filename = os.path.abspath('~ data_CD.xlsx')의 ~부분을 엑셀파일이 저장된 directory로 변경해주시면 코드 정상작동합니다.
"""

import numpy as np
import pandas as pd
from scipy import stats
import scipy.optimize as optim
from statsmodels.tools.numdiff import approx_fprime
import os


#Call Data and data trimming 
def get_data(filename):
    
    data = pd.ExcelFile(filename).parse(0, index_col = 0, header = 12)[1:].astype('float64')
    data.columns = ['Rate']
    
    return data/100

#Constructing J func
def moments_func(params, data, step, gamma):

    a, b, sigma = params
    
    m1 = data['Rate'].diff().dropna() - a - b * data['Rate'].shift().dropna()  
    m2 = m1 * data['Rate'].shift().dropna() 
    m3 = np.power(m1,2) - 1/12 * sigma * np.power(data['Rate'].shift().dropna(),2 * gamma)
    m4 = m3 * data['Rate'].shift().dropna()
    
    u1,u2,u3,u4 = map(np.asarray, [m1,m2,m3,m4])
    moms = np.array([u1,u2,u3,u4]).T #(T*K) matrix
    
    m1,m2,m3,m4 = map(np.mean, [m1,m2,m3,m4])
    u = np.array([m1,m2,m3,m4])
    w = np.identity(4) if step == 1 else weight(moms)
    J = u.dot(w).dot(u)
    
    return J

#Hansen's W
def weight(moms): 
    
    obs, k = moms.shape
    momv = moms - moms.mean(0)
    w = np.dot(momv.T, momv)
    w /= obs
    
    return np.linalg.inv(w)  #(K*K) matrix 

#Optimization			
def estm_params(data, gamma):
	
    step = 1
    init = [0,0,1] # initial value of a, b, sigma
    res = optim.fmin_bfgs(moments_func, init, args = (data,step,gamma,))
    step = 2
    final = optim.fmin_bfgs(moments_func, res, args = (data,step,gamma,))
    return final

#Preparation for statistical test
def moments(params, data, gamma):
  
    a, b, sigma = params
    
    m1 = data['Rate'].diff().dropna() - a - b *data['Rate'].shift().dropna()
    m2 = m1 * data['Rate'].shift().dropna()
    m3 = np.power(m1,2) - 1/12 * sigma * np.power(data['Rate'].shift().dropna(),2*gamma)
    m4 = m3 *data['Rate'].shift().dropna()
				
    u1,u2,u3,u4 = map(np.asarray, [m1,m2,m3,m4])

    return np.array([u1,u2,u3,u4]).T #(T*K) matrix

def moments_mean(params, data, gamma):
        
    moms = moments(params, data, gamma)
    
    return moms.mean(0)  #(1*K) vector
    

# Test-statistics
def se_params(final, data, gamma):  # for t-test
 
    moms = moments(final, data, gamma) #(T*K)
    Omega = weight(moms) #(K*K)
    obs = moms.shape[0]
    omegahat = Omega
    se_d = approx_fprime(final, moments_mean, epsilon = 0.0001, args=(data,gamma,))
    cov = np.linalg.inv(np.dot(se_d.T, np.dot(omegahat, se_d)))
    
    return np.sqrt(np.diag(cov/obs)) # 1*3 vector
				
def xtest(final, data, gamma):
    
    momsm = moments_mean(final, data, gamma) #(1*K) vector
    weights = weight(moments(final, data, gamma)) #(K*K)
    xstat = np.dot(np.dot(momsm, weights), momsm) * (data.shape[0]-1)
    df = momsm.size - final.size

    return xstat, stats.chi2.sf(xstat, df), df
 
def test(gamma, data):
    final = estm_params(data, gamma)
    se = se_params(final, data, gamma)
    t_stat = final / se
    x_stat, p_val, df = xtest(final, data, gamma)
    print(final)
    print(t_stat)
    print(x_stat)
    print(p_val)
    print(df)
  
#Start Optimization		
def start():
    
    filename = os.path.abspath('/Users/na88555/Documents/MFE_18_2/논문연구방법론/assignment2/GMM_YoonyoungKim/data_CD.xlsx')
    data = get_data(filename)
    gamma = 0.5 # CIR SR
    test(gamma, data)
    
   
if __name__ == '__main__':
    start()
    
    
#!/usr/bin/python
import numpy as np
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
import time
import copy
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

from numpy.linalg import inv
from numpy import linalg as LA
import pandas as pd

import math
from scipy.sparse.linalg import svds

from sklearn import preprocessing

from sklearn.neighbors import KernelDensity


from scipy.optimize import curve_fit


import sys

def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)




file_obs='U_analysis_group1.csv'
def get_d(file_ob):
  obs_f= pd.read_csv(file_ob)
  obs_f = obs_f.values
  #print (obs_f, obs_f.flatten())
  U=obs_f.flatten()
  return U

U_g1=get_d('new_U_analysis_group1.csv')
V_g1=get_d('new_V_analysis_group1.csv')
W_g1=get_d('new_W_analysis_group1.csv')



U_g2=get_d('new_U_analysis_group2.csv')
V_g2=get_d('new_V_analysis_group2.csv')
W_g2=get_d('new_W_analysis_group2.csv')

U_g3=get_d('new_U_analysis_group3.csv')
V_g3=get_d('new_V_analysis_group3.csv')
W_g3=get_d('new_W_analysis_group3.csv')


U_g123=get_d('new_U_analysis_group123.csv')
V_g123=get_d('new_V_analysis_group123.csv')
W_g123=get_d('new_W_analysis_group123.csv')



U_gall_tr1=get_d('new_U_analysis_groupall_tr1.csv')
V_gall_tr1=get_d('new_V_analysis_groupall_tr1.csv')
W_gall_tr1=get_d('new_W_analysis_groupall_tr1.csv')



x_d=np.linspace(0,10,500)
im_dist= np.reshape(x_d,  ( len(np.array(x_d).flatten()), 1))

    

WS_gall_tr1=np.sqrt(U_gall_tr1**2 + V_gall_tr1**2 + W_gall_tr1**2) 



kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
R=np.sort(np.array(WS_gall_tr1).flatten())
L= np.reshape(R,  ( len(np.array(WS_gall_tr1).flatten()), 1))
kde.fit(L)
probgall_tr1 =  np.exp(kde.score_samples(im_dist)).tolist() # [:,None]))




WS_g1=np.sqrt(U_g1**2 + V_g1**2 + W_g1**2)
x_d=np.linspace(0,10,500)
im_dist= np.reshape(x_d,  ( len(np.array(x_d).flatten()), 1))




kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
R=np.sort(np.array(WS_g1).flatten())
L= np.reshape(R,  ( len(np.array(WS_g1).flatten()), 1))
kde.fit(L)
prob1 =  np.exp(kde.score_samples(im_dist)).tolist() # [:,None]))


'''
plt.figure(111)
y, x,_=plt.hist(WS_g1, color='b' ,bins=30,  density=True, label='PDF RANS_run$^{DA, Group 1}_{t_{a}}$', alpha=0.3)
x=(x[1:]+x[:-1])/2 # for len(x)==len(y)


def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

expected = (1, 2, 0.25, 3, 1, 0.32)
params, cov = curve_fit(bimodal, x, y, expected)
sigma=np.sqrt(np.diag(cov))
x_fit = np.linspace(x.min(), x.max(), 500)
plt.plot(x_fit, gauss(x_fit, *params[:3]), color='red', lw=1, ls="--", label='Mode 1')
plt.plot(x_fit, gauss(x_fit, *params[3:]), color='red', lw=1, ls=":", label='Mode 2')
plt.savefig('dec.png')
'''

print('uuuu')

WS_g2=np.sqrt(U_g2**2 + V_g2**2 + W_g2**2)
WS_g3=np.sqrt(U_g3**2 + V_g3**2 + W_g3**2)

WS_g123=np.sqrt(U_g123**2 + V_g123**2 + W_g123**2)



file_foam = 'RANS_OKC_NW/100/U'
#print (ff, file_foam)
with open(file_foam) as f:
   AA=f.readlines()
   B=[]
   for i in range(len(AA)):
      B.append(AA[i].replace('\n',''))
   for i in range(len(B)):
     if (B[i][0:8] == 'internal' ):
         COMPS = B[i+3:i+3+np.int(B[i+1])]

   U=[]
   V=[]
   W=[]

   for i in range(len(COMPS)):
         U.append(float(COMPS[i].split(' ')[0][1:]))
         V.append(float(COMPS[i].split(' ')[1][0:]))
         W.append(float(COMPS[i].split(' ')[2][:len(COMPS[i].split(' ')[2]) -1]))
WS_rans=[]
for i in range(len(COMPS)):
    WS_rans.append(np.sqrt((U[i]**2 + V[i]**2 + W[i]**2)))


kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
R=np.sort(np.array(WS_rans).flatten())
L= np.reshape(R,  ( len(np.array(WS_rans).flatten()), 1))
kde.fit(L)
probrans =  np.exp(kde.score_samples(im_dist)) # [:,None]))



y, x,_=plt.hist(WS_rans, color='b' ,bins=30,  density=True, label='PDF RANS_run$^{DA, Group 1}_{t_{a}}$', alpha=0.3)
x=(x[1:]+x[:-1])/2 # for len(x)==len(y)



expected = (1, 2, 0.25, 3, 1, 0.32)
params, cov = curve_fit(bimodal, x, y, expected)
sigma=np.sqrt(np.diag(cov))
x_fit = np.linspace(x.min(), x.max(), 50)

#X_train = np.vstack([shifted_gaussian, stretched_gaussian])

gm = GaussianMixture(n_components=2).fit(np.array(WS_rans).reshape(-1, 1))
#def gauss(x, mu, sigma, A):
print (gm.means_[0], np.sqrt(gm.covariances_[0]), gm.weights_[0])
print (gm.means_[1], np.sqrt(gm.covariances_[1]), gm.weights_[1])

plt.plot(x_d, norm.pdf(x_d, gm.means_[0][0], np.sqrt(gm.covariances_[0][0]))* gm.weights_[0], color='m', lw=2, ls='dashdot', label='PDF Mode 1')
plt.plot(x_d, norm.pdf(x_d, gm.means_[1][0], np.sqrt(gm.covariances_[1][0]))* gm.weights_[1], color='m', lw=2, ls=':', label='PDF Mode 2')

#plt.plot(x_fit, gauss(x_fit, *params[3:]), color='m', lw=1, ls=":", label='Mode 2')
plt.plot( x_d, probrans,'y-', label='PDF RANS_run$_{t_{a}}$')
plt.legend(fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.xlabel('Wind speed ($m.s^{-1}$)', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.savefig('dec.png')

print ('iii')



file_foam = 'les_2/100/U'
#print (ff, file_foam)
with open(file_foam) as f:
   AA=f.readlines()
   B=[]
   for i in range(len(AA)):
      B.append(AA[i].replace('\n',''))
   for i in range(len(B)):
     if (B[i][0:8] == 'internal' ):
         COMPS = B[i+3:i+3+np.int(B[i+1])]

   U=[]
   V=[]
   W=[]

   for i in range(len(COMPS)):
         U.append(float(COMPS[i].split(' ')[0][1:]))
         V.append(float(COMPS[i].split(' ')[1][0:]))
         W.append(float(COMPS[i].split(' ')[2][:len(COMPS[i].split(' ')[2]) -1]))
WS_les=[]
for i in range(len(COMPS)):
    WS_les.append(np.sqrt((U[i]**2 + V[i]**2 + W[i]**2)))


x_d=np.linspace(0,10,500)
im_dist= np.reshape(x_d,  ( len(np.array(x_d).flatten()), 1))


'''
WS_g1=preprocessing.normalize(np.reshape(WS_g1, (-1,1)), axis=0)
print (WS_g1)
WS_g2=preprocessing.normalize(np.reshape(WS_g2, (-1,1)), axis=0)
WS_g3=preprocessing.normalize(np.reshape(WS_g3, (-1,1)), axis=0)
WS_g123=preprocessing.normalize(np.reshape(WS_g123, (-1,1)), axis=0)

WS_les=preprocessing.normalize(np.reshape(WS_les, (-1,1)), axis=0)
WS_rans=preprocessing.normalize(np.reshape(WS_rans, (-1,1)), axis=0)
'''


kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
R=np.sort(np.array(WS_g1).flatten())
L= np.reshape(R,  ( len(np.array(WS_g1).flatten()), 1))
kde.fit(L)
prob1 =  np.exp(kde.score_samples(im_dist)) # [:,None]))




kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
R=np.sort(np.array(WS_g2).flatten())
L= np.reshape(R,  ( len(np.array(WS_g2).flatten()), 1))
kde.fit(L)
prob2 =  np.exp(kde.score_samples(im_dist)) # [:,




kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
R=np.sort(np.array(WS_g3).flatten())
L= np.reshape(R,  ( len(np.array(WS_g3).flatten()), 1))
kde.fit(L)
prob3 =  np.exp(kde.score_samples(im_dist)) # [:,








kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
R=np.sort(np.array(WS_g123).flatten())
L= np.reshape(R,  ( len(np.array(WS_g123).flatten()), 1))
kde.fit(L)
prob123 =  np.exp(kde.score_samples(im_dist)) # [:,


kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
R=np.sort(np.array(WS_les).flatten())
L= np.reshape(R,  ( len(np.array(WS_les).flatten()), 1))
kde.fit(L)
probles =  np.exp(kde.score_samples(im_dist)) # [:,None]))


kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
R=np.sort(np.array(WS_rans).flatten())
L= np.reshape(R,  ( len(np.array(WS_rans).flatten()), 1))
kde.fit(L)
probrans =  np.exp(kde.score_samples(im_dist)) # [:,None]))



'''



kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
R=np.sort(np.array(S3).flatten())
L= np.reshape(R,  ( len(np.array(S3).flatten()), 1))
kde.fit(L)
prob3 =  np.exp(kde.score_samples(im_dist)) # [:,None]))












kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
R=np.sort(np.array(WS).flatten())
L= np.reshape(R,  ( len(np.array(WS).flatten()), 1))
kde.fit(L)
prob4 =  np.exp(kde.score_samples(im_dist)) # [:,None]))





'''
'''
plt.figure(111)
y, x,_=plt.hist(WS_g1, color='b' ,bins=30,  density=True, label='PDF RANS_run$^{DA, Group 1}_{t_{a}}$', alpha=0.3)


def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

expected = (1, .2, 250, 2, .2, 125)
params, cov = curve_fit(bimodal, x, y, expected)
sigma=np.sqrt(np.diag(cov))
x_fit = np.linspace(x.min(), x.max(), 500)
plt.plot(x_fit, gauss(x_fit, *params[:3]), color='c', lw=1, ls="- -", label='distribution 1')
plt.plot(x_fit, gauss(x_fit, *params[3:]), color='c', lw=1, ls=":", label='distribution 2')
plt.plot( x_d, probrans,'y-', label='PDF RANS_run$_{t_{a}}$')
plt.legend(fontsize=14)
plt.savefig('dec.png')
'''



def hellinger_explicit(p, q):
    """Hellinger distance between two discrete distributions.
       Same as original version but without list comprehension
    """
    list_of_squares = []
    for p_i, q_i in zip(p, q):

        # caluclate the square of the difference of ith distr elements
        s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2

        # append 
        list_of_squares.append(s)

    # calculate sum of squares
    sosq = math.sqrt(sum(list_of_squares))    

    return sosq / math.sqrt(2)

print ('HD rans, les', hellinger_explicit (0.9*probrans, 0.96*probles ))
print ('HD rans DA, les', hellinger_explicit (prob123, 0.96*probles ))
print ('HD rans DA tr1, les', hellinger_explicit (probgall_tr1, 0.96*probles ))
#print ('HD rans, les', hellinger_explicit (0.9*probrans, 0.96*probles ))

plt.figure(221)


#plt.plot( x_d, prob,'r-')
#hist=plt.hist(WS_g1, color='b' ,bins=30,  density=True, label='PDF RANS_run$^{DA, Group 1}_{t_{a}}$', alpha=0.3)
hist=plt.hist(WS_g123, color='r' ,bins=30,  density=True, label='PDF RANS_run$^{DA, TSVD, Group 1 U Group 2 U Group 3}_{t_{a}}$', alpha=0.3)
hist=plt.hist(WS_les, color='c' ,bins=30,  density=True, label='PDF LES_run$_{t_{a}}$', alpha=0.3)
hist=plt.hist(WS_gall_tr1, color='m' ,bins=30,  density=True, label='PDF RANS_run$^{DA, PCA, Group 1 U Group 2 U Group 3}_{t_{a}}$', alpha=0.3)
hist=plt.hist(WS_rans, color='y' ,bins=30,  density=True, label='PDF RANS_run$_{t_{a}}$', alpha=0.3)




plt.plot( x_d, 0.96*probles,'c-')
plt.plot( x_d, prob123,'r-')
plt.plot( x_d, probgall_tr1,'m-')
plt.plot( x_d, 0.9*probrans,'y-')



plt.grid(True)
plt.ylabel('Density', fontsize=14)
plt.xlabel('Wind speed ($m.s^{-1}$)', fontsize=14)
plt.xlim(0,10)
plt.legend(loc=1)
plt.savefig('gusts_tr1.png')
plt.show()

print ('hhhhh tr1')


plt.figure(222)


#plt.plot( x_d, prob,'r-')
hist=plt.hist(WS_g1, color='b' ,bins=30,  density=True, label='PDF RANS_run$^{DA, Group 1}_{t_{a}}$', alpha=0.3)
hist=plt.hist(WS_g2, color='g' ,bins=30,  density=True, label='PDF RANS_run$^{DA, Group 2}_{t_{a}}$', alpha=0.3)
hist=plt.hist(WS_g3, color='m' ,bins=30,  density=True, label='PDF RANS_run$^{DA, Group 3}_{t_{a}}$', alpha=0.3)
hist=plt.hist(WS_g123, color='r' ,bins=30,  density=True, label='PDF RANS_run$^{DA, Group 1 U Group 2 U Group 3}_{t_{a}}$', alpha=0.3)
#hist=plt.hist(WS_les, color='c' ,bins=30,  density=True, label='LES', alpha=0.3)
hist=plt.hist(WS_rans, color='y' ,bins=30,  density=True, label='PDF RANS_run$_{t_{a}}$', alpha=0.3)

'''
hist=plt.hist(S2, color='g' ,bins=30,  density=True, label='S2', alpha=0.5)
hist=plt.hist(S3, color='r' ,bins=30,  density=True, label='S4', alpha=0.5)
hist=plt.hist(WS, color='m' ,bins=30,  density=True, label='S5', alpha=0.5)
print ('stat S1', np.mean(S1), np.std(S1))
print ('stat S2', np.mean(S2), np.std(S2))
print ('stat S3', np.mean(S3), np.std(S3))
print ('stat WS', np.mean(WS), np.std(WS))
'''

plt.plot( x_d, prob1,'b-')
plt.plot( x_d, prob2,'g-')
plt.plot( x_d, prob3,'m-')
plt.plot( x_d, prob123,'r-')
#plt.plot( x_d, probles,'c-')
plt.plot( x_d, probrans,'y-')

plt.grid(True)
plt.ylabel('Density', fontsize=14)
plt.xlabel('Wind speed ($m.s^{-1}$)', fontsize=14)
plt.xlim(0,10)
plt.legend(loc=1)
plt.savefig('gusts.png')
plt.show()




from scipy.stats import kurtosis, skew

print('skew pro 1 ',skew(prob1, axis=0, bias=True))
print('skew pro 2 ',skew(prob2, axis=0, bias=True))
print('skew pro 3 ',skew(prob3, axis=0, bias=True))
print('skew pro 123 ',skew(prob123, axis=0, bias=True))
print('skew pro rans ',skew(probrans, axis=0, bias=True))




print('kurto pro 1 ',kurtosis(prob1, axis=0, bias=True))
print('kurto pro 2 ',kurtosis(prob2, axis=0, bias=True))
print('kurto pro 3 ',kurtosis(prob3, axis=0, bias=True))
print('kurto pro 123 ',kurtosis(prob123, axis=0, bias=True))
print('kurto pro rans ',kurtosis(probrans, axis=0, bias=True))


def hellinger_explicit(p, q):
    """Hellinger distance between two discrete distributions.
       Same as original version but without list comprehension
    """
    list_of_squares = []
    for p_i, q_i in zip(p, q):

        # caluclate the square of the difference of ith distr elements
        s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2

        # append 
        list_of_squares.append(s)

    # calculate sum of squares
    sosq = math.sqrt(sum(list_of_squares))    

    return sosq / math.sqrt(2)


print ('Hellinger group 1 obs', hellinger_explicit(prob1 ,probrans ) )
print ('Hellinger group 2 obs', hellinger_explicit(prob2 ,probrans ) )
print ('Hellinger group 3 obs', hellinger_explicit(prob3 ,probrans ) )

print ('Hellinger group 123 obs', hellinger_explicit(prob123 ,probrans ) )
#print ('Hellinger rans obs', hellinger_explicit(probrans ,probles ) )


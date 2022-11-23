#!/usr/bin/python
import numpy as np
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
import time
import copy
import pandas as pd

group='groupall_tr1'
file_obs='new_U_analysis_'+group+'.csv'
obs_f= pd.read_csv(file_obs)
obs_f = obs_f.values
#print ('obs_f', obs_f)
U=[]
for u in range(len(obs_f)):
  #  print (obs_f[u])
    LL=obs_f[u][0] #.split(' ')
    U.append(float(LL))

print ('U')
file_obs='new_V_analysis_'+group+'.csv'
obs_f= pd.read_csv(file_obs)
obs_f = obs_f.values
#print ('obs_f', obs_f)
V=[]
for u in range(len(obs_f)):
 #   print (obs_f[u])
    LL=obs_f[u][0] #.split(' ')
    V.append(float(LL))

print ('V')


file_obs='new_W_analysis_'+group+'.csv'
obs_f= pd.read_csv(file_obs)
obs_f = obs_f.values
#print ('obs_f', obs_f)
W=[]
for u in range(len(obs_f)):
#    print (obs_f[u])
    LL=obs_f[u][0] #.split(' ')
    W.append(float(LL))

print ('W')



dir = './'
file1 = open(dir + "UVW_"+group+".out","w")

for i in range(len(U)):
             file1.write('(' + str(np.round(U[i],9)) + ' ' + str(np.round(V[i],9)) + ' ' +str(np.round(W[i])) + ')'+'\n')

file1.close()


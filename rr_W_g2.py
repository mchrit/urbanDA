#!/usr/bin/python
import numpy as np
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
import time
import copy

from numpy.linalg import inv
from numpy import linalg as LA
import pandas as pd

import math
from scipy.sparse.linalg import svds


import sys
sys.path.append('fluidity-master')
#import vtktools


ntime = 20
ntimelocal = 20

Ba_time = 0
#obs_time = 19



#indLoc = np.loadtxt('/home/mounir.chrit/optimal-placement/sensor-placement-master/data/csv_data/index.txt') # indexes of nodes
#pos = pd.read_csv('/home/mounir.chrit/optimal-placement/sensor-placement-master/data/csv_data/positions.csv')
#tracer_file ='/home/mounir.chrit/hampton-NE_WS=3.281123477404996_WD=45.0/tracer_time.csv' # '/home/mounir.chrit/optimal-placement/sensor-placement-master/data/csv_data/tracer_new-new.csv'
tracer_file = 'RANS_OKC_NW/tracer_W_time_all.csv' #'/home/mounir.chrit/hampton-NE_WS=3.281123477404996_WD=45.0/tracer_time_all.csv'
tracer= pd.read_csv(tracer_file)
tracer = tracer.values
#pos = pos[['X', 'Y', 'Z']].copy().values

'''
Nx=110
Ny=110
Nz=30
pos=np.zeros([3, Nz])
pos=[]
pos.append(np.arange(-10,100,1))
pos.append(np.arange(-10,100,1))
pos.append(np.arange(-10,100,1))
'''
#print ('---', pos, 'tracer', tracer[0], '***',len(tracer))

indLoc= range(len(tracer)) #[52 , 0 , 4,  3, 53, 15, 49, 32, 21 ,16, 33,  8, 48, 14, 45 ] #np.arange(0,3*len(pos[0]))
NindLoc = len(indLoc)


def da():
    file_obs='RANS_OKC_NW/index_time_W.csv'
    obs_f= pd.read_csv(file_obs)
    obs_f = obs_f.values
    print ('obs_f', obs_f)
    index_obs_all=[]
    obs_all=[]
    for u in range(len(obs_f)):
        LL=obs_f[u][0].split(' ')
        obs_all.append(float(LL[1]))
        index_obs_all.append(int(LL[2]))
    print ('obs_all', obs_all)
    print ('index_obs_all', index_obs_all)
    uvwTot = np.array([])
    for i in range(ntimelocal):
        '''
        filename = './data/raw_data/subdomains/LSBU_*_6/LSBU_'+str(i+1)+'_6.vtu' # path to all vtu files
        ug=vtktools.vtu(filename)
        ug.GetFieldNames()
        xFluidity = ug.GetScalarField('TracerGeorge') # Output variable
        xMLocal = np.array([])
        '''
        xMLocal=[]
        #print (NindLoc)
        for j in range(NindLoc):
            indexLocal = indLoc[j]
            indexLocal = int(indexLocal)
            xMpoint = tracer[indexLocal,i]
            xMLocal = np.append(xMLocal,xMpoint)
        '''
#        print (random(NindLoc))
#        uvwTot = np.append(uvwTot,np.random.normal(10,3,NindLoc))
        '''
        uvwTot = np.append(uvwTot, xMLocal )

        #print ('uvwTot',uvwTot,len(uvwTot))

    n = NindLoc

    dimuvwTot = len(uvwTot)
    print (dimuvwTot,NindLoc,n,ntime,uvwTot)
    m = np.array([])
    m = np.zeros(int(dimuvwTot/ntime))
    for j in range(n): #every node
            for i in range(1,ntime+1):
                    m[j] = np.copy(m[j] + uvwTot[j+(i-1)*n])
            m[j] = m[j]/ntime #temporal mean for every node



    err = np.array([])
    err = np.zeros(dimuvwTot)

    for j in range(n):
            for i in range(1,ntime+1):
    #                print ('******',n, ntime)
                    err[j+(i-1)*j] = (uvwTot[j+(i-1)*j]-m[j])

    W =  np.transpose(np.reshape(err, (ntime,n)))

    #trnc = 4
    #Utrunc, strunc, Wtrunc = svds(W, k=trnc)
    uu, ss, ww= np.linalg.svd(W, False)
    sing_1 = ss[0]
    threshold = np.sqrt(sing_1)

    trunc_idx = 0 #number of modes to retain
    for sing in ss:
        if sing >= threshold:
                trunc_idx += 1
    #if trunc_idx == 0: #when all singular values are < 1
    #        trunc_idx = 1
    #strunc=ss[:]
    trunc_idx = 1

    Utrunc = uu[:, :trunc_idx]
    Wtrunc = ww[:trunc_idx, :]
    strunc = ss[:trunc_idx]

    print ('ss-----index ',trunc_idx,  uu, ss, ww)
    #X = Utrunc.dot(np.diag(np.sqrt(strunc)))
    X = Utrunc @ np.diag(strunc) @ Wtrunc
    #X = Utrunc * strunc @ Wtrunc

#np.savetxt("./validation/matrixVpreclocal"+str(trnc)+".txt", X)
    print ('truc u v', Utrunc, Wtrunc)
    print ('s', strunc, np.diag(np.sqrt(strunc)))
    V = X.copy()

    #lam = 1e-60


    #put the observation file (time step 536)
    print ('------',NindLoc, indLoc)
    '''
    ugg=vtktools.vtu('./data/raw_data/subdomains/LSBU_*_6/LSBU_536_6.vtu') #the last one
    ugg.GetFieldNames()
    uvwVecobstot = ugg.GetScalarField('TracerGeorge') # n2_TracerFluidity_WT?
    uvwVecobs = np.array([])
    for i in range(NindLoc):
        indexLocal = indLoc[i]
        indexLocal = int(indexLocal)
        xMpointobs = uvwVecobstot[indexLocal]
        uvwVecobs = np.append(uvwVecobs,xMpointobs)
     '''
    print ('uvwTot',uvwTot, len(uvwTot))
    #uvwVecobs =uvwTot[obs_time*len(indLoc):(obs_time+1)*len(indLoc)]
    uvwVecobs =[]
    index_obs=[]
#    H=np.zeros([len(uvwVecobs), len(tracer)])
    obs_f= pd.read_csv(file_obs)
    obs_f = obs_f.values
    print ('obs_f', obs_f)
    '''
    index_obs_all=[]
    obs_all=[]
    for u in range(len(obs_f)):
        LL=obs_f[u][0].split(' ')
        obs_all.append(float(LL[1]))
        index_obs_all.append(int(LL[2]))
    print ('obs_all', obs_all)
    print ('index_obs_all', iindex_obs_all)
    '''
    #Group 1
    #nstart=0
    #nend=6
    group ='group2_tr1'
    #Group 2
    nstart=6
    nend=18
    #Group 3
    #nstart=18
    #nend=28
    #Group1+2

    #nstart=0
    #nend=18

    #Group1+3
    #nstart=18
    #nend=28
#    indd=[0,1,2,3,4,5,18,19,20,21,22,23,24,25,26,27]
    #Group 2+3
    #nstart=6
    #nend=28


    #Group 1+2+3
#    nstart=0
#    nend=28



    nn=100
    print ('************ number of obs ',len(obs_f[0:nn]))
    for u in range(len(obs_f[nstart:nend])):
#    for u in indd: #range(len(obs_f[nstart:nend])):
        print (u, obs_f[u]) 
        LL=obs_f[u][0].split(' ')
        uvwVecobs.append(float(LL[1]))
        index_obs.append(int(LL[2]))
#    keydict = dict(zip(index_obs, uvwVecobs))
#    index_obs.sort(key=keydict.get)
#    oo=sorted(uvwVecobs)
#    uvwVecobs=oo
    #index_obs=r
    #H=np.zeros([len(uvwVecobs), len(tracer)])

    #print ('OBS DA',uvwVecobs,'--',index_obs, oo, keydict )#, len(index_obs))
    #H[np.arange(len(uvwVecobs)),index_obs]=1.0
    H=np.zeros([len(uvwVecobs), len(tracer)])

    H[range(len(uvwVecobs)),index_obs]=1.0

    #print ('H',H,np.shape(H))
    #put the background (time step 100)
    #print ('vecobs', uvwVecobs,len(uvwVecobs), 'uvwTot', uvwTot)
    nstobs = len(uvwVecobs)
    '''
    ug=vtktools.vtu('./data/raw_data/subdomains/LSBU_*_6/LSBU_300_6.vtu')
    ug.GetFieldNames()
    uvwVectot = ug.GetScalarField('TracerGeorge')
    nRec = len(uvwVectot)
    uvwVec = np.array([])
    print ('------',NindLoc)
    for i in range(NindLoc):
        indexLocal = indLoc[i]
        indexLocal = int(indexLocal)
        xMpointFl = uvwVectot[indexLocal]
        uvwVec = np.append(uvwVec,xMpointFl)
    '''
    uvwVec =uvwTot[Ba_time*len(indLoc):(Ba_time+1)*len(indLoc)]
    nst = len(uvwVec)
#    pos=ug.GetLocations()
    #print (pos,pos[0])
    #z=np.transpose(pos)[2]
    #print (pos,z)
    
    n = len(uvwVec)

    #m = trnc
    xB = uvwVec.copy()
    y = uvwVecobs.copy()
    R = 1e-3 ** 2 #lam * 0.9

    x0 = uvwVec

    Vin = np.linalg.pinv(V)
    #print (Vin, len(Vin))
    #print (x0, len(x0))
    v0 = np.dot(Vin,x0)
    VT = np.transpose(V)
    #print (np.shape(H), np.shape(xB.copy()))
    HxB = H @ xB.copy()
    d = np.subtract(y,HxB)

    # Cost function J
    def J(v):
            vT = np.transpose(v)
            vTv = np.dot(vT,v)
            Vv = np.dot(V,v)
            Jmis = np.subtract(H @ Vv,d)
            invR = 1/R
    #       invR = 1e+60
            JmisT = np.transpose(Jmis)
            RJmis = JmisT.copy()
            J1 = invR*np.dot(Jmis,RJmis)
            Jv = (vTv + J1) / 2
            return Jv

    # Gradient of J
    def gradJ(v):
            Vv = np.dot(V,v)
            Jmis = np.subtract(H @ Vv,d)
            invR = 1/R
    #       invR = 1e+60
            g1 = Jmis.copy()
            VT = np.transpose(V)
            HT = np.transpose(H)
#            g2 = np.dot(VT,g1)
            #print ('---- grad',np.shape(VT), np.shape(g1))
            g2 = np.dot(VT @ HT,g1)

            gg2 = np.multiply(invR , g2)
            ggJ = v + gg2
            return ggJ

    # Compute the minimum
    t = time.time()

    res = minimize(J, v0, method='L-BFGS-B', jac=gradJ,
                    options={'disp': True})

    #print ('res',res)
    vDA = np.array([])
    vDA = res.x
    deltaxDA = np.dot(V,vDA)
    xDA = xB + deltaxDA
    file_a = open('new_W_analysis_'+group+'.csv','w')
    for f in range(len(xDA)):
          file_a.write (str(xDA[f])+'\n') 
    file_a.close()
    print ('background',xB, 'delta xda',deltaxDA, 'xda',xDA,'obs',y) 
    C=xB[index_obs_all]
    C_da=xDA[index_obs_all]
    res_list = [xDA[i] for i in index_obs]
    print ('****** C da', C_da, '----- obs all ',obs_all, '--- C ',C)
    MSExb=LA.norm(C-obs_all, 2)/LA.norm(obs_all, 2)

    MSExDA=LA.norm(C_da-obs_all, 2) /LA.norm(obs_all, 2)
    print (' errors MSE background', MSExb, 'DA',MSExDA)
    print (' errors MAE ', np.mean(abs(C_da-obs_all)))
    plt.figure(1)
    plt.plot([0, 5], [0, 5], 'k--')

    plt.plot(obs_all, C, '*' , label='no DA', color='b')
    plt.plot(obs_all, C_da, 'o', label='with DA', color='r')
    plt.legend()
    plt.savefig('scatter_V.png')
    '''
    elapsed = time.time() - t
    #print 'elapsed' , elapsed , '\n'

    errxB = y - xB
    MSExb = LA.norm(errxB, 2)/LA.norm(y, 2)
    print ('L2 norm of the background error' , MSExb , '\n')

    errxDA = y - xDA
    MSExDA = LA.norm(errxDA, 2)/LA.norm(y, 2)
    print (y, 'L2 norm of the error in DA solution' , MSExDA , '\n')
    '''
    return xDA #MSExb, MSExDA

    # errxBtot = np.array([])
    # errxBtot = np.zeros(nRec)
    #
    # for j in range(NindLoc):
    #     indexLocal = indLoc[j]
    #     indexLocal = int(indexLocal)
    #     errxBtot[indexLocal] = errxB[j]
    #
    # abserrxBtot = np.absolute(errxBtot)
    #
    # ug.AddScalarField('u_0^M - u_C', abserrxBtot)
    # # ug.Write('./validation/abserruM400-local.vtu') # create result folder and
    #
    #
    # errxDAtot = np.array([])
    # errxDAtot = np.zeros(nRec)
    #
    # for j in range(NindLoc):
    #     indexLocal = indLoc[j]
    #     indexLocal = int(indexLocal)
    #     errxDAtot[indexLocal] = errxDA[j]
    #
    #
    # abserrxDAtot = np.absolute(errxDAtot)
    #
    # ug.AddScalarField('u^DA - u_C', abserrxDAtot)
    # # ug.Write('./validation/abserruDA400-local.vtu')
    #
    #
    # errxDAtot = np.array([])
    # errxDAtot = np.zeros(nRec)
    #
    # for j in range(NindLoc):
    #     indexLocal = indLoc[j]
    #     indexLocal = int(indexLocal)
    #     errxDAtot[indexLocal] = errxDA[j]
    #
    #
    # abserrxDAtot = np.absolute(errxDAtot)
    #
    # ug.AddScalarField('u^DA - u_C', abserrxDAtot)
    # # ug.Write('./validation/abserruDA400-local.vtu')
    #
    #
    # xDAtot = uvwVectot.copy()
    #
    # for j in range(NindLoc):
    #     indexLocal = indLoc[j]
    #     indexLocal = int(indexLocal)
    #     xDAtot[indexLocal] = xDA[j]
    #
    # ug.AddScalarField('uDA', xDAtot)
    # # ug.Write('./validation/uDA400-local.vtu')
    #
    #
    # xBtot = uvwVectot.copy()
    #
    # # ug.AddScalarField('uM', xBtot)
    # # ug.Write('/home') # result folder

da()
# background_error = 0
# DA_error = 0
#
# for i in range(100):
#     print(i)
#     indLoc = np.random.randint(pos.shape[0], size=(4))
#     be, dae = da()
#     background_error += be
#     DA_error += dae
#
# print("Background Error Mean: ", background_error/100)
# print("DA Error Mean: ", DA_error/100)


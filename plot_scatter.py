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



obs_all=[3.1129902712240543, 2.832448823834394, 2.8062263017304376, 2.839194028387001, 5.3091791328731786, 2.9049841153956835, 0.4214136221770839, 2.8260570691695697, 0.49827781480837574, 2.773981191659426, 2.817816182893591, 2.8374836612444905, 2.9273559133178395, 2.8316626914062786, 2.844207462581222, 2.827801778346113, 2.8275118695386516, 2.827801778346113, 1.015187441685032, 3.159502030203369, 2.8424542469114096, 3.389275286644772, 2.3575457638238966, 2.958489352135816, 2.8414292778438393, 2.8253181214717316, 2.839672008300254, 2.827762313037715]



C_1 = [1.68604845, 2.83460369, 2.91650502, 0.88799687, 4.79711507 ,2.94332181, 0.50487434 ,2.74661453 ,0.54646505 ,2.60476085 ,2.79738367 ,2.83964845, 2.99276356 ,2.83225913 ,2.85481942 ,2.82258348 ,2.8264625  ,2.82258348, 1.06774754 ,3.21542219 ,2.86036554 ,2.54535926 ,2.11900993 ,2.96717107, 2.86779056 ,2.82391003 ,2.85407948 ,2.82618011]


#Group 1
C_da_1 = [3.11299015 , 2.83244889,  2.80622644 , 2.83919387,  5.30917905  ,2.90498417,-0.0444287  , 3.87261112 ,-0.17445584 , 1.6025726 ,  1.89029587  ,3.80099445,  3.42672542 , 3.45601033 , 3.06073541 , 2.92167324,  3.38522312 , 2.92167324, -0.1423125  , 1.69403522 , 4.20318269 , 2.06558248,  1.64261876 , 3.45430352,  2.21104787 , 2.500579   , 3.16474504 , 2.80203584 ]




#Group 2
C_da_2 = [ 3.11299014 , 2.83244933 , 2.80622647  ,2.83919346,  5.30917878,  2.9049843,  0.42141361 , 2.82605722 , 0.4982778  , 2.77398114  ,2.81781589,  2.83748416,  2.82547284 , 2.55593495 , 2.66709013 , 2.13608453  ,4.25965266,  2.13608453, -1.95361226 , 2.08758233 , 4.2251883  , 2.83187391  ,1.86751758,  3.44146095,  3.34071793 , 1.73112169 , 2.59354536 , 2.62603627]



#Group 3
C_da_3 = [3.11299007 , 2.83244889 , 2.80622651,  2.83919383 , 5.309179 ,   2.90498416,  0.42141358 , 2.8260571   ,0.4982778  , 2.77398109,  1.45074694 , 3.65802217,  3.77850674 , 3.83427964  ,2.19892393 , 3.03667969,  3.66209341 , 3.03667969, -0.07139977 , 2.09905626  ,4.23462577 , 1.75347582,  1.93932624 , 2.74732303,  1.96597355 , 2.16239018  ,3.48846509 , 2.94887402 ]



#Group 1+2
C_da_12 = [ 3.11298064 , 2.83244251,  2.80622686,  2.83919139,  5.30917305 , 2.90499111,  0.4214156 ,  2.82606357 , 0.49827833,  2.77398393,  2.81781377 , 2.83749568,  2.92736771,  2.83165023 , 2.84420788,  2.82780201,  2.8275251  , 2.82780201, 10.87497066, -0.44323027 , 0.88068413, 12.5841738 ,  2.92217962 , 5.3890562,  4.03560287,  3.29430371 , 3.20802224,  1.92596566]



#Group 1+3
C_da_13 = [ 3.11298907 ,2.83244942 ,2.80622699, 2.83919318 ,5.3091784 , 2.90498417, 5.39130466, 2.35661516 ,0.05927457 ,2.54903377, 2.86169443 ,3.3557489, 3.08702285, 3.03743965 ,2.74290569 ,4.16620146, 3.27672389 ,4.16620146, 1.01518727, 3.1595018  ,2.84245517 ,3.38927542, 2.35754559 ,2.95849033, 2.84142815, 2.82531775 ,2.83967227 ,2.82776232]



#Group 2+3
C_da_23 = [ 2.86295327 , 2.62929183,  2.95232833,  2.83920737 , 5.16248887,  3.11817921 , 0.46493771 , 2.9110397  , 0.45898807 , 2.66293162,  2.76486305 , 2.83426559, 2.99855669 , 2.91165847 , 2.82489962 , 2.83058014,  3.30650708 , 2.83058014, 1.03765443 , 3.15948649 , 2.75729244 , 3.39780425, -2.41210964 , 3.10820542 , 3.31222662 , 2.64616808 , 4.40028865 , 3.0994802 ]




#Group 1+2+3
C_da_123 = [2.6926422 , 2.955215 ,  2.8611614,  2.58596712, 4.84094528, 2.97562294, 0.63010647 ,3.03402689 ,0.47715597 ,2.6326272 , 2.72148576, 3.05398129, 3.00841986 ,2.93875938 ,2.42670891 ,2.78896127, 3.31757687, 2.78896127, 0.95776172 ,3.15949629 ,2.95349914 ,3.53444236, 2.21678864, 3.2934908, 2.75343427 ,2.76713808 ,2.93734678 ,2.72481623 ]


MSExB=LA.norm(np.array(C_1)-np.array(obs_all), 2)/LA.norm(obs_all, 2)

MSExB_1=LA.norm(np.array(C_da_1)-np.array(obs_all), 2)/LA.norm(obs_all, 2)
MSExB_2=LA.norm(np.array(C_da_2)-np.array(obs_all), 2)/LA.norm(obs_all, 2)
MSExB_3=LA.norm(np.array(C_da_3)-np.array(obs_all), 2)/LA.norm(obs_all, 2)
MSExB_12=LA.norm(np.array(C_da_12)-np.array(obs_all), 2)/LA.norm(obs_all, 2)
MSExB_13=LA.norm(np.array(C_da_13)-np.array(obs_all), 2)/LA.norm(obs_all, 2)
MSExB_23=LA.norm(np.array(C_da_23)-np.array(obs_all), 2)/LA.norm(obs_all, 2)
MSExB_123=LA.norm(np.array(C_da_123)-np.array(obs_all), 2)/LA.norm(obs_all, 2)


print (MSExB, MSExB_1, MSExB_2, MSExB_3, MSExB_12, MSExB_13, MSExB_23, MSExB_123)




plt.figure(1)
plt.plot([0, 6], [0, 6], 'k--')

plt.plot(obs_all, C_1, '*' , label='RANS_run$_{t_{a}}$', color='b')
plt.plot(obs_all, [abs(ele) for ele in C_da_1], 'o', label='RANS_run$^{DA}_{t_{a}}$', color='r')
plt.xlabel('Observed wind speed $m.s^{-1}$', fontsize=13)
plt.ylabel('Simulated wind speed $m.s^{-1}$', fontsize=13)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend( fontsize=13)
plt.savefig('scatter_group_1.png')





plt.figure(2)
plt.plot([0, 6], [0, 6], 'k--')

plt.plot(obs_all, C_1, '*' , label='RANS_run$_{t_{a}}$', color='b')
plt.plot(obs_all, [abs(ele) for ele in C_da_2], 'o', label='RANS_run$^{DA}_{t_{a}}$', color='r')
plt.xlabel('Observed wind speed $m.s^{-1}$', fontsize=13)
plt.ylabel('Simulated wind speed $m.s^{-1}$', fontsize=13)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend( fontsize=13)
plt.savefig('scatter_group_2.png')




plt.figure(3)
plt.plot([0, 6], [0, 6], 'k--')

plt.plot(obs_all, C_1, '*' , label='RANS_run$_{t_{a}}$', color='b')
plt.plot(obs_all, [abs(ele) for ele in C_da_3], 'o', label='RANS_run$^{DA}_{t_{a}}$', color='r')
plt.xlabel('Observed wind speed $m.s^{-1}$', fontsize=13)
plt.ylabel('Simulated wind speed $m.s^{-1}$', fontsize=13)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend( fontsize=13)
plt.savefig('scatter_group_3.png')





plt.figure(4)
plt.plot([0, 6], [0, 6], 'k--')

plt.plot(obs_all, C_1, '*' , label='RANS_run$_{t_{a}}$', color='b')
plt.plot(obs_all, [abs(ele) for ele in C_da_12], 'o', label='RANS_run$^{DA}_{t_{a}}$', color='r')
plt.xlabel('Observed wind speed $m.s^{-1}$', fontsize=13)
plt.ylabel('Simulated wind speed $m.s^{-1}$', fontsize=13)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend( fontsize=13)
plt.savefig('scatter_group_12.png')






plt.figure(5)
plt.plot([0, 6], [0, 6], 'k--')

plt.plot(obs_all, C_1, '*' , label='RANS_run$_{t_{a}}$', color='b')
plt.plot(obs_all, [abs(ele) for ele in C_da_13], 'o', label='RANS_run$^{DA}_{t_{a}}$', color='r')
plt.xlabel('Observed wind speed $m.s^{-1}$', fontsize=13)
plt.ylabel('Simulated wind speed $m.s^{-1}$', fontsize=13)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend( fontsize=13)
plt.savefig('scatter_group_13.png')




plt.figure(6)
plt.plot([0, 6], [0, 6], 'k--')

plt.plot(obs_all, C_1, '*' , label='RANS_run$_{t_{a}}$', color='b')
plt.plot(obs_all, [abs(ele) for ele in C_da_23], 'o', label='RANS_run$^{DA}_{t_{a}}$', color='r')
plt.xlabel('Observed wind speed $m.s^{-1}$', fontsize=13)
plt.ylabel('Simulated wind speed $m.s^{-1}$', fontsize=13)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend( fontsize=13)
plt.savefig('scatter_group_23.png')




plt.figure(7)
plt.plot([0, 6], [0, 6], 'k--')

plt.plot(obs_all, C_1, '*' , label='RANS_run$_{t_{a}}$', color='b')
plt.plot(obs_all, [abs(ele) for ele in C_da_123], 'o', label='RANS_run$^{DA}_{t_{a}}$', color='r')
plt.xlabel('Observed wind speed $m.s^{-1}$', fontsize=13)
plt.ylabel('Simulated wind speed $m.s^{-1}$', fontsize=13)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend( fontsize=13)
plt.savefig('scatter_group_123.png')







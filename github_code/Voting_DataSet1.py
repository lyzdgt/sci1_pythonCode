import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import keras
from keras.layers import Dropout
import numpy as np
import time
import argparse
import json
from sklearn import metrics
from math import sqrt, ceil
from keras . layers . recurrent import LSTM
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import datetime
start = datetime.datetime.now()
n_in=7
pc1=[ ]

pc2=[]

pc3=[]


tcr1=[ ]

tcr2=[]

tcr3=[]

thq1=[]

thq2=[]

thq3=[]

#1RNN2GRU3LSTM
f1=np.array(
)
f1_tcr=np.array(
)
f1_thq=np.array(
)



    
list_1=[0.084155605,0.089870344,0.171766021]
list_2=[0.227279693,0.137551006,0.31944929]
list_3=[0.169323524,0.606899942,0.358251874]
list_4=[0.346319826,0.197022136,0.619699176]
list_s=['Pc','TCR','THQ']
df1= pd.DataFrame(np.zeros((n_in,len(list_s))),columns=list_s)
for i in range(n_in):
    df1.loc[i,'Pc']=f1[i]
    df1.loc[i,'TCR']=f1_tcr[i]
    df1.loc[i,'THQ']=f1_thq[i]

for i in range(n_in):
    d1=np.sqrt((df1['Pc'][i:i+1]-list_1[0])**2+(df1['TCR'][i:i+1]-list_1[1])**2+(df1['THQ'][i:i+1]-list_1[2])**2)
    d2=np.sqrt((df1['Pc'][i:i+1]-list_2[0])**2+(df1['TCR'][i:i+1]-list_2[1])**2+(df1['THQ'][i:i+1]-list_2[2])**2)
    d3=np.sqrt((df1['Pc'][i:i+1]-list_3[0])**2+(df1['TCR'][i:i+1]-list_3[1])**2+(df1['THQ'][i:i+1]-list_3[2])**2)
    d4=np.sqrt((df1['Pc'][i:i+1]-list_4[0])**2+(df1['TCR'][i:i+1]-list_4[1])**2+(df1['THQ'][i:i+1]-list_4[2])**2)
    d=[d1.values,d2.values,d3.values,d4.values]
    d_min=np.min(d)
    if d_min==d1.values:
        df1.loc[i,'level']=0
    if d_min==d2.values:
        df1.loc[i,'level']=1
    if d_min==d3.values:
        df1.loc[i,'level']=2
    if d_min==d4.values:
        df1.loc[i,'level']=3
l=[]
v=[]
v1=[]
for a in range(0,99):
    for b in range(0,99):
        for c in range(0,99):
            if a+b+c==100:
                list_s=['Pc','TCR','THQ']
                df0= pd.DataFrame(np.zeros((n_in,len(list_s))),columns=list_s)
                for i in range(n_in):
                    df0.loc[i,'Pc']=(a/100)*pc1[i][0]+(b/100)*pc2[i][0]+(c/100)*pc3[i][0]
                    df0.loc[i,'TCR']=(a/100)*tcr1[i][0]+(b/100)*tcr2[i][0]+(c/100)*tcr3[i][0]
                    df0.loc[i,'THQ']=(a/100)*thq1[i][0]+(b/100)*thq2[i][0]+(c/100)*thq3[i][0]
                    d1=np.sqrt((df0.loc[i,'Pc']-list_1[0])**2+(df0.loc[i,'TCR']-list_1[1])**2+(df0.loc[i,'THQ']-list_1[2])**2)
                    d2=np.sqrt((df0.loc[i,'Pc']-list_2[0])**2+(df0.loc[i,'TCR']-list_2[1])**2+(df0.loc[i,'THQ']-list_2[2])**2)
                    d3=np.sqrt((df0.loc[i,'Pc']-list_3[0])**2+(df0.loc[i,'TCR']-list_3[1])**2+(df0.loc[i,'THQ']-list_3[2])**2)
                    d4=np.sqrt((df0.loc[i,'Pc']-list_4[0])**2+(df0.loc[i,'TCR']-list_4[1])**2+(df0.loc[i,'THQ']-list_4[2])**2)
                    d=[d1,d2,d3,d4]
                    d_min=np.min(d)
                    if d_min==d1:
                        df0.loc[i,'level']=0
                    if d_min==d2:
                        df0.loc[i,'level']=1
                    if d_min==d3:
                        df0.loc[i,'level']=2
                    if d_min==d4:
                        df0.loc[i,'level']=3
                    df0.loc[i,'accuray']=df1.loc[i,'level']-df0.loc[i,'level']

                if n_in > 34:
                    _sum35 = 0
                    for i in range(35):
                        df0.loc[i, 'accuray'] = df1.loc[i, 'level'] - df0.loc[i, 'level']
                        if df0.loc[i, 'accuray'] == 0:
                            _sum35 = _sum35 + 1
                    print('三十五天准确率：', _sum35 / 35)

                if n_in > 41:
                    _sum42 = 0
                    for i in range(42):
                        df0.loc[i, 'accuray'] = df1.loc[i, 'level'] - df0.loc[i, 'level']
                        if df0.loc[i, 'accuray'] == 0:
                            _sum42 = _sum42 + 1
                    print('四十二天', _sum42 / 42)

                if n_in > 27:
                    _sum28 = 0
                    for i in range(28):
                        df0.loc[i, 'accuray'] = df1.loc[i, 'level'] - df0.loc[i, 'level']
                        if df0.loc[i, 'accuray'] == 0:
                            _sum28 = _sum28 + 1
                    print('28天准确率：', _sum28 / 28)

                if n_in > 20:
                    _sum21 = 0
                    for i in range(21):
                        df0.loc[i, 'accuray'] = df1.loc[i, 'level'] - df0.loc[i, 'level']
                        if df0.loc[i, 'accuray'] == 0:
                            _sum21 = _sum21 + 1
                    print('二十一天', _sum21 / 21)

                if n_in > 13:
                    _sum14 = 0
                    for i in range(14):
                        df0.loc[i, 'accuray'] = df1.loc[i, 'level'] - df0.loc[i, 'level']
                        if df0.loc[i, 'accuray'] == 0:
                            _sum14 = _sum14 + 1
                    print('14天', _sum14 / 14)

                if n_in > 6:
                    _sum7 = 0
                    for i in range(7):
                        df0.loc[i, 'accuray'] = df1.loc[i, 'level'] - df0.loc[i, 'level']
                        if df0.loc[i, 'accuray'] == 0:
                            _sum7 = _sum7 + 1
                    print('七天准确率：', _sum7 / 7)


                l.append((_sum7/7,_sum14/14,_sum21/21))
                print((_sum7/7,_sum14/14,_sum21/21))
                v1.append(_sum7/7+_sum14/14+_sum21/21)
                print(_sum7/7+_sum14/14+_sum21/21)
                v.append((a,b,c))
                print(a,b,c)
    if a==99:
        break
print("outcome:",l[v1.index(max(v1))])
print("RNN:",v[v1.index(max(v1))][0],"GRU:",v[v1.index(max(v1))][1],"LSTM:",v[v1.index(max(v1))][2])
end = datetime.datetime.now()
cost=end - start
print('totally time is ',cost)
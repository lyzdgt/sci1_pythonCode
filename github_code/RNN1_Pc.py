import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras . layers . recurrent import GRU
import keras
from keras.layers import Dropout
from xgboost import XGBRegressor as XGBR

#导入数据
test='D:/pycharm项目/PyCharmProject/食品安全项目/Pre_data_nor.xlsx'
test = pd.read_excel(test, index_col=0, parse_dates=[0])
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
from keras.layers import SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import warnings
import datetime
start = datetime.datetime.now()
def Metrics_M(real1,pre1):
    y = np.array(real1)
    y_hat = np.array(pre1)
    MSE = metrics.mean_squared_error(y, y_hat)
    RMSE = metrics.mean_squared_error(y, y_hat)**0.5
    MAE = metrics.mean_absolute_error(y, y_hat)
    n = len(real1)
    return RMSE,MAE
#################################
index="Pc"
n_in=21
n_out =1
#################################
t_pc=test[index]
t_real=test[index]
t_real1=test[index]
t_real2=test[index]
X = t_real.values
X_tcr=t_real1.values
X_thq=t_real2.values
dataset=pd.DataFrame(X)
dataset_tcr=pd.DataFrame(X_tcr)
dataset_thq=pd.DataFrame(X_thq)
eader_row_index =0
index_col_name =0
col_to_predict =0
cols_to_drop =None
if index_col_name:
    dataset.set_index(index_col_name, inplace=True)    
if cols_to_drop:
    dataset.drop(cols_to_drop, axis =1, inplace = True)    
col_names = dataset.columns.values.tolist()
values = dataset.values   
# move the column to predict to be the first col: 把预测列调至第一列
col_to_predict_index = col_to_predict if type(col_to_predict) == int else col_names.index(col_to_predict)
output_col_name = col_names[col_to_predict_index]
if col_to_predict_index > 0:
    col_names = [col_names[col_to_predict_index]] + col_names[:col_to_predict_index] + col_names[col_to_predict_index+1:]
values = np.concatenate((values[:, col_to_predict_index].reshape((values.shape[0], 1)), values[:,:col_to_predict_index], values[:,col_to_predict_index+1:]), axis=1)
values = values.astype("float32")
col_names, values,n_features, output_col_name=col_names,values,values.shape[1], output_col_name
verbose = 2
dropnan = True
n_vars = 1 if type(values) is list else values.shape[1]
if col_names is None: col_names = ["var%d" % (j+1) for j in range(n_vars)]
df = DataFrame(values)
cols, names = list(), list()
# input sequence (t-n, ... t-1)
for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [("%s(t-%d)" % (col_names[j], i)) for j in range(n_vars)]
# forecast sequence (t, t+1, ... t+n)
for i in range(0, n_out):
    cols.append(df.shift(-i))         #这里循环结束后cols是个列表，每个列表都是一个shift过的矩阵
    if i == 0:
        names += [("%s(t)" % (col_names[j])) for j in range(n_vars)]
    else:
        names += [("%s(t+%d)" % (col_names[j], i)) for j in range(n_vars)]
# put it all together
agg = concat(cols, axis=1)    #将cols中的每一行元素一字排开，连接起来，vala t-n_in, valb t-n_in ... valta t, valb t... vala t+n_out-1, valb t+n_out-1
agg.columns = names
# drop rows with NaN values
if dropnan:
    agg.dropna(inplace=True)
if verbose:
    print("\nsupervised data shape:", agg.shape)
values=agg.values

train_X=values[:values.shape[0]-2*n_in,:n_in]
test_X=values[values.shape[0]-2*n_in:values.shape[0]-n_in,:n_in]
train_Y=values[:values.shape[0]-2*n_in,n_in:]
test_Y=values[values.shape[0]-2*n_in:values.shape[0]-n_in,n_in:]

model = Sequential()
model .add(SimpleRNN (units =128, activation = 'relu'
            ,return_sequences=True
            ,input_shape = (n_in,1)))
model .add(SimpleRNN (units =128, activation = 'tanh'
            ,return_sequences=True
            ,input_shape = (n_in,1)))
model .add(SimpleRNN (units=n_out, activation = 'linear'
            ,input_shape = (n_in,1)))
adam = keras.optimizers.Adam(lr = 0.0005, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
model.compile(loss='mae', optimizer=adam)
model.fit(train_X.reshape(train_X.shape[0],train_X.shape[1],1),train_Y.reshape(train_Y.shape[0],1),epochs=500,shuffle=True,batch_size=7)
pre1=[]
pre0=model.predict(train_X.reshape(train_X.shape[0],train_X.shape[1],1))
txt1=test_X[0]
for i in range(len(test_X)):
    txt1=txt1.reshape(i+1,n_in,1)
    list_xgt=model.predict(txt1[i].reshape(1,n_in,1))
    pre1.append(list_xgt)
    list_xgtt=np.concatenate([txt1[i,1:n_in],list_xgt])
    txt1=np.concatenate([txt1.reshape(i+1,n_in,1),list_xgtt.reshape((1,n_in,1))])
pre1=np.array(pre1).reshape(len(pre1),1)
plt.subplot(2,1,1)
plt.plot(pre1)
plt.plot(test_Y)
plt.subplot(2,1,2)
plt.plot(pre0)
plt.plot(train_Y)

#输出
print("预测：")
list=np.array(pre1)
print("[",list[0])
for i in range(1,len(list)-1):
    print(list[i],',')
print(list[i+1],']')


print("实际：")
list=np.array(test_Y)
print("[",list[0])
for i in range(1,len(list)-1):
    print(list[i],',')
print(list[i+1],']')

print("训练预测：")
list=np.array(pre0)
print("[",list[0])
for i in range(1,len(list)-1):
    print(list[i],',')
print(list[i+1],']')

print("训练实际：",)
list=np.array(train_Y)
print("[",list[0])
for i in range(1,len(list)-1):
    print(list[i],',')
print(list[i+1],']')


RMSE,MAE=Metrics_M(test_Y,pre1)
print("RMSE:",RMSE)
print("MAE:",MAE)

end = datetime.datetime.now()
cost=end - start
print('totally time is ',cost)

plt.show()

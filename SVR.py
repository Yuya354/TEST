# -*- coding: utf-8 -*-
"""
@author: Yuya
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVR 
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

dataname = input("教師データを入力してください(csv形式)：")
dataset = pd.read_csv(dataname)
dataset1 = dataset.T[0:250]
datanum =input("使用する教師データ数を入力してください。")
X = dataset1.T[0:int(datanum)]
select_data = input("波形分離を行うデータ番号を入力してください。")

temp = []

for i in range(0,9):
    y1 = dataset.T[250+i:251+i]
    y2 = np.ravel(y1)
    y = y2.T[0:int(datanum)]
    
    #特微量と正解を訓練データとテストデータに分割
    X_train = X
    y_train = y
    X_test1 = dataset1[int(select_data)].to_numpy()
    X_test2 = X_test1.reshape(-1,1)
    X_test = X_test2.T
    y_test = y2[int(select_data)].reshape(-1,1)

    #特微量の標準化
    sc = StandardScaler()
    #訓練データを変換器で標準化
    X_train_std = sc.fit_transform(X_train)
    #テストデータを作成した変換器で標準化
    X_test_std = sc.transform(X_test)

    #グリッドサーチの実行
    param_grid =[ {'kernel':['rbf'],'C': [0.1,0.3,0.5,1.0,3.0,10.,30.,100.,300.,1000.0],
                   'gamma':[0.01,0.1,0.3,0.8,1.0],
                   'epsilon':[0.008,0.01,0.02,0.03,0.1,0.3,1.0,3.0] } ] 
    
    model = SVR()
    
    grid_search = GridSearchCV(model,param_grid,cv=5,scoring='neg_mean_squared_error',
                               verbose=2,refit=True)
    
    grid_search.fit(X_train_std,y_train)
    
    print(grid_search.best_estimator_)
    print(grid_search.best_params_)
    
    #グリッドサーチの最良モデルで予測
    model_grs = grid_search.best_estimator_
    
    y_train_grs_pred = model_grs.predict(X_train_std)
    y_test_grs_pred = model_grs.predict(X_test_std)
    
    #MSEの計算
    print('MSE train: %.7f, test: %.7f' % (
        mean_squared_error(y_train, y_train_grs_pred),
        mean_squared_error(y_test, y_test_grs_pred)))

    temp.append(mean_squared_error(y_train, y_train_grs_pred))
    temp.append(mean_squared_error(y_test, y_test_grs_pred))
    temp.append(y_test_grs_pred[0])
    
    #残差プロット
    plt.figure(figsize=(8,4)) #plotsizeの指定
    
    plt.scatter(y_train_grs_pred, y_train_grs_pred - y_train, c='red',
                marker='o', edgecolor='white', label='Training data')
    
    plt.scatter(y_test_grs_pred, y_test_grs_pred - y_test, c='blue',
                marker='s', edgecolor='white', label='Test data')
    
    plt.xlabel('Predicted values')
    plt.ylabel('residuals')
    plt.legend(loc='upper left')
    #plt.hlines(y=0, xmin=99.2, xmax=99.3, color='black',lw=2)
    #plt.xlim([99.20,99.30])
    plt.tight_layout()
    plt.show()
    
# ガウス関数を定義
x1 = np.arange(95.5,108.0,0.05)

def gauss(x1, amp = 1, loc = 0, sigma = 1):
    return amp * np.exp(-(x1 - loc)**2 / (2*sigma**2))

# フォントの種類とサイズを設定する。
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'
 
# 目盛を内側にする。
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
 
fig = plt.figure()
ax1 = plt.subplot(111)
 
# グラフの上下左右に目盛線を付ける。
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')
 
# 軸のラベルを設定する。
ax1.set_ylabel('Intensity[a.u]')
ax1.set_xlabel('Band Energy[eV]') 

# データの範囲と刻み目盛を明示する。
ax1.set_xticks(np.arange(94, 108, 1))
ax1.set_xlim(93, 109)
ax1.invert_xaxis()

#波形分離したいスペクトル/y_test3
y_test1 = dataset1[int(select_data)]
y_test2 = (y_test1).T
y_test3 = y_test2[0:250]
y_test4 = y_test3.to_numpy()
plt.plot(x1,y_test4,label='PDF', lw=1,color="black")

#予測した9つのパラメータから作成したスペクトル。
#平均値(loc),標準偏差(scale)
y_1 = gauss(x1,amp = temp[20], loc = temp[2], sigma = temp[11]/2.354820)

y_2 = gauss(x1,amp = temp[23], loc = temp[5], sigma = temp[14]/2.354820)

y_3 = gauss(x1,amp = temp[26], loc = temp[8], sigma = temp[17]/2.354820)

y_4 = y_1 + y_2 + y_3

plt.plot(x1,y_4,label='PDF', lw=1, color="red")

print('Data',datanum,': %.5f' % (
    mean_squared_error(y_test4,y_4 )))

plt.show()
    
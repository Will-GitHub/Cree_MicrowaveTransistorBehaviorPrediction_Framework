import copy as cp
import sys
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#import xlrd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

 #evaluate fit score
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#dnn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import History,TensorBoard,EarlyStopping,ModelCheckpoint

from sklearn.utils import shuffle
from sklearn.utils.validation import check_array
import threading
import time
from time import ctime,sleep
import platform

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

from matplotlib import rcParams
from smithplot import SmithAxes

filenames=['../../splitted_data/D9090_split/pe_0.csv',
           '../../splitted_data/D9090_split/pe_1.csv','../../splitted_data/D9090_split/pe_2.csv',
           '../../splitted_data/D9090_split/pe_3.csv','../../splitted_data/D9090_split/pe_4.csv',
           '../../splitted_data/D9090_split/pe_5.csv','../../splitted_data/D9090_split/pe_6.csv',
           '../../splitted_data/D9090_split/pe_7.csv','../../splitted_data/D9090_split/pe_8.csv',
           '../../splitted_data/D9090_split/pe_9.csv','../../splitted_data/D9090_split/pe_10.csv',
           '../../splitted_data/D9090_split/pe_11.csv','../../splitted_data/D9090_split/pe_12.csv',
           '../../splitted_data/D9090_split/pe_13.csv','../../splitted_data/D9090_split/pe_14.csv',
           '../../splitted_data/D9090_split/pe_15.csv','../../splitted_data/D9090_split/pe_16.csv',
           '../../splitted_data/D9090_split/pe_17.csv']


class Profiling_Model:
    '''Class Name: Profiling_Model

    Synopsis
    import loo_dnn_memory
    def run_it();

    Description
    This is the main class of module loo_dnn_memory. This class contains everything we need to run this model.
    You can find an example to run this model in run_it() function defined outside this class.
    '''
    def __init__(self,tmp=0):
        '''Init parameters and load data.
        '''
        self.init_params(tmp=tmp)
        self.load_data()
        #self.train_model()
        #print('done')
        
    def init_params(self,tmp):
        '''Init parameters. Include:
        layer_points - points number per layer
        tmp_layer - the goal layer number you wish to predict
        absent_layer - the goal layer number in Leave_One_Out experiments
        mse_array - an array to store mse values
        r2_array - an array to store r2 values
        '''
        self.layer_points=9090
        self.tmp_layer=tmp
        self.absent_layer=tmp
        self.mse_array = np.zeros((1,9))
        self.r2_array = np.zeros((1,9))

    def load_data(self):
        '''Load data from '../../splitted_data/D9090_split/pe_18f.csv'
        '''
        #self.raw_df = pd.read_csv(filenames[self.tmp_layer])
        self.raw_df = pd.read_csv('../../splitted_data/D9090_split/pe_18f.csv')
        #print(self.tmp_layer,'th Layer.')
        #,names = ['2nd_Real','2nd_Imaginary','Real','Imaginary','P_e'])
        a = self.absent_layer * self.layer_points
        b = (self.absent_layer+1)*self.layer_points
        tmp = self.raw_df.iloc[a:b,:]
        self.train_set = self.raw_df[-self.raw_df.isin(tmp)].dropna().reset_index(drop=True)
        self.test_set = tmp.reset_index(drop=True)

    def mean_absolute_percentage_error(self,y_true, y_pred): 
        '''
        Parameters:
        y_true - real values as a matrix
        y_pred - predictced values as a matrix

        Return:
        MAPE(Mean Absolute Percentage Error)
        '''
        #y_true, y_pred = check_array(y_true, y_pred)

        ## Note: does not handle mix 1d representation
        #if _is_1d(y_true): 
        #    y_true, y_pred = _check_1d_array(y_true, y_pred)

        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def train_evaluate(self,y_true,y_pred):
        '''Accuracy Metrics for training model.
        '''
        self.train_r2 = metrics.r2_score(y_true,y_pred)
        self.train_mape = self.mean_absolute_percentage_error(y_true,y_pred)
        self.train_mae = metrics.mean_absolute_error(y_true,y_pred)
        self.train_mse = metrics.mean_squared_error(y_true,y_pred)
        self.train_rmse = metrics.mean_squared_error(y_true,y_pred,squared=False)
        
    def test_evaluate(self,y_true,y_pred):
        '''Accuracy Metrics for testing model
        '''
        self.test_r2 = metrics.r2_score(y_true,y_pred)
        self.test_mape = self.mean_absolute_percentage_error(y_true,y_pred)
        self.test_mae = metrics.mean_absolute_error(y_true,y_pred)
        self.test_mse = metrics.mean_squared_error(y_true,y_pred)
        self.test_rmse = metrics.mean_squared_error(y_true,y_pred,squared=False)

    def train_model(self,random_state=1,frac=1):
        '''Start training the model.
        Parameter:
        random_state - random seed.
        frac - the fraction of training set
        '''
        #estimator = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,
        #                         random_state=random_state)
        #estimator = XGBRegressor(n_estimators=n_estimators,max_depth=max_depth,random_state=random_state,objective='reg:squarederror')
        #estimator = SVR(C=C,epsilon=epsilon,gamma=gamma)

        tmp_df=self.train_set.sample(frac=frac,random_state=random_state)
        X, y = tmp_df[['2nd_Real','2nd_Imaginary','Real','Imaginary']],tmp_df['P_e']

        temp=self.test_set
        test_X, test_y = temp[['2nd_Real','2nd_Imaginary','Real','Imaginary']],temp['P_e']
        history = tf.keras.callbacks.History()
        model = Sequential()
        model.add(Dense(256,input_dim=X.shape[1],activation="relu"))
        #model.add(Dropout(rate=0.5))
        model.add(Dense(256,activation = "relu"))
        #model.add(Dropout(rate=0.5))
        model.add(Dense(256,activation = "relu"))
        #model.add(Dropout(rate=0.5))
        model.add(Dense(128,activation = "relu"))
        #model.add(Dropout(rate=0.5))
        model.add(Dense(64,activation = "relu"))
        model.add(Dense(1,activation = "linear"))
        model.compile(optimizer='adam',loss="mean_absolute_error",metrics=['mae','mse'])
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_mse',patience=5,mode='min',restore_best_weights=True)
        model.fit(X.values,y.values,validation_data=(test_X.values,test_y.values),epochs=30,batch_size=128,callbacks=[history,es_callback])
        #estimator.fit(X,y.values.ravel())
        self.estimator = model
        self.pred2 = model.predict(test_X)
        #print(self.pred2.shape)
        self.test_y=test_y
        self.test_X=test_X
        #print(self.test_y.shape)

    def test_model(self):
        '''Start testing the model.
        No parameters needed. Just run train_model() before test_model().
        '''
        #pred1 = estimator.predict(X)
        #self.train_evaluate(y.values.ravel(),pred1)
        #temp=self.raw_df
        #test_X, test_y = temp[['2nd_Real','2nd_Imaginary','Real','Imaginary']],temp['P_e']
        #pred2 = self.estimator.predict(test_X)
        self.test_evaluate(self.test_y.values.ravel(),self.pred2)
        #self.xx=test_X
        #self.y1=test_y.values.ravel()
        #self.y2=pred2
        self.Smith_2lines(xx=np.array(self.test_X.iloc[:,2:4]),y1=np.array(self.test_y.values.ravel()),y2=np.array(self.pred2.ravel()),name0=self.tmp_layer+1)

    def Smith_2lines(self,xx=[0.0,0],y1=[0.0,0],y2=[0.0,0],name0=0,name1=0,name2=0):
        '''Draw two lines in a 2d Smith Chart.
        xx - values of X coordination
        y1 - values of Y coordination for original line
        y2 - values of Y coordination for predicted line
        '''
        xx1=xx
        xx2=xx
        yy1=y1
        yy2=y2
        
        ax = plt.subplot(1, 1, 1, projection='smith')
        plt.delaxes(ax)
        
        ax = plt.subplot(1, 1, 1, projection='smith')
        rcParams['figure.figsize'] = 12, 12
        rcParams['axes.linewidth'] = 1
        rcParams['font.size'] = 15
    #    X_test1=xx1
    #    y_pred1=yy1
    #    nested_pred1 = np.c_[X_test1,y_pred1.T] #add y_pred as a new column to X_test
    #    R1_n = np.unique(X_test1[:, 0]) / 50 #extract R1 from X_test in Z format
    #    I1_n = np.unique(X_test1[:, 1]) / 50
    #    board_1 = np.full([len(I1_n),len(R1_n)], np.nan) #generate empty board to match contour() format
        
        X_test1=xx1
        y_pred1=yy1
        nested_pred1 = np.c_[X_test1,y_pred1.T] #add y_pred as a new column to X_test
        R1_n = np.unique(X_test1[:, 0]) / 50 #extract R1 from X_test in Z format
        I1_n = np.unique(X_test1[:, 1]) / 50
        board_1 = np.full([len(I1_n),len(R1_n)], np.nan) #generate empty board to match contour() format
        pred_pan1 = pd.DataFrame(board_1, index = I1_n , columns = R1_n)
        
        X_test2=xx2
        y_pred2=yy2
        nested_pred2 = np.c_[X_test2,y_pred2.T] #add y_pred as a new column to X_test
        R2_n = np.unique(X_test2[:, 0]) / 50 #extract R1 from X_test in Z format
        I2_n = np.unique(X_test2[:, 1]) / 50
        board_2 = np.full([len(I2_n),len(R2_n)], np.nan) #generate empty board to match contour() format
        pred_pan2 = pd.DataFrame(board_2, index = I2_n , columns = R2_n)
        
        for i in range(len(nested_pred1)): #fill the values
            pred_pan1[nested_pred1[i, 0] / 50][nested_pred1[i, 1] / 50] = nested_pred1[i, 2]
        for i in range(len(nested_pred2)): #fill the values
            pred_pan2[nested_pred2[i, 0] / 50][nested_pred2[i, 1] / 50] = nested_pred2[i, 2]
        y_for_xx1 = pred_pan1.values #convert from pandas to numpy
        y_for_xx2 = pred_pan2.values

        h, v = np.meshgrid(R1_n, I1_n)
        #P_e1 = np.reshape(y_pred,(len(I1_n),len(R1_n)))
        heights = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        cs1 = plt.contour(h, v, y_for_xx1, levels = heights, colors = 'blue', linewidths = 0.9, linestyles = 'solid', alpha = 0.7)
        cs2 = plt.contour(h, v, y_for_xx2, levels = heights, colors = 'red', linewidths = 0.9, linestyles = 'solid', alpha = 0.7)
        plt.clabel(cs1, inline = 1, inline_spacing = 18, fontsize = 10, fmt = '%1.1f')
        plt.clabel(cs2, inline = 1, inline_spacing = 18, fontsize = 10, fmt = '%1.1f')

        cs1.collections[0].set_label('Original')
        cs2.collections[0].set_label('Predicted')
        plt.legend(loc="upper right",frameon = True)
        #plt.show()
        plt.savefig('./Profiling_output/'+str(name0)+'_'+str(name1)+'_'+str(name2)+'.pdf')



def run_it(tmp=0):
    ''' An example to use this model
    '''
    for i in range(0,18):
        print(i,'th Layer processing.')
        pm=Profiling_Model(tmp=i) #init and load data
        pm.train_model() # train estimator
        pm.test_model() # predict and evaluate
        original_stdout = sys.stdout
        with open('profiling_cache.txt', 'a') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print(i,'th Layer.')
            print("MSE",pm.test_mse,"RMSE",pm.test_rmse,"MAE",pm.test_mae,"MAPE",pm.test_mape,"R2",pm.test_r2)
            sys.stdout = original_stdout # Reset the standard output to its original value
    #Profiling_Model(tmp=tmp)


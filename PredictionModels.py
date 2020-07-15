#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense,LSTM,GRU
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from pylab import rcParams

#load dataset
def load_dataset(filename):
    df = pd.read_csv(filename)
    df.rename(columns={'Date/Time':'Time', 'LV ActivePower (kW)':'ActivePower', 'Theoretical_Power_Curve (KWh)':'TheoriticalPower', 'Wind Direction (°)':'WindDirection', 'Wind Speed (m/s)':'Speed'}, inplace=True)
    return df

#function for finding months
def find_month(x):
    if " 01 " in x:
        return 1
    elif " 02 " in x:
        return 2
    elif " 03 " in x:
        return 3    
    elif " 04 " in x:
        return 4    
    elif " 05 " in x:
        return 5    
    elif " 06 " in x:
        return 6    
    elif " 07 " in x:
        return 7    
    elif " 08 " in x:
        return 8    
    elif " 09 " in x:
        return 9    
    elif " 10 " in x:
        return 10    
    elif " 11 " in x:
        return 11    
    else:
        return 12

#appending date, month and time columns
def append_time(df):
    df['Month'] = df.Time.apply(find_month)
    df['Time']=pd.to_datetime(df.Time)
    df['Day'] = df['Time'].dt.day
    df['Hour'] = df['Time'].dt.hour
    df['Minute'] = df['Time'].dt.minute
    return df
    
#to categorize direction of wind
def find_direction(x):
    if x==0:
        return "N"
    if x==30:
        return "NNE"
    if x==60:
        return "NEE" 
    if x==90:
        return "E" 
    if x==120:
        return "SEE" 
    if x==150:
        return "SSE" 
    if x==180:
        return "S" 
    if x==210:
        return "SSW" 
    if x==240:
        return "SWW" 
    if x==270:
        return "W" 
    if x==300:
        return "NWW" 
    if x==330:
        return "NNW"

#to find mean direction from given values
def mean_direction(x):
    list = []
    i = 15
    while i <= 375:
        list.append(i)
        i += 30
        
    for i in list:
        if x < i:
            x = i-15
            if x == 360:
                return 0
            else:
                return x

#adding features
def add_features(df):
    df["meanDirection"] = df["WindDirection"].apply(mean_direction)
    df["Direction"] = df["meanDirection"].apply(find_direction)
    return df

#separating features and targets
def features_targets(df):
    X = df.drop(labels = ['Time','TheoriticalPower', 'ActivePower'], axis = 1)
    y = df['ActivePower']
    return X, y

def data_split_reg(X, y):
    #train test split for regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 47)

    X_train_enc = pd.concat([X_train['WindDirection'], pd.get_dummies(X_train, prefix = 'Dir')], axis = 1) 
    X_train_enc.drop('WindDirection', axis = 1, inplace = True)
    X_test_enc = pd.concat([X_test['WindDirection'], pd.get_dummies(X_test, prefix = 'Dir')], axis = 1)
    X_test_enc.drop('WindDirection', axis = 1, inplace = True)
    return X_train_enc, X_test_enc, y_train, y_test

#predict new data
def model_predict(model, X_test):
   y_pred = model.predict(X_test)
   return y_pred

#error calculation
def reg_model_accuracy(model, x_test, y_pred, y_test):
    acc = model.score(x_test, y_test)
    MSE = np.sqrt(mean_squared_error(y_test, y_pred))
    #MAE = mean_absolute_error(y_test, y_pred)
    return acc, MSE

#Regression Models
def models(x_train, y_train, x_test, y_test):

    #Lineear Regression
    model1 = LinearRegression().fit(x_train, y_train.values.reshape(len(y_train), 1))
    y_pred = model_predict(model1, x_test)
    acc, mse = reg_model_accuracy(model1, x_test, y_pred, y_test)
    print("Linear Regressor accuracy: " + str(acc*100) + "%")
    print("Linear Regressor MSE: " + str(mse))
    #Bayesian Ridge
    model2 = BayesianRidge().fit(x_train, y_train.values.ravel())
    y_pred = model_predict(model2, x_test)
    acc, mse = reg_model_accuracy(model2, x_test, y_pred, y_test)
    print("Bayesian Ridge Regressor accuracy: " + str(acc*100) + "%")
    print("Bayesian Ridge Regressor MSE: " + str(mse))
    #RandomForest Regression
    model3 = RandomForestRegressor().fit(x_train, y_train.values.ravel())
    y_pred = model_predict(model3, x_test)
    acc, mse = reg_model_accuracy(model3, x_test, y_pred, y_test)
    print("RandomForest Regressor accuracy: " + str(acc*100) + "%")
    print("RandomForest Regressor MSE: " + str(mse))
    return model3, y_pred


#LSTM preprocessing
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#ordinal encoding features: direction
def feature_encoding(X):
    #encoding cols
    X = pd.concat([X['Direction'], pd.get_dummies(X, prefix = 'dir')], axis = 1) 
    X.drop('Direction', axis = 1, inplace = True)
    return X

#LSTM Scaling
def scale(X):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(X)
    reframed = series_to_supervised(scaled,1,1)
    reframed.drop(reframed.columns[[21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]], axis=1, inplace=True)
    return reframed

#LSTM train test split
def data_split(df):
  x = df.shape[0]
  n_train = int(math.ceil(x*0.8))
  train = df[:n_train]
  test = df[n_train:]
  trainX,trainY = train[:,:-1],train[:,-1]
  testX,testY = test[:,:-1],test[:,-1]
  trainX = trainX.reshape(trainX.shape[0],1,trainX.shape[1])
  testX = testX.reshape(testX.shape[0],1,testX.shape[1])
  return trainX, testX, trainY, testY

#Model 4: LSTM neural networks training
def LSTM_train(X_train, X_test, y_train, y_test):
    stop_noimprovement = EarlyStopping(patience=10)
    model4 = Sequential()
    model4.add(LSTM(50,input_shape=(X_train.shape[1], X_train.shape[2]),dropout=0.2))
    model4.add(Dense(1))
    model4.compile(optimizer = 'adam', loss = 'mae',metrics=['mae','mse'])
    #Check Point
    checkpoint_name = 'req_model.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='loss', verbose = 2, save_best_only = True, mode ='max')
    #callbacks_list = [checkpoint]
    history = model4.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=100,verbose=1,shuffle=False,batch_size=500)
    return model4, history

#error calculation
def lstm_model_accuracy(y_pred, y_test):
    MSE = np.sqrt(mean_squared_error(y_test, y_pred))
    acc = 0
    #MAE = mean_absolute_error(y_test, y_pred)
    return acc, MSE

#LSTM performance
def plot_history(history):

    #plotting style
    plt.style.use("ggplot")
    #plotting data
    plt.figure(figsize=(25, 8))
    plt.plot(history.history['loss'],label='train')
    plt.plot(history.history['val_loss'],label='test')
    plt.legend()
    plt.show()


def run_models(path):
    df = load_dataset(path)
    df = append_time(df)
    df = add_features(df)
    X, y = features_targets(df)
    X_train_enc, X_test_enc, y_train, y_test = data_split_reg(X, y)
    model, y_pred =  models(X_train_enc, y_train, X_test_enc, y_test)
    '''
    X = feature_encoding(X)
    reframed = scale(X)
    values = reframed.values
    X_train, X_test, y_train, y_test = data_split(values)
    model4, history = LSTM_train(X_train, X_test, y_train, y_test)
    #LSTM prediction
    y_pred = model_predict(model4, X_test)
    #error calculation
    acc, mse = lstm_model_accuracy(y_pred, y_test)
    #LSTM accuracy
    #print("LSTM Regressor accuracy: " + str(acc*100) + "%")
    print("LSTM Regressor MSE: " + str(mse))
    #LSTM performance
    plot_history(history)
    '''
    return model, y_pred
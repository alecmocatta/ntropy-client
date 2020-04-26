import requests
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as Kb
from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau 

class Model:
    def __init__(self, n_input, num_layers = 2, num_units = None, layer_activation = 'sigmoid', final_activation = 'sigmoid', loss = 'mse', optimizer = 'adam', dropout_frac = 0.2, encoder = False):

        if not num_units:
            num_units = n_input

        Kb.clear_session()

        self.model = Sequential()
        self.model.add(Dense(num_units, activation=layer_activation, input_shape=(n_input,), trainable=True, name="in"))
        for l in range(num_layers):
            self.model.add(Dropout(dropout_frac))
            self.model.add(Dense(num_units, activation=layer_activation, trainable=True, name=str(l)))

        if not encoder:
            self.model.add(Dropout(dropout_frac))
            self.model.add(Dense(n_input, activation=final_activation, trainable=True, name="out"))

        self.model.compile(loss=loss, optimizer=optimizer)

    def fit(self, x_train, verbose = False):
        callbacks = [
                    EarlyStopping(monitor='val_loss', min_delta=0, patience=50, mode='min', baseline=None, restore_best_weights=True, verbose=verbose),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5, mode='min', verbose=verbose),
                    ]
        self.model.fit(x_train, x_train, batch_size=min(100,int(0.1 * len(x_train))), epochs=1000, validation_split=0.2, shuffle=True, callbacks=callbacks, verbose=verbose) 

    def predict(self, x):
        return self.model.predict(x)

def encode(x_train, num_layers = 2):
    model = Model(n_input=x_train.shape[1], num_layers=num_layers)
    model.fit(x_train)
    weights = model.model.get_weights()
    num_layers_encoder = int(num_layers/2)
    encoder = Model(n_input=x_train.shape[1], num_layers=num_layers_encoder, encoder = True)
    encoder.model.set_weights(weights[:(2 * (1 + num_layers_encoder))])
    return encoder.predict(x_train)

class NetworkClassifier():
    def __init__(self, server, api_key):
        self.server = server
        self.api_key = api_key
        self.data_hash = None

    def fit(self, x_train, y_train, use_encoding = True):
        if use_encoding:
            print("Encoding data...")
            x_train = encode(x_train)
            print("Done")
        data = np.zeros((x_train.shape[0],x_train.shape[1]+1))
        data[:,:-1] = x_train
        data[:,-1] = y_train
        payload = data.tostring()

        data_hash = str(hash(payload))
        self.data_hash = data_hash
        status = requests.get(self.server + "/check_status", params = {"API_KEY": self.api_key, "data_hash": data_hash}).text
        if status == "key invalid":
            print("API key invalid")
            return 
        if status == "ready":
            requests.post(self.server + '/fit', data=payload, params = {"API_KEY": self.api_key, "data_hash": data_hash, "x": data.shape[0], "y": data.shape[1]})
        while status != "done":
            time.sleep(3)
            status = requests.get(self.server + "/check_status", params = {"API_KEY": self.api_key, "data_hash": data_hash}).text

    def predict(self, valid_features):
        if not self.data_hash:
            print("Please fit model first")
            return
        status = requests.get(self.server + "/check_status", params = {"API_KEY": self.api_key, "data_hash": self.data_hash}).text
        if status == "key invalid":
            print("API key invalid")
            return 
        if status == "ready":
            print("Please fit model first")
            return
        elif status == "busy":
            print("Your model is being trained. Please come back later.")
            return
        else:
            valid_labels = requests.post(self.server + "/predict", params = {"API_KEY": self.api_key, "data_hash": self.data_hash, "x": valid_features.shape[0], "y": valid_features.shape[1]}, data = valid_features.tostring()).json()
            return np.array(valid_labels)


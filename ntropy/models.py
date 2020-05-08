import requests
import numpy as np
import time
from sklearn.ensemble import RandomTreesEmbedding

class NetworkClassifier():
    def __init__(self, server, api_key):
        self.server = server
        self.api_key = api_key
        self.data_hash = None
        self.encoder = None

    def fit(self, x_train, y_train, use_encoder = True):
        if use_encoder:
            print("Generating encoder...")
            self.encoder = RandomTreesEmbedding(n_estimators=200, max_depth=8, random_state=0)
            self.encoder.fit(x_train)
            x_train = self.encoder.transform(x_train).todense()
            print("Done")
        data = np.zeros((x_train.shape[0],x_train.shape[1]+1))
        data[:,:-1] = x_train
        data[:,-1] = y_train
        payload = data.astype(float).tostring()

        data_hash = str(hash(payload))
        self.data_hash = data_hash
        status = requests.get(self.server + "/check_status", params = {"API_KEY": self.api_key, "data_hash": data_hash}).text
        if status == "key invalid":
            print("API key invalid")
            return 
        if status == "ready":
            requests.post(self.server + '/fit', data=payload, params = {"API_KEY": self.api_key, "data_hash": data_hash, "d0": data.shape[0], "d1": data.shape[1]})
        while status != "done":
            time.sleep(3)
            status = requests.get(self.server + "/check_status", params = {"API_KEY": self.api_key, "data_hash": data_hash}).text

    def predict(self, x):
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
            if self.encoder:
                print("Encoding data...")
                x = self.encoder.transform(x).todense()
                print("Done...")
            payload = x.astype(float).tostring()
            y = requests.post(self.server + "/predict", params = {"API_KEY": self.api_key, "data_hash": self.data_hash, "d0": x.shape[0], "d1": x.shape[1]}, data = payload).json()
            return np.array(y)


import requests
import numpy as np
import time

class NetworkClassifier():
    def __init__(self, server, api_key):
        self.server = server
        self.api_key = api_key
        self.data_hash = None

    def fit(self, train_features, train_labels):
        data = np.zeros((train_features.shape[0],train_features.shape[1]+1))
        data[:,:-1] = train_features
        data[:,-1] = train_labels
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


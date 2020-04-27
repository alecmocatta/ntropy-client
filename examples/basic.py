import numpy as np
import random
import json
from ntropy.models import NetworkClassifier

#get server url and api key from config file
with open("config.json", "r") as f:
    config = json.load(f)
    SERVER = config["server"]
    API_KEY = config["api_key"]

N = 1000 #number of observations
M = 100 #number of features

#initialising data
data_x = np.random.rand(N,M) #numerical observations
data_y = np.array([[random.randint(0,1)] for i in range(N)]).flatten() #binary labels

#initialising model
model = NetworkClassifier(SERVER, API_KEY)

#fitting model to first 80% (training data)
Nc = int(0.8 * len(data_x))
model.fit(data_x[:Nc], data_y[:Nc], use_encoder = True)

#predicting labels on last 20% (test data)
y = model.predict(data_x[Nc:])


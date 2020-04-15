import numpy as np
import random
from ntropy.models import NetworkClassifier

SERVER = "COMPUTE_SERVER_URL"
API_KEY = "YOUR_API_KEY"

N = 1000 #number of observations
M = 1000 #number of features

#initialising data
train_features = np.random.rand(N,M)
train_labels = np.array([[random.randint(0,1)] for i in range(N)]).flatten()
valid_features = np.random.rand(N,M)

#initialising model
model = NetworkClassifier(SERVER, API_KEY)

#fitting model
model.fit(train_features, train_labels)

#predicting labels on validation data
valid_labels = model.predict(valid_features)


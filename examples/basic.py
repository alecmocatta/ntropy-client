import numpy as np
import random
import json
from ntropy.models import NetworkGenerator

# get server url and api key from config file
with open("config.json", "r") as f:
    config = json.load(f)
    SERVER = config["server"]
    API_KEY = config["api_key"]

# initialize data
N = 1000  # number of observations
M = 100  # number of features
dataset_id = "example_dataset"  # unique dataset id
samples = np.random.rand(N, M)  # numerical observations
labels = np.random.randint(2, size=N)  # labels
weights = np.random.uniform(0, 1, N)  # sample weights

# initialize generator
generator = NetworkGenerator(
    SERVER,
    API_KEY,
    x_dim=samples.shape[1],
    h_dim=512,
    z_dim=8,
    c_dim=len(np.unique(labels)),
    num_layers=2,
)  

# train generator locally on our data
generator.fit(
    samples, labels, weights, batch_size=128, num_epochs=1, learning_rate=0.001
)  

# upload generator to server
generator.upload(dataset_id)  

# get improved dataset from server
samples_new, labels_new, weights_new = generator.generate(
    dataset_id, N
)  

# train model on new data...

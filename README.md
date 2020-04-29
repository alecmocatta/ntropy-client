# ntropy-client

A python package to run, train and benchmark machine learning models on data residing in multiple data silos.

To achieve this, we train a global model on data from all datasets at once. Data from different datasets can boost accuracy and robustness of a machine learning model far beyond just a single dataset.

For example, from combining 4 datasets of credit card transactions from 3 different organizations, ROC AUC already improves by more than 5%. See https://medium.com/ntropy-network/dissolving-data-silos-21e5eaab11f6 for more details.

![Benchmark 1](https://raw.githubusercontent.com/ntropy-network/ntropy-client/master/images/img2.png | width=300px)

There are currently two modes of operation for the global model:

1. encoder that maps data from all distributions onto a shared latent space. Respective local models can then be trained locally on this latent data. This is optimal for a large number of datasets and supports different use cases for each client.

2. end-to-end classifier what translates each incoming observation into the final label. This is suitable for smaller scale, narrower setups, where all clients are interested in the same type of labels.

![Deployment diagram](https://raw.githubusercontent.com/ntropy-network/ntropy-client/master/images/img1.png | width=300px)

## FAQ

#### What data privacy guarantees does this have?

A machine learning model only learns from the distribution of the data, independent of the encoding. Instead of each client sending raw data directly, it can therefore only send a latent representation of the data. Although this approach does not have any formal privacy guarantees, it is currently one of the most common ways of dealing with private or sensitive datasets.  

#### I have data I want to monetise. Is this possible?

Yes. The network relies on both data producers and data consumers. We will be rolling out algorithmic data monetization soon.

#### What kind of data is the global model currently trained on?

The framework itself is data agnostic. However, Ntropy is currently focusing on fraud and underwriting data in fintech.

#### How do you prevent malicious data from polluting the global model? 

The server has a separate validation dataset which is used to weigh each dataset by how much it improves performance on that validation data. If performance decreases, the dataset is discarded. 

#### How do I start using this?

Please contact partners@ntropy.network to join our partner network.

Alternatively, you can set up your own server to enable training on internal datasets. Will be releasing our server-side code in the near future. 

#### How do I contribute to this project?

Check out https://ntropy.network for more info or write to jobs@ntropy.network. We are actively hiring.

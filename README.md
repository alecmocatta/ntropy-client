# ntropy-client

A python package to train machine learning models on data residing in multiple data silos. Data from different datasets can boost accuracy and robustness of a machine learning model far beyond whats possible on just a single dataset.

To achieve this, a data generator on the server is trained on data from all participants in the network. Each participant can then sample a dataset in their local feature encoding. As the server combines information from multiple structurally similar datasets, it can generate significantly better data than any of the loal datasets. The client-server data flow is the following:

<img src="https://raw.githubusercontent.com/ntropy-network/ntropy-client/master/images/img1.png" width="50%">

For example, from combining 4 datasets of credit card transactions from 3 different organizations, ROC AUC already improves by more than 5%. See https://medium.com/ntropy-network/dissolving-data-silos-21e5eaab11f6 for more details.

<img src="https://raw.githubusercontent.com/ntropy-network/ntropy-client/master/images/img2.png" width="50%">

## FAQ

#### What data privacy guarantees does this have?

Local datasets are not sent directly to the server. Instead, a generator is first trained on its local dataset on the client. Then, noise is added to the weights of the trained generator to enforce mathematical privacy guarantees. Only then is the weights vector from that generator sent to the server.

When the client requests a new dataset, this data is sampled from the global generator on the server, and hence does not have associated privacy risks.

#### What kind of data is the global model currently trained on?

The framework itself is data agnostic. However, Ntropy is currently focusing on financial fraud and underwriting data.

#### How do you prevent malicious data from polluting the global model? 

The server has a number of validation metrics which are used to weigh each dataset by how much it improves performance. If performance does not improve, the respective dataset is not added to the server model. 

#### How do I start using this?

Please contact partners@ntropy.network to join our partner network.

#### How do I contribute to this project?

Check out https://ntropy.network for more info or write to jobs@ntropy.network. We are actively hiring.

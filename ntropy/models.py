import time

import requests
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as distributions
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

HAS_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")


class NetworkGenerator:
    r"""Train generator on local dataset and samples improved dataset
    from server model.

    Args:
        server: server address
        api_key: key to query the server. A unique key is allocated to
            each network participant by the server 
        x_dim: dimension of input vector
        h_dim: number of units in each layer
        z_dim: dimension of latent space distribution
        c_dim: dimension of dataset label vector which conditions both
            the encoder and decoder
        num_layers: number of layers in the encoder and decoder
    """

    class Model(nn.Module):
        def __init__(self, x_dim, h_dim, z_dim, c_dim, num_layers):
            super(NetworkGenerator.Model, self).__init__()

            self.fe0 = nn.Linear(h_dim, z_dim)
            self.fe1 = nn.Linear(h_dim, z_dim)

            self.encoder_list = nn.ModuleList(
                [nn.Linear(x_dim + c_dim, h_dim), nn.ReLU()]
            )
            for _ in range(num_layers - 1):
                self.encoder_list.extend([nn.Linear(h_dim, h_dim), nn.ReLU()])

            self.decoder_list = nn.ModuleList(
                [nn.Linear(z_dim + c_dim, h_dim), nn.ReLU()]
            )
            for _ in range(num_layers - 1):
                self.decoder_list.extend([nn.Linear(h_dim, h_dim), nn.ReLU()])
            self.decoder_list.extend([nn.Linear(h_dim, x_dim), nn.Sigmoid()])

        def encoder(self, x, c):
            out = torch.cat([x, c], 1)
            for f in self.encoder_list:
                out = f(out)
            return self.fe0(out), self.fe1(out)

        def decoder(self, z, c):
            out = torch.cat([z, c], 1)
            for f in self.decoder_list:
                out = f(out)
            return out

        def forward(self, x, c):
            mu, log_var = self.encoder(x, c)
            z = torch.distributions.Normal(mu, torch.exp(0.5 * log_var)).rsample()
            return self.decoder(z, c), mu, log_var

    def __init__(
            self,
            server,
            api_key,
            x_dim,
            h_dim,
            z_dim,
            c_dim,
            num_layers,
        ):
        if num_layers <= 0:
            raise ValueError("model must have at least 1 layer")
        self.server = server
        self.models = [self.Model(
            x_dim=x_dim,
            h_dim=h_dim,
            z_dim=z_dim,
            c_dim=c_dim,
            num_layers=num_layers
        ).to(DEVICE) for _ in range(2)]
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.params = {
            "api_key": api_key,
            "x_dim": x_dim,
            "h_dim": h_dim,
            "z_dim": z_dim,
            "c_dim": c_dim,
            "num_layers": num_layers,
        }

    def _one_hot(self, labels):
        targets = torch.zeros(labels.size(0), self.c_dim)
        for i, label in enumerate(labels):
            targets[i, label] = 1
        return Variable(targets)

    def _loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

    def _fit_model(self, ind, samples, labels, weights, batch_size, num_epochs, learning_rate):
        dataset = TensorDataset(
            torch.from_numpy(samples).float(), torch.from_numpy(labels)
        )
        kwargs = {"num_workers": 1, "pin_memory": True} if HAS_CUDA else {}
        if type(weights).__name__ == "ndarray":
            sampler = WeightedRandomSampler(
                weights=weights, num_samples=len(samples))
            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                **kwargs)
        else:
            loader = DataLoader(
                dataset=dataset, batch_size=batch_size, shuffle=True, **kwargs
            )
        optimizer = optim.Adam(self.models[ind].parameters(), lr=learning_rate)
        for epoch in range(1, num_epochs + 1):
            self.models[ind].train()
            train_loss = 0
            for batch_ind, (data, cond) in enumerate(loader):
                data, cond = data.to(DEVICE), self._one_hot(cond).to(DEVICE)
                optimizer.zero_grad()
                recon_batch, mu, log_var = self.models[ind](data, cond)
                loss = self._loss_function(recon_batch, data, mu, log_var)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                if batch_ind % 100 == 0:
                    print(
                        "epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_ind * len(data),
                            len(loader.dataset),
                            100.0 * batch_ind / len(loader),
                            loss.item() / len(data),
                        )
                    )
            print(
                "====> epoch: {} avg loss: {:.4f}".format(
                    epoch, train_loss / len(loader.dataset)
                )
            )

    def fit(
        self,
        samples,
        labels,
        weights=None,
        batch_size=1,
        num_epochs=20,
        learning_rate=0.001,
    ):
        r"""Trains generator on local data

        Args:
            samples: 2D N x M numpy array of floats with N observations
                and M features per observaton
            labels: 1D numpy array of integers with N sample labels
            weights: 1D numpy array of floats with N sample weights. Default: None
            batch_size: batch_size. Default: 1
            num_epochs: number of epochs to train. Default: 20
            learning_rate: learning rate for training. Default: 0.001
        """

        Nc = int(0.75 * len(samples)) 
        samples_train = samples[:Nc]
        labels_train = labels[:Nc]
        weights_train = weights[:Nc] if type(weights).__name__ == "ndarray" else None
        samples_test = samples[Nc:]
        labels_test = labels[Nc:]
        weights_test = weights[Nc:] if type(weights).__name__ == "ndarray" else None

        self._fit_model(0, samples_train, labels_train, weights_train, batch_size, num_epochs, learning_rate)
        self._fit_model(1, samples_test, labels_test, weights_test, batch_size, num_epochs, learning_rate)

    def generate(self, dataset_id, N):
        r"""Samples from server generator. 

        Args:
            dataset_id: unique id of local dataset
            N: number of samples to generate
        """
        self.params["dataset_id"] = dataset_id
        self.params["N"] = N
        status = requests.get(
            self.server + "/check_status", params=self.params).text
        if status == "key invalid":
            print("API key invalid")
            return
        requests.post(self.server + "/generate", params=self.params)
        while status != "done":
            time.sleep(3)
            status = requests.get(
                self.server + "/check_status", params=self.params
            ).text
        data = requests.get(self.server + "/data", params=self.params).json()
        return np.array(data["samples"]), np.array(data["labels"]), np.array(data["weights"])

    def upload(self, dataset_id, dataset_type):
        r"""Uploads local model to server

        Args:
            dataset_id: unique id of the local dataset
            dataset_type: type of dataset. Use by server to match with
            the right global model
        """
        self.params["dataset_id"] = dataset_id
        self.params["dataset_type"] = dataset_type
        status = requests.get(
            self.server + "/check_status", params=self.params).text
        if status == "key invalid":
            raise ValueError("API key invalid")
        filename_train = "./model_train.pt"
        filename_test = "./model_test.pt"
        torch.save(self.models[0].state_dict(), filename_train)
        torch.save(self.models[1].state_dict(), filename_test)
        with open(filename_train, "rb") as fp_train:
            with open(filename_test, "rb") as fp_test:
                res = requests.post(
                        self.server + "/upload", files={"train": fp_train, "test": fp_test}
                        , params=self.params
                ).text
                print(res)

import requests
import time
import torch

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


class NetworkGenerator:
    r"""Train generator on local dataset and samples improved dataset from server model.

    Args:
        SERVER: server address
        API_KEY: key to query the server. A unique key is allocated to each network participant by the server 
        x_dim: dimension of input vector
        h_dim: number of units in each layer
        z_dim: dimension of latent space distribution
        c_dim: dimension of dataset label vector which conditions both the encoder and decoder
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

        def sampling(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)

        def forward(self, x, c):
            mu, log_var = self.encoder(x, c)
            z = self.sampling(mu, log_var)
            return self.decoder(z, c), mu, log_var

    def __init__(self, SERVER, API_KEY, x_dim, h_dim, z_dim, c_dim, num_layers):
        assert num_layers > 0, "model must have at least 1 layer"
        self.SERVER = SERVER
        self.model = self.Model(
            x_dim=x_dim, h_dim=h_dim, z_dim=z_dim, c_dim=c_dim, num_layers=num_layers
        ).to(device)
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.params = {
            "API_KEY": API_KEY,
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
            samples: 2D N x M numpy array of floats with N observations and M features per observaton
            labels: 1D numpy array of integers with N sample labels
            weights: 1D numpy array of floats with N sample weights
            batch_size: batch_size. Default: 1
            num_epochs: number of epochs to train. Default: 20
            learning_rate: learning rate for training. Default: 0.001
        """
        dataset = TensorDataset(
            torch.from_numpy(samples).float(), torch.from_numpy(labels)
        )
        kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}
        if type(weights).__name__ == "ndarray":
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(samples))
            loader = DataLoader(
                dataset=dataset, batch_size=batch_size, sampler=sampler, **kwargs
            )
        else:
            loader = DataLoader(
                dataset=dataset, batch_size=batch_size, shuffle=True, **kwargs
            )
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            train_loss = 0
            for batch_idx, (data, cond) in enumerate(loader):
                data, cond = data.to(device), self._one_hot(cond).to(device)
                optimizer.zero_grad()
                recon_batch, mu, log_var = self.model(data, cond)
                loss = self._loss_function(recon_batch, data, mu, log_var)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                if batch_idx % 100 == 0:
                    print(
                        "epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(loader.dataset),
                            100.0 * batch_idx / len(loader),
                            loss.item() / len(data),
                        )
                    )
            print(
                "====> epoch: {} avg loss: {:.4f}".format(
                    epoch, train_loss / len(loader.dataset)
                )
            )

    def generate_local(self, N):
        r"""Samples from local generator. The labels will be mixed evently.

        Args:
            N: number of samples to generate
        """
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(N, self.z_dim).to(device)
            c = torch.zeros(N, self.c_dim)
            labels = torch.empty(N, dtype=torch.long)
            for i in range(N):
                label = random.randint(0, self.c_dim - 1)
                c[i, label] = 1
                labels[i] = label
            samples = self.model.decoder(z, c.to(device)).cpu()
        weights = torch.ones(labels.size(0))
        return samples, labels, weights

    def generate(self, dataset_id, N):
        r"""Samples from server generator. 

        Args:
            dataset_id: unique id of local dataset
            N: number of samples to generate

        """
        self.params["dataset_id"] = dataset_id
        self.params["N"] = N
        status = requests.get(self.SERVER + "/check_status", params=self.params).text
        if status == "key invalid":
            print("API key invalid")
            return
        requests.post(self.SERVER + "/generate", params=self.params)
        while status != "done":
            time.sleep(3)
            status = requests.get(
                self.SERVER + "/check_status", params=self.params
            ).text
        data = requests.get(self.SERVER + "/data", params=self.params).json()
        return data["samples"], data["labels"], data["weights"]

    def save(self, filename):
        r"""Saves local model state to file

        Args:
            filename: path to file where to save the model state

        """
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        r"""Loads local model state from file

        Args:
            filename: path to file which stores the model state 
        """
        self.model.load_state_dict(torch.load(filename))

    def upload(self, dataset_id):
        r"""Uploads local model to server

        Args:
            dataset_id: unique id of the local dataset
        """
        self.params["dataset_id"] = dataset_id
        status = requests.get(self.SERVER + "/check_status", params=self.params).text
        if status == "key invalid":
            print("API key invalid")
            return
        filename = "./model.pt"
        self.save(filename)
        with open(filename, "rb") as fp:
            res = requests.post(
                self.SERVER + "/upload", files={"file": fp}, params=self.params
            ).text
            print(res)

import torch
import torch.nn.functional as F
from pyod.models.base import BaseDetector
from time import time
from pyod.models.knn import KNN
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
from torch.optim import Adam
import numpy as np


def stochastic_instance_closeness_pdist_loss(mX, easy_ratio, hard_ratio):
    dists = torch.pdist(mX) ** 2

    num_dists = dists.shape[0]

    dists, _ = torch.sort(dists)

    loss = 0.0

    if easy_ratio > 0:
        easy_dists = dists[:int(num_dists * easy_ratio)]

        easy_loss = torch.mean(easy_dists)

        loss += easy_loss

    if hard_ratio > 0:
        hard_dists = dists[int(-num_dists * hard_ratio):]

        print(dists.shape, hard_dists.shape)
        hard_loss = (hard_ratio / (hard_ratio + easy_ratio)) * torch.mean(hard_dists)

        loss += hard_loss

    return loss

class FFNeuralNetwork(torch.nn.Module):

    def __init__(self, input_dim, output_dim, num_hidden_neurons, num_hidden_neurons2, device):

        super().__init__()
        self.layers = []

        self.model = self.fc_small(input_dim, output_dim, num_hidden_neurons, num_hidden_neurons2).to(device)


    def forward(self, X):

        return self.model.forward(X)

    def get_regularization_loss(self):
        loss = 0.0
        for layer in self.layers:
            loss += torch.sum(layer.weight**2 - 1)**2

        return loss

    def fc_small(self, input_dim, output_dim, num_hidden_neurons = 64, num_hidden_neurons2 = 64):

        if num_hidden_neurons > 0:
            if num_hidden_neurons2 == 0:
                layer1 = torch.nn.Linear(input_dim, num_hidden_neurons, bias=True)
                layer2 = torch.nn.Linear(num_hidden_neurons, output_dim, bias=True)

                self.layers += [layer1, layer2]

                model = torch.nn.Sequential(
                    layer1,
                    torch.nn.Tanh(),
                    layer2,
                    torch.nn.Tanh(),

                )
            else:
                model = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, num_hidden_neurons, bias=True),
                    torch.nn.Tanh(),
                    torch.nn.Linear(num_hidden_neurons, num_hidden_neurons2, bias=True),
                    torch.nn.Tanh(),
                    torch.nn.Linear(num_hidden_neurons2, output_dim, bias=True),
                    torch.nn.Softmax(),
                )

        else:
            model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, output_dim, bias=True),
                torch.nn.Softmax(),

            )

        return model


class NNTrainer:

    def __init__(self, model_cls, metric_dim, num_epochs, lr, weight_decay, val_size, mini_batch_size, patience,
                 num_hidden_neurons, num_hidden_neurons2, device, normal_ratio, anomaly_ratio, easy_ratio,
                 hard_ratio, **kwargs):

        self.hard_ratio = hard_ratio
        self.easy_ratio = easy_ratio
        self.normal_ratio = normal_ratio
        self.anomaly_ratio = anomaly_ratio
        self.loss = None
        self.mini_batch_size = mini_batch_size
        self.val_size = val_size
        self.model_cls = model_cls
        self.model = None

        self.num_epochs = num_epochs

        self.metric_dim = metric_dim
        self.weight_decay = weight_decay
        self.lr = lr
        self.patience = patience
        self.num_hidden_neurons = num_hidden_neurons
        self.num_hidden_neurons2 = num_hidden_neurons2

        self.device = torch.device(device)

        self.c = None

        self.normal_indices = None

    def shuffled_mini_batch(self, num_train, X_train):

        train_indices = shuffle(list(range(num_train)))

        for mb_start in range(0, num_train, self.mini_batch_size):
            batch = X_train[train_indices[mb_start:min(mb_start + self.mini_batch_size, num_train - 1)], :]

            yield batch

    def shuffled_filtered_mini_batch(self, num_train, X_train):

        train_indices = shuffle(self.normal_indices)
        num_train = len(self.normal_indices)

        for mb_start in range(0, num_train, self.mini_batch_size):
            batch = X_train[train_indices[mb_start:min(mb_start + self.mini_batch_size, num_train - 1)], :]

            yield batch

    def fit(self, X):
        X = torch.Tensor(X)

        self.model = self.model_cls(input_dim=X.shape[1],
                                    output_dim=self.metric_dim,
                                    num_hidden_neurons=self.num_hidden_neurons,
                                    num_hidden_neurons2=self.num_hidden_neurons2,
                                    device=self.device).to(self.device)

        # self.model.apply(lambda m: torch.nn.init.kaiming_uniform_(m.weight.data) if hasattr(m, 'weight') else None)
        self.model.apply(lambda m: torch.nn.init.xavier_uniform(m.weight.data) if hasattr(m, 'weight') else None)

        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        num_train = int(X.shape[0] * (1 - self.val_size))

        X_train, X_val = X[:num_train, :], X[num_train:, :]

        best_val = float("inf")
        best_model = self.model
        patience_left = self.patience

        self.calculate_center(X_train)
        total_duration = 0
        total_epochs = 0
        for i in range(self.num_epochs):
            t0 = time()

            batches = self.shuffled_filtered_mini_batch(num_train, X_train)

            for batch in batches:
                self.model.zero_grad()
                optimizer.zero_grad()

                loss = self.forward_pass(batch)

                loss.backward()

                optimizer.step()

            self.calculate_center(X_train)

            if self.val_size > 0:

                with torch.no_grad():
                    loss = self.forward_pass(X_val)
                    val_loss = loss.item()

                val_loss = val_loss

                if val_loss < best_val:
                    best_val = val_loss
                    patience_left = self.patience

                    model_clone = self.model_cls(input_dim=X.shape[1], output_dim=self.metric_dim, num_hidden_neurons=self.num_hidden_neurons, num_hidden_neurons2=self.num_hidden_neurons2, device=self.device)
                    model_clone.load_state_dict(self.model.state_dict())

                    best_model = model_clone
                else:
                    patience_left -= 1

                t1 = time()
                duration = round(t1 - t0, ndigits=4)
                total_duration += duration
                total_epochs += 1

                if patience_left <= 0 or i == self.num_epochs - 1:
                    self.model = best_model.to(self.device)

                    # print(f"Early stopping at epoch {i}")

                    break

        return total_duration / total_epochs

    def generate_all_combinations(self, length):
        anchors = []
        positives = []
        for i in range(length):
            for j in range(i + 1, length):
                anchors.append(i)
                positives.append(j)

        return anchors, positives

    def calculate_loss(self, mX):

        return stochastic_instance_closeness_pdist_loss(mX, self.easy_ratio, self.hard_ratio)

    def calculate_center(self, X_train):

        with torch.no_grad():
            mX = self.forward(X_train)
            self.c = torch.mean(mX, dim=0)

            center_dists = torch.nn.functional.pairwise_distance(mX, self.c.expand_as(mX), p=2)

            if self.normal_ratio >= 1:
                self.normal_indices = np.array(list(range(mX.shape[0])))
            else:
                self.normal_indices = torch.topk(center_dists, int(mX.shape[0] * self.normal_ratio), largest=False).indices.detach().cpu().numpy()

    def forward_pass(self, X):
        mX = self.model.forward(X.to(device=self.device))

        loss = self.calculate_loss(mX)

        return loss

    def forward(self, X):

        X = torch.Tensor(X).to(device=self.device)

        return self.model.forward(X)


class OCML(BaseDetector):

    def __init__(self, contamination=0.1, **kwargs):
        super(OCML, self).__init__(contamination=contamination)

        self.prediction_head = self.get_prediction_head(**kwargs)

        self.model = NNTrainer(FFNeuralNetwork, **kwargs)

    def get_prediction_head(self, prediction_head, knn_method, n_neighbors, **kwargs):
        if prediction_head == "knn":
            return KNNPredictionHead(method=knn_method, n_neighbors=n_neighbors)
        elif prediction_head == "center_distance":
            return CenterDistancePredictionHead()
        elif prediction_head == "all_distance":
            return L2NormPredictionHead()
        else:
            return None

    def fit(self, X, y=None):
        self._set_n_classes(y)

        avg_epoch_duration = self.model.fit(X)

        mX = self.model.forward(X) #.detach().cpu().numpy()

        self.prediction_head.fit(mX)

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()

        return avg_epoch_duration

    def decision_function(self, X):

        mX = self.model.forward(X) #.detach().cpu().numpy()

        return self.prediction_head.decision_function(mX)


class KNNPredictionHead:

    def __init__(self, n_neighbors=5, method="largest", **kwargs):

        self.model = KNN(n_neighbors=n_neighbors, method=method)

    def fit(self, X):

        return self.model.fit(X.detach().cpu().numpy())

    def decision_function(self, X):

        return self.model.decision_function(X.detach().cpu().numpy())


class CenterDistancePredictionHead:

    def __init__(self):
        self.c = None

    def fit(self, X):

        self.c = torch.mean(X, dim=0)

        return self

    def decision_function(self, X):

        return torch.sum((X - self.c)**2, dim=1).detach().cpu().numpy()


class L2NormPredictionHead:

    def __init__(self):
        self.c = None

    def fit(self, X):

        self.X = X.detach().cpu().numpy()

        return self

    def decision_function(self, X):

        return np.sum(cdist(X.detach().cpu().numpy(), self.X), axis=1)

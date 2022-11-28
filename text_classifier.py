import math
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from text_preproc import TextPreproc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataclasses import dataclass


@dataclass
class Parameters:
    # Preprocessing parameters
    seq_len: int = 35
    num_words: int = 2000

    # Model parameters
    embedding_size: int = 64
    out_size: int = 32
    stride: int = 2

    # Training parameters
    epochs: int = 20
    batch_size: int = 1
    learning_rate: float = 0.001


class DenseNN(nn.Module):

    """3 layer dense neural network"""

    def __init__(self, input_size):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(in_features=input_size, out_features=input_size // 100, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=input_size // 100, out_features=input_size // 100, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=input_size // 100, out_features=2, bias=True),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class PandaSet(Dataset):

    """
    Dataset from pandas dataframe.
    Target column must be the last one
    """

    def __init__(self, data):
        super().__init__()

        x = data[data.columns[:-1]].values
        y = data[data.columns[-1]].values

        self.x = torch.tensor(x).to(torch.float32)
        self.y = torch.tensor(y).to(torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class NetworkMaster:

    """master class: training DenseNet model"""

    def __init__(self, train_data, test_data, device):

        self.device = device
        self.train_data = train_data
        self.test_data = test_data


        self.net_model = DenseNN(self.train_data.shape[1] - 1).to(device)
        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
        self.loss_fn = None

    def train_model(self):

        size = len(self.train_loader.dataset)
        self.net_model.train()

        for batch, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            # loss between forward and real vals
            pred = self.net_model(x)
            loss = self.loss_fn(pred, y)

            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 1000 == 0:
                loss, current = loss.item(), batch * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_model(self):

        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        self.net_model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.net_model(x)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def fit(self, epochs=10, batch_size=1, learning_rate=1e-3):

        self.train_loader = DataLoader(PandaSet(self.train_data), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(PandaSet(self.test_data),
                                      batch_size=batch_size,
                                      shuffle=True)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net_model.parameters(), lr=learning_rate)

        for num in range(epochs):
            print(f"Epoch {num + 1}\n-------------------------------")
            self.train_model()
            self.test_model()

        return self.net_model


if __name__ == "__main__":

    # preprocessing:
    preproc_model = TextPreproc(rebalance=True)
    train_data, test_data = preproc_model.get_train_test_preprocd()

    # network training:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net_model = NetworkMaster(train_data, test_data, device).fit(epochs=20, batch_size=1)

    # testing:
    spam_dict = {
        "Hi, how are you feeling? You haven't written for a long time, so I thought something might have happened.": 0,
        'Only today! buy one king-size pizza, get one cola for free! Hurry up!': 1,
        'love you sweetie :)': 0,
        "Buy my book and I'll tell you how to become rich!": 1,
        'bae i cannot wait anymore. I want you now!': 0,
        'You’ve won a price! our phone number: +7 911 XXX-XX-XX': 1,
        'The IRS is trying to contact you': 1,
        'You have a refund coming': 1,
        'Verify your bank account': 1,
        'You have a package delivery': 0,
        'Verify your Apple iCloud ID': 0,
        'A family member needs help': 1,
        'You have a new billing statement': 1}

    pred_y, real_y = [], list(spam_dict.values())
    for letter in spam_dict:
        vectorized_letter = preproc_model.preproc_letter(letter)
        pred_y.append(net_model(torch.Tensor(vectorized_letter[0]).to(torch.float32).to(device)).argmax().item())

    print(pred_y, '\n', real_y)
    print('accuracy:', accuracy_score(pred_y, real_y))
    print('precision:', precision_score(pred_y, real_y))
    print('recall:', recall_score(pred_y, real_y))
    print('f1_score:', f1_score(pred_y, real_y))

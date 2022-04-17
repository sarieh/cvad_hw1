import torch.nn as nn
import torch

d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiLayerQ(nn.Module):
    """Q network consisting of an MLP."""

    def __init__(self, config):
        super(MultiLayerQ, self).__init__()

        self.features = len(config['features'])
        self.features_plus = 2
        self.net = nn.Sequential(
            nn.Linear(self.features + self.features_plus, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).double()

        for model in self.modules():
            if isinstance(model, nn.Linear):
                nn.init.xavier_uniform_(model.weight)
                nn.init.constant_(model.bias, 0.1)

    def forward(self, features, action):

        features = features.view(-1, self.features).to(device=d)
        action = action.view(-1, self.features_plus).to(device=d)

        x = torch.cat((features, action), 1).double()
        return self.net(x)

from torch._C import device
import torch.nn as nn
import torch


d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiLayerPolicy(nn.Module):
    """An MLP based policy network"""

    def __init__(self, config):
        super(MultiLayerPolicy, self).__init__()

        self.features = len(config['features'])
        self.net = nn.Sequential(
            nn.Linear(self.features + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh()
        ).double()

        for model in self.modules():
            if isinstance(model, nn.Linear):
                nn.init.xavier_uniform_(model.weight)
                nn.init.constant_(model.bias, 0.1)

    def forward(self, features, command):

        command = torch.tensor(command).view(-1, 1).to(device=d)
        features = features.view(-1, self.features).to(device=d)

        out = torch.cat((features, command), 1)
        out = self.net(out.to(device=d)).view(-1, 2)
        return out

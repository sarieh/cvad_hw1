import torch.nn as nn
from torchvision import models
import torch

IN_LEN = 1
MID_LEN = 128
MID_LEN_2 = 256
FET_LEN = 512
COMMAND_LEN = 3


class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""

    def __init__(self):
        super(AffordancePredictor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.perception = nn.Sequential(*(list(resnet.children())[:-1]))

        self.lane_task = nn.ModuleList([TaskPredictor(FET_LEN + 1) for i in range(4)])  # 4 branches
        self.angle_task = nn.ModuleList([TaskPredictor(FET_LEN + 1) for i in range(4)])  # 4 branches

        self.light_distance_task = TaskPredictor(FET_LEN)
        self.light_existence_task = TaskPredictor(FET_LEN, out_dim=2, outputState=True)  # Binary classification

    def forward(self, img, command):
        perception = self.perception(img).reshape(-1, FET_LEN)
        command = command.reshape(-1, 1)

        result = {}

        result["lane_task"] = [model(perception, command) for model in self.lane_task]
        result["angle_task"] = [model(perception, command) for model in self.angle_task]
        result["light_distance_task"] = self.light_distance_task(perception)
        result["light_existence_task"] = self.light_existence_task(perception)

        return result


class TaskPredictor(nn.Module):
    def __init__(self, in_dim, mid_dim=MID_LEN_2, mid_2_dim=MID_LEN, out_dim=IN_LEN, outputState=False):
        super(TaskPredictor, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_2_dim),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(mid_2_dim, out_dim)
        )

        self.softmax = nn.Softmax(dim=1)
        self.outputState = outputState

        for model in self.modules():
            if isinstance(model, nn.Linear):
                nn.init.xavier_uniform_(model.weight)
                nn.init.constant_(model.bias, 0.1)

    def forward(self, img, command=None):

        if command is not None:
            x = torch.cat((img, command), 1)
        else:
            x = img

        out = self.model(x)

        return out

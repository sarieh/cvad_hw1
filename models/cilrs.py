import torch.nn as nn
from torchvision import models
import torch
import torch.nn as nn

IN_LEN = 1
MID_LEN = 128
MID_LEN_2 = 256
FET_LEN = 512
COMMAND_LEN = 3

class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self):
        super(CILRS, self).__init__()

        self.speed_mes = nn.Sequential(
                            nn.Linear(IN_LEN, MID_LEN),
                            nn.ReLU(),
                            nn.Linear(MID_LEN, MID_LEN),
                            nn.ReLU(),
        )
 
        self.speed_pred = nn.Sequential( 
                            nn.Linear(FET_LEN, MID_LEN_2),
                            nn.ReLU(),
                            nn.Linear(MID_LEN_2, MID_LEN_2),
                            nn.Dropout(p=0.25),
                            nn.ReLU(),
                            nn.Linear(MID_LEN_2, 1),
        )
    
        self.after_joining = nn.Sequential( 
                            nn.Linear(FET_LEN + MID_LEN, FET_LEN),
                            nn.ReLU(),
        )
    
    
        self.speed_branches = nn.ModuleList([nn.Sequential(
                                    nn.Linear(FET_LEN, MID_LEN_2), 
                                    nn.ReLU(),
                                    nn.Linear(MID_LEN_2, MID_LEN_2), 
                                    nn.Dropout(p=0.25),
                                    nn.ReLU(),
                                    nn.Linear(MID_LEN_2, COMMAND_LEN)
                            ) for i in range(4)])

        for model in self.modules():
            if isinstance(model, nn.Linear):
                nn.init.xavier_uniform_(model.weight)
                nn.init.constant_(model.bias, 0.1)

        resnet = models.resnet18(pretrained=True)
        self.perception = nn.Sequential(*(list(resnet.children())[:-1]))

    def forward(self, img, command):
        perception_out = self.perception(img).view(-1, FET_LEN)
        # print("perception_out {}".format(perception_out.shape))
        
        speed_mes_out = self.speed_mes(command.view(-1, 1))
        # print("speed_mes_out {}".format(speed_mes_out.shape))
        
        joint = torch.cat((perception_out, speed_mes_out), 1)
        # print("joint {}".format(joint.shape))

        after_joining = self.after_joining(joint)
        # print("after_joining {}".format(after_joining.shape))
        
        speed_pred_out = self.speed_pred(perception_out)
        # print("speed_pred_out {}".format(speed_pred_out.shape))
        
        action_branche_out = [model(after_joining) for model in self.speed_branches]
        # print("action_branche_out {}".format(len(action_branche_out)))
                
        return speed_pred_out, action_branche_out
    
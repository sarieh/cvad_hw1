from operator import ipow
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor
from os.path import exists
from torch.optim.lr_scheduler import StepLR

cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    criterion = nn.MSELoss()

    model.eval()

    running_branch_loss = [0.0, 0.0]
    running_tld_loss = 0.0
    running_tls_loss = 0.0

    # Your code here
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        loss12 = [0.0, 0.0]
        loss3 = 0.0
        loss4 = 0.0

        img = data['rgb']
        command = data['command']

        result = model(img, command.to(cuda))

        lane_task = result["lane_task"]
        angle_task = result["angle_task"]
        light_distance_task = result["light_distance_task"]
        light_existence_task = result["light_existence_task"]

        gt = [data['lane_dist'].to(cuda),
              data['route_angle'].to(cuda),
              data['tl_dist'].to(cuda)[:, None],
              data['tl_state'].to(cuda)[:, None]]

        for gti, task in enumerate([lane_task, angle_task]):
            for b, branch in enumerate(task):
                command_branch = (torch.tensor(b) == command).to(device=cuda)

                branch = torch.squeeze(branch)
                command_branch = torch.squeeze(command_branch)

                l = branch[command_branch].size()
                if len(l) == 0 or l[0] == 0:
                    continue

                loss12[gti] = criterion(branch[command_branch], gt[gti][command_branch])

        loss3 = criterion(light_distance_task, gt[2])
        loss4 = criterion(light_existence_task, gt[3].reshape(light_existence_task.size()))

        running_branch_loss[0] += loss12[0]
        running_branch_loss[1] += loss12[1]
        running_tld_loss += loss3
        running_tls_loss += loss4

        if i % 80 == 0:
            print("{} branch_loss[0]: {}, branch_loss[1]: {}, tld_loss: {}, tls_loss: {}".format(i, loss12[0], loss12[1], loss3, loss4))

    return running_branch_loss[0], running_branch_loss[1], running_tld_loss, running_tls_loss


def train(model, dataloader, lr=0.01):
    """Train model on the training dataset for one epoch"""

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    running_branch_loss = [0.0, 0.0]
    running_tld_loss = 0.0
    running_tls_loss = 0.0

    # Your code here
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        loss = 0.0
        loss12 = [0.0, 0.0]
        loss3 = 0.0
        loss4 = 0.0

        img = data['rgb']
        command = data['command']

        optimizer.zero_grad()

        result = model(img, command.to(cuda))

        lane_task = result["lane_task"]
        angle_task = result["angle_task"]
        light_distance_task = result["light_distance_task"]
        light_existence_task = result["light_existence_task"]

        gt = [data['lane_dist'].to(cuda),
              data['route_angle'].to(cuda),
              data['tl_dist'].to(cuda)[:, None],
              data['tl_state'].to(cuda)[:, None]]

        for gti, task in enumerate([lane_task, angle_task]):
            for b, branch in enumerate(task):
                command_branch = (torch.tensor(b) == command).to(device=cuda)

                branch = torch.squeeze(branch)
                command_branch = torch.squeeze(command_branch)

                l = branch[command_branch].size()
                if len(l) == 0 or l[0] == 0:
                    continue

                loss12[gti] = criterion(branch[command_branch], gt[gti][command_branch])

        loss3 = criterion(light_distance_task, gt[2])
        loss4 = criterion(light_existence_task, gt[3].reshape(light_existence_task.size()))

        loss = loss12[0] + loss12[1] + loss3 + loss4

        loss.backward()
        optimizer.step()

        # print statistics
        running_branch_loss[0] += loss12[0]
        running_branch_loss[1] += loss12[1]
        running_tld_loss += loss3
        running_tls_loss += loss4

        if i % 80 == 0:
            print("{} branch_loss[0]: {}, branch_loss[1]: {}, tld_loss: {}, tls_loss: {}".format(i, loss12[0].item(), loss12[1].item(), loss3.item(), loss4.item()))

    return running_branch_loss[0], running_branch_loss[1], running_tld_loss, running_tls_loss


def plot_losses(train_loss, val_loss):

    train_lane = [a[0] for a in train_loss]
    train_angle = [a[1] for a in train_loss]
    train_light_distance = [a[2] for a in train_loss]
    train_light_state = [a[3] for a in train_loss]

    val_lane = [a[0] for a in val_loss]
    val_angle = [a[1] for a in val_loss]
    val_light_distance = [a[2] for a in val_loss]
    val_light_state = [a[3] for a in val_loss]

    train_losses = [train_lane, train_angle, train_light_distance, train_light_state]
    val_losses = [val_lane, val_angle, val_light_distance, val_light_state]
    losses = ['lane_dist', 'route_angle', 'tl_dist', 'tl_state']
    for i, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
        """Visualize your plots and save them for your report."""
        plt.plot(t_loss)
        plt.plot(v_loss)
        plt.title(f"{losses[i]} loss")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend([f"train loss", f"test loss"], loc='upper left')
        plt.savefig(f"pred_{losses[i]}.png")
        plt.show()


def main():
    # Change these paths to the correct paths in your downloaded expert dataset

    train_root = "/userfiles/eozsuer16/expert_data/train/"
    val_root = "/userfiles/eozsuer16/expert_data/val"

    train_root = "/home/sarieh/Documents/expert_data/train/"
    # val_root = "/home/sarieh/Documents/expert_data/val/"

    model = AffordancePredictor()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 20
    batch_size = 4
    save_path = "pred_model.ckpt"

    model.cuda()

    if(exists(save_path)):
        model = torch.load(save_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []

    for i in range(num_epochs):
        f = open("pred_loss_last.txt", "a")
        train_losses.append(train(model, train_loader, 0.05 * pow(0.8, i)))
        val_losses.append(validate(model, val_loader))
        f.write("(train_losses: {} - val_losses: {})\n".format(train_losses[-1], val_losses[-1]))
        f.close()
        torch.save(model, save_path)
        model = torch.load(save_path)

    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()

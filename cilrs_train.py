import torch
from torch.utils.data import DataLoader

from expert_dataset import ExpertDataset
from models.cilrs import CILRS
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from os.path import exists
from torch.optim.lr_scheduler import StepLR
from math import pow

cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate(model, dataloader):
    """Validate model performance on the validation dataset"""

    criterion = nn.MSELoss()

    # Your code here
    running_speed_loss = 0.0
    running_steer_loss = 0.0
    running_throttle_loss = 0.0
    running_brake_loss = 0.0

    model.eval()

    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        loss1 = 0.0
        loss2 = 0.0
        loss3 = 0.0

        img = data['rgb']
        command = data['command'][:, None]

        speed_gt = data['speed'][:, None]  # [64]

        steer_gt = data['steer'][:, None]  # [64]
        throttle_gt = data['throttle'][:, None]  # [64]
        brake_gt = data['brake'][:, None]  # [64]

        speed_out, branch_out = model(img, speed_gt)

        for b, branch in enumerate(branch_out):

            command_branch = (torch.tensor(b) == command).to(device=cuda)

            branch = torch.squeeze(branch)
            command_branch = torch.squeeze(command_branch)

            loss1 += criterion(branch[:, 0] * command_branch, steer_gt * command_branch)
            loss2 += criterion(branch[:, 1] * command_branch, throttle_gt * command_branch)
            loss3 += criterion(branch[:, 2] * command_branch, brake_gt * command_branch)

        loss4 = criterion(speed_out, speed_gt.reshape(speed_out.size()))

        running_steer_loss = loss1.item()
        running_throttle_loss = loss2.item()
        running_brake_loss = loss3.item()
        running_speed_loss = loss4.item()

        if i % 80 == 0:
            print("{} steer_loss {} throttle_loss {} brake_loss {} speed_loss {}".format(i, loss1.item(), loss2.item(), loss3.item(), loss4.item()))

    return running_steer_loss, running_throttle_loss, running_brake_loss, running_speed_loss


def train(model, dataloader, lr=0.01):
    """Train model on the training dataset for one epoch"""

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    # Your code here
    running_speed_loss = 0.0
    running_steer_loss = 0.0
    running_throttle_loss = 0.0
    running_brake_loss = 0.0

    for i, data in enumerate(dataloader, 0):
        loss = 0.0
        loss1, loss2 = 0.0, 0.0
        loss3, loss4 = 0.0, 0.0

        img = data['rgb']
        command = data['command']

        speed_gt = data['speed']  # [64]

        steer_gt = data['steer']  # [64]
        throttle_gt = data['throttle']  # [64]
        brake_gt = data['brake']  # [64]

        optimizer.zero_grad()

        speed_out, branch_out = model(img, speed_gt)

        for b, branch in enumerate(branch_out):

            command_branch = (torch.tensor(b) == command).to(device=cuda)

            branch = torch.squeeze(branch)
            command_branch = torch.squeeze(command_branch)

            loss1 += criterion(branch[:, 0] * command_branch, steer_gt * command_branch)
            loss2 += criterion(branch[:, 1] * command_branch, throttle_gt * command_branch)
            loss3 += criterion(branch[:, 2] * command_branch, brake_gt * command_branch)

        loss4 = criterion(speed_out, speed_gt.view(speed_out.size()))

        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        optimizer.step()

        running_steer_loss = loss1.item()
        running_throttle_loss = loss2.item()
        running_brake_loss = loss3.item()
        running_speed_loss = loss4.item()

        if i % 80 == 0:
            print("{} train_steer_loss {} train_throttle_loss {} train_brake_loss {} train_speed_loss {}".format(i, loss1.item(), loss2.item(), loss3.item(), loss4.item()))

    return running_steer_loss, running_throttle_loss, running_brake_loss, running_speed_loss


def plot_losses(train_loss, val_loss):
    train_steer = [a[0] for a in train_loss]
    train_throttle = [a[1] for a in train_loss]
    train_brake = [a[2] for a in train_loss]
    train_speed = [a[3] for a in train_loss]

    val_steer = [a[0] for a in val_loss]
    val_throttle = [a[1] for a in val_loss]
    val_brake = [a[2] for a in val_loss]
    val_speed = [a[3] for a in val_loss]

    train_losses = [train_steer, train_throttle, train_brake, train_speed]
    val_losses = [val_steer, val_throttle, val_brake, val_speed]
    losses = ['steer', 'throttle', 'brake', 'speed']
    for i, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses)):
        """Visualize your plots and save them for your report."""
        plt.plot(t_loss)
        plt.plot(v_loss)
        plt.title(f"{losses[i]} loss")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend([f"train loss", f"val loss"], loc='upper left')
        plt.savefig(f"cilrs_{losses[i]}.png")
        plt.show()


def main():
    # Change these paths to the correct paths in your downloaded expert dataset

    train_root = "/userfiles/eozsuer16/expert_data/train/"
    val_root = "/userfiles/eozsuer16/expert_data/val/"

    # train_root = "/home/sarieh/Documents/expert_data/train/"
    # val_root = "/home/sarieh/Documents/expert_data/val/"

    model = CILRS()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 20
    batch_size = 16
    save_path = "cilrs_model.ckpt"

    if(exists(save_path)):
        model = torch.load(save_path)

    model.cuda()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []

    for i in range(num_epochs):
        train_losses.append(train(model, train_loader, 0.1 * pow(0.8, i)))
        val_losses.append(validate(model, val_loader))
        torch.save(model, save_path)

    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()

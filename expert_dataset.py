from tkinter.tix import InputOnly
from torch.utils.data import Dataset
import os
import json
from PIL import Image
from torchvision import transforms
import torch
from torch.nn.functional import one_hot as hot

Image.LOAD_TRUNCATED_IMAGES = True


class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""

    def __init__(self, data_root, use_cuda=True):
        self.data_root = data_root
        self.images_root = os.path.join(self.data_root, "rgb")
        self.measurements_root = os.path.join(self.data_root, "measurements")
        self.images = sorted(os.listdir(self.images_root))
        self.measurements = sorted(os.listdir(self.measurements_root))
        self.transforms = transforms.ToTensor()
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    def __getitem__(self, index):
        """Return RGB images and measurements"""
        m_path = open(os.path.join(self.measurements_root, self.measurements[index]))
        i_path = os.path.join(self.images_root, self.images[index])

        item = json.load(m_path)
        image = Image.open(i_path)
        m_path.close()

        item['rgb'] = self.transforms(image).to(device=self.device)

        item['speed'] = torch.tensor(item['speed'], dtype=torch.float).to(device=self.device)
        item['throttle'] = torch.tensor(item['throttle'], dtype=torch.float).to(device=self.device)
        item['steer'] = torch.tensor(item['steer'], dtype=torch.float).to(device=self.device)
        item['brake'] = torch.tensor(item['brake'], dtype=torch.float).to(device=self.device)

        item['lane_dist'] = torch.tensor(item['lane_dist'], dtype=torch.float).to(device=self.device)
        item['route_angle'] = torch.tensor(item['route_angle'], dtype=torch.float).to(device=self.device)
        item['tl_dist'] = torch.tensor(item['tl_dist'], dtype=torch.float).to(device=self.device)
        item['tl_state'] = hot(torch.tensor(item['tl_state']), num_classes=2).float().to(device=self.device)

        return item

    def __len__(self):
        return len(self.images)

import os

import yaml
from models.cilrs import CILRS
from carla_env.env import Env
import torch
from torchvision import transforms


class Evaluator():
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.agent = self.load_agent()
        self.transforms = transforms.ToTensor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initial = True

    def load_agent(self):
        model = CILRS()
        model = torch.load("cilrs_model.ckpt")
        model.eval()
        model.cuda()
        return model

    def generate_action(self, rgb, command, speed):
        rgb = self.transforms(rgb).to(device=self.device)[None, :]
        speed = torch.tensor(speed, dtype=torch.float).to(device=self.device)

        _, actions = self.agent(rgb, speed)
        steer, throttle, brake = actions[command].tolist()[0]
        return steer, throttle, brake

    def take_step(self, state):
        rgb = state["rgb"]
        command = state["command"]
        speed = state["speed"]
        steer, throttle, brake = self.generate_action(rgb, command, speed)

        steer = self.adjust(steer, -1.)
        throttle = self.adjust(throttle)
        brake = self.adjust(brake)

        action = {
            "steer": steer,
            "throttle": throttle,
            "brake":  brake
        }
        state, reward_dict, is_terminal = self.env.step(action)
        return state, is_terminal

    def evaluate(self, num_trials=100):
        terminal_histogram = {}
        for i in range(num_trials):
            state, _, is_terminal = self.env.reset()
            for i in range(5000):
                if is_terminal:
                    break
                state, is_terminal = self.take_step(state)
            if not is_terminal:
                is_terminal = ["timeout"]
            terminal_histogram[is_terminal[0]] = (terminal_histogram.get(is_terminal[0], 0)+1)
        print("Evaluation over. Listing termination causes:")
        for key, val in terminal_histogram.items():
            print(f"{key}: {val}/100")

    def adjust(self, a, mn=0.0, mx=1.0):
        a = max(min(mx, a), mn)
        return int(a * 10) / 10.0


def main():
    with open(os.path.join("configs", "cilrs.yaml"), "r") as f:
        config = yaml.full_load(f)

    with Env(config) as env:
        evaluator = Evaluator(env, config)
        evaluator.evaluate()


if __name__ == "__main__":
    main()

import numpy as np
from sympy import Q
import torch
from torch._C import device
from func_timeout import FunctionTimedOut, func_timeout
from utils.rl_utils import generate_noisy_action_tensor

from .off_policy import BaseOffPolicy
import torch.nn.functional as F

d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TD3(BaseOffPolicy):
    def _compute_q_loss(self, data):
        """Compute q loss for given batch of data."""

        features = data[0]
        command = data[1]
        actions = data[2].view(-1, 2)
        reward = data[3]
        new_features = torch.cat([a.view(-1, 1) for a in data[4]], dim=1)
        new_command = data[5]
        is_terminal = data[6]

        discount = torch.tensor(self.config["discount"]).to(device=d)

        # p_noise = self.config['policy_noise']
        t_p_noise = self.config['target_policy_noise']
        # noise = (torch.randn_like(actions) * p_noise).clamp(-p_noise, p_noise).to(device=d)
        new_action = (self.target_policy(new_features, new_command) + actions.to(device=d)).clamp(-t_p_noise, t_p_noise)

        q_target = [q(new_features.to(device=d), new_action) for q in self.target_q_nets]
        target_q1, target_q2 = q_target[0], q_target[1]
        m_i_n = torch.min(target_q1, target_q2)

        target_q = reward.to(device=d) + is_terminal.to(device=d) * discount * m_i_n

        q = [q(new_features, actions) for q in self.q_nets]
        q1, q2 = q[0], q[1]

        q_loss = F.mse_loss(target_q, q1) + F.mse_loss(target_q, q2)

        return None, q_loss.mean()

    def _compute_p_loss(self, data):
        """Compute policy loss for given batch of data."""
        # Your code here.for datum in data:

        features = torch.cat([a.view(-1, 1) for a in data[0]], dim=1)
        command = data[1]
        actions = data[2].view(-1, 2)
        reward = data[3]
        new_features = torch.cat([a.view(-1, 1) for a in data[4]], dim=1)
        new_command = data[5]
        is_terminal = data[6]

        # Gradient ascent
        p_loss = -self.q_nets[0](features.to(device=d), self.policy(features, command))

        return p_loss.mean()

    def _extract_features(self, state):
        """Extract whatever features you wish to give as input to policy and q networks."""

        # Your code here
        keys = self.config['features']
        vals = [state[key] for key in keys]

        return torch.tensor(np.array(vals)).double()  # size([17])

    def _take_step(self, state, action):
        try:
            action_dict = {
                "throttle": np.clip(action[0, 0].item(), 0, 1),
                "brake": abs(np.clip(action[0, 0].item(), -1, 0)),
                "steer": np.clip(action[0, 1].item(), -1, 1),
            }
            new_state, reward_dict, is_terminal = func_timeout(
                20, self.env.step, (action_dict,))
        except FunctionTimedOut:
            print("\nEnv.step did not return.")
            raise
        return new_state, reward_dict, is_terminal

    def _collect_data(self, state):
        """Take one step and put data into the replay buffer."""
        features = self._extract_features(state)
        if self.step >= self.config["exploration_steps"]:
            action = self.policy(features, [state["command"]])
            action = generate_noisy_action_tensor(
                action, self.config["action_space"], self.config["policy_noise"], 1.0)
        else:
            action = self._explorer.generate_action(state)
        if self.step <= self.config["augment_steps"]:
            action = self._augmenter.augment_action(action, state)

        # Take step
        new_state, reward_dict, is_terminal = self._take_step(state, action)

        new_features = self._extract_features(state)
        # Prepare everything for storage
        stored_features = [f.detach().cpu().squeeze(0) for f in features]
        stored_command = state["command"]
        stored_action = action.detach().cpu().squeeze(0)
        stored_reward = torch.tensor([reward_dict["reward"]], dtype=torch.float)
        stored_new_features = [f.detach().cpu().squeeze(0) for f in new_features]
        stored_new_command = new_state["command"]
        stored_is_terminal = bool(is_terminal)

        self._replay_buffer.append(
            (stored_features, stored_command, stored_action, stored_reward,
             stored_new_features, stored_new_command, stored_is_terminal)
        )
        self._visualizer.visualize(new_state, stored_action, reward_dict)
        return reward_dict, new_state, is_terminal

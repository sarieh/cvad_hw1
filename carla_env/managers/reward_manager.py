import math
from time import sleep


class RewardManager():
    """Computes and returns rewards based on states and actions."""

    def __init__(self):
        pass

    def get_reward(self, state, action):
        """Returns the reward as a dictionary. You can include different sub-rewards in the
        dictionary for plotting/logging purposes, but only the 'reward' key is used for the
        actual RL algorithm, which is generated from the sum of all other rewards."""
        reward_dict = {}
        # Your code here

        # actions
        brake = action["brake"]
        steer = action["steer"]
        throttle = action["throttle"]

        # states
        optimal_speed = state["optimal_speed"]
        speed = state['speed']
        tl_state = state['tl_state']
        tl_dist = state['tl_dist']
        is_junction = state['is_junction']

        route_dist = state['route_dist']
        route_angle = state['route_angle']
        command = state['command']

        # other states
        lane_angle = state["lane_angle"]
        lane_dist = state["lane_dist"]

        collision = state["collision"]

        hazard = state["hazard"]
        hazard_dist = state["hazard_dist"]
        hazard_coords = state["hazard_coords"]

        waypoint_dist = state["waypoint_dist"]
        waypoint_angle = state["waypoint_angle"]

        # print("-" * 30)
        # print(f"{int(lane_angle * 1000.0)/1000.0} {int(route_angle * 1000.0)/1000.0} {int(waypoint_angle * 1000.0)/1000.0}")
        # print(f"{int(lane_dist * 1000.0)/1000.0} {int(route_dist * 1000.0)/1000.0} {int(waypoint_dist * 1000.0)/1000.0}")

        is_junction_reward = 1.2 if is_junction else 1.0

        reward_dict["speed_limit"] = 0.0
        reward_dict["steering_route_angle"] = 0.0
        reward_dict["tl_reward"] = 0.0
        reward_dict["waypoint_angle"] = 0.0
        reward_dict["lane_dist"] = 0.0
        reward_dict["lane_angle"] = 0.0
        reward_dict['collision'] = 0.0

        # speed_limit
        a = self.abs_tanh((abs(speed) - abs(optimal_speed)) / 2.0)
        reward_dict["speed_limit"] = self.check_reward(a, 0.05)

        if reward_dict["speed_limit"] < 0:
            reward_dict["speed_limit"] = reward_dict["speed_limit"] * is_junction_reward

        if reward_dict["speed_limit"] < 0:
            if speed < optimal_speed:
                reward_dict["speed_limit"] = reward_dict["speed_limit"] * (1.0 / (throttle + 0.001))
                if brake > 0.01:
                    reward_dict["speed_limit"] = reward_dict["speed_limit"] * (math.pow(math.e, brake) - 1.0)
            else:
                reward_dict["speed_limit"] = reward_dict["speed_limit"] * (1.0 + throttle)

        # steering_route_angle
        if(lane_angle * steer < 0):
            reward_dict["steering_route_angle"] = -self.abs_tanh(abs(lane_angle) + abs(steer) + 0.1) * is_junction_reward

        if reward_dict["speed_limit"] < 0 and tl_state:
            c = abs(math.pow(math.e, 0.3*tl_dist)) * is_junction_reward
            reward_dict["tl_reward"] = self.check_reward(c, 0.1) * is_junction_reward

        if command == 3:
            if abs(waypoint_angle) > 0.1:
                reward_dict["waypoint_angle"] = - 1.5 * math.pow(math.e, waypoint_angle)

        if abs(lane_dist) > 2.0:
            reward_dict["lane_dist"] = 1.5 * -lane_dist * is_junction_reward

        if abs(waypoint_dist) > 2.0:
            reward_dict["waypoint_dist"] = -1.5 * waypoint_dist

        reward_dict["lane_angle"] = -self.abs_tanh(2 * lane_angle * lane_dist) * is_junction_reward

        if collision:
            reward_dict['collision'] = -2

        reward = 0.0

        keys = list(reward_dict.keys())
        for i, val in enumerate(reward_dict.values()):
            # print(f"{keys[i]}:{int(val * 100) / 100}", end=" -- ")
            reward = self.clip(val)
        # print(flush=True)
        reward_dict["reward"] = reward
        return reward_dict

    def check_reward(self, tan, threshold):
        return 1.0 if tan < threshold else -tan

    def clip(self, a, mn=-5.0, mx=5.0):
        return max(min(mx, a), mn)

    def abs_tanh(self, a):
        return abs(math.tan(a))

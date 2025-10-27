import numpy as np
import gymnasium as gym
from gymnasium import spaces

class QueueEnvironment(gym.Env):
    """A simple queue simulation environment for scheduling optimization."""

    def __init__(self):
        super(QueueEnvironment, self).__init__()

        # Example setup: 8-hour day, max queue length = 20
        self.max_hours = 8
        self.max_queue = 20

        # Observation space: [current hour, queue length]
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([self.max_hours, self.max_queue], dtype=np.float32),
            dtype=np.float32
        )

        # Action space: 0 = idle, 1 = serve one student
        self.action_space = spaces.Discrete(2)

        # Internal state
        self.hour = 0
        self.queue = 0

    def reset(self, seed=None, options=None):
        """Reset environment at the start of a new episode."""
        super().reset(seed=seed)
        self.hour = 0
        self.queue = np.random.randint(0, 5)
        obs = np.array([self.hour, self.queue], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        """Take one action step."""
        arrivals = np.random.randint(0, 4)  # 0–3 students arrive
        reward = 0

        if action == 1 and self.queue > 0:
            self.queue -= 1
            reward = 1  # reward for serving
        else:
            reward = -0.1  # penalty for idling

        # Add new arrivals
        self.queue = min(self.queue + arrivals, self.max_queue)

        # Advance time
        self.hour += 1
        terminated = self.hour >= self.max_hours
        truncated = False

        obs = np.array([self.hour, self.queue], dtype=np.float32)
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Hour: {self.hour}, Queue: {self.queue}")

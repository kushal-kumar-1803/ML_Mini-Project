import numpy as np
import gymnasium as gym
from gymnasium import spaces


class QueueEnvironment(gym.Env):
    """A simple queue simulation for scheduling optimization."""

    def __init__(self):
        super(QueueEnvironment, self).__init__()

        # Example: 8 hours per day, max queue length 20
        self.max_hours = 8
        self.max_queue = 20

        # Observation space: [hour, queue_length]
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.max_hours, self.max_queue]),
            dtype=np.float32
        )

        # Action space: 0 = idle, 1 = serve one student
        self.action_space = spaces.Discrete(2)

        # Internal state
        self.hour = 0
        self.queue = 0
        self.state = np.array([self.hour, self.queue], dtype=np.float32)

    def reset(self):
        """Reset environment at the start of a new episode."""
        self.hour = 0
        self.queue = np.random.randint(0, 5)
        self.state = np.array([self.hour, self.queue], dtype=np.float32)
        return self.state

    def step(self, action):
        """Take one action step."""
        # Random arrivals between 0–3 students per hour
        arrivals = np.random.randint(0, 4)
        reward = 0

        if action == 1 and self.queue > 0:
            # Served one student
            self.queue -= 1
            reward = 1
        else:
            reward = -0.1  # Penalty for idle time

        # Add new arrivals
        self.queue += arrivals
        self.queue = min(self.queue, self.max_queue)

        # Advance time
        self.hour += 1
        done = self.hour >= self.max_hours

        # Update and return new state
        self.state = np.array([self.hour, self.queue], dtype=np.float32)
        return self.state, reward, done, {}

    def render(self, mode="human"):
        print(f"Hour: {self.hour}, Queue: {self.queue}")

    def _get_state(self):
        """Return the current environment state."""
        return np.array(self.state, dtype=np.float32)

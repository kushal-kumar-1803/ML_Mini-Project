import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from rl.env_queue import QueueEnvironment
import random
from collections import deque


# --- Hyperparameters ---
EPISODES = 10
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 2000


# --- Build DQN model ---
def build_dqn(state_size, action_size):
    model = Sequential([
        Input(shape=(state_size,)),
        Dense(24, activation='relu'),
        Dense(24, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
    return model


# --- Initialize Environment ---
env = QueueEnvironment()
state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
state_size = len(state)
action_size = env.action_space.n

model = build_dqn(state_size, action_size)
memory = deque(maxlen=MEMORY_SIZE)


# --- Training Loop ---
for e in range(EPISODES):
    reset_output = env.reset()
    state = reset_output[0] if isinstance(reset_output, tuple) else reset_output
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(env.max_hours):
        # ε-greedy policy
        if np.random.rand() <= EPSILON:
            action = random.randrange(action_size)
        else:
            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])

        # Take step — handle 4 or 5 return values
        step_output = env.step(action)
        if len(step_output) == 5:
            next_state, reward, terminated, truncated, _ = step_output
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_output

        next_state = np.reshape(next_state, [1, state_size])
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode {e + 1}/{EPISODES} — Reward: {total_reward:.2f}, Epsilon: {EPSILON:.3f}")
            break

        # Experience replay
        if len(memory) > BATCH_SIZE:
            minibatch = random.sample(memory, BATCH_SIZE)
            for s, a, r, s_next, d in minibatch:
                target = r
                if not d:
                    target += GAMMA * np.amax(model.predict(s_next, verbose=0)[0])
                target_f = model.predict(s, verbose=0)
                target_f[0][a] = target
                model.fit(s, target_f, epochs=1, verbose=0)

    # Epsilon decay
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY


# --- Save trained model ---
model.save('models/dqn_scheduler.keras')
print("✅ DQN model trained and saved as models/dqn_scheduler.keras")

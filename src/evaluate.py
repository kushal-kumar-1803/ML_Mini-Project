import numpy as np
from tensorflow.keras.models import load_model
from rl.env_queue import QueueEnvironment

# ✅ Use the correct model path and format
MODEL_PATH = "models/dqn_scheduler.keras"

# ✅ Load trained model safely
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Could not load model at {MODEL_PATH}: {e}")
    exit()

# ✅ Initialize environment
env = QueueEnvironment()
state = env.reset()

total_reward = 0
STEPS = 50

print(f"\n🎯 Starting evaluation for {STEPS} steps...\n")

# ✅ Evaluation loop
for step in range(STEPS):
    # Predict Q-values for the current state
    q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)
    action = np.argmax(q_values[0])  # Select the best action
    next_state, reward, done, info = env.step(action)

    total_reward += reward
    state = next_state

    print(f"Step {step+1:02d} | Action: {action} | Reward: {reward:.2f} | Total: {total_reward:.2f}")

    if done:
        print("⚡ Episode ended early — resetting environment.\n")
        state = env.reset()

# ✅ Summary
print(f"\n✅ Evaluation complete.")
print(f"🏁 Total Reward over {STEPS} steps: {total_reward:.2f}")
print(f"⭐ Average Reward per step: {total_reward / STEPS:.2f}")

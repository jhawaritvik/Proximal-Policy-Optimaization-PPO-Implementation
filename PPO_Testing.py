import torch
import numpy as np
from PPO_Environment import Environment      # <- adjust if file is named differently
from PPO_Network import PolicyNetwork          # <- your policy network definition

# --- Hyperparameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "ReinforcementLearning\\PPO_saved_models\\policy_net_v4_v4.pt"  # Path to your saved model
max_test_episodes = 5

# --- Load environment and model ---
env = Environment()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

policy_net = PolicyNetwork(state_dim, action_dim).to(device)
policy_net.load_state_dict(torch.load(model_path, map_location=device))
policy_net.eval()

# --- Run test episodes ---
for ep in range(max_test_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, std = policy_net(state_tensor)
            mean = mean.cpu().numpy()[0]
            action = np.clip(mean, env.action_space.low, env.action_space.high)

        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated


    print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}, Ended due to: {info['reason']}")

env.close()

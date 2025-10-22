import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from tqdm import trange
import random

# Import headless environment
from geometry_dash_env import HeadlessGeometryDashEnv as GeometryDashEnv, STATE_SIZE, ACTION_SIZE

# --- Hyperparameters ---
NUM_ENVS = 8              # run 8 games in parallel
BATCH_SIZE = 256           # batch for training
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LR = 0.001
MAX_EPISODES = 1000
MAX_STEPS = 1000
TRAIN_EVERY = 4
MEMORY_CAPACITY = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DQN model ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)

# --- Replay buffer ---
class ReplayMemory:
    def __init__(self, capacity=MEMORY_CAPACITY):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.stack(states), dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(np.stack(next_states), dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device)
        )

    def __len__(self):
        return len(self.memory)

# --- Training loop ---
def train():
    # Create vector of environments
    envs = [GeometryDashEnv(render=False) for _ in range(NUM_ENVS)]
    states = np.stack([env.reset() for env in envs])  # shape (NUM_ENVS, STATE_SIZE)

    dqn = DQN(STATE_SIZE, ACTION_SIZE).to(device)
    optimizer = optim.Adam(dqn.parameters(), lr=LR)
    memory = ReplayMemory()
    epsilon = EPSILON_START

    pbar = trange(MAX_EPISODES, desc="Episodes", leave=True, mininterval=0.5, dynamic_ncols=False)
    for episode in pbar:
        total_rewards = np.zeros(NUM_ENVS)
        dones = np.zeros(NUM_ENVS, dtype=bool)

        for step in range(MAX_STEPS):
            # --- epsilon-greedy actions ---
            if np.random.rand() < epsilon:
                actions = np.random.randint(ACTION_SIZE, size=NUM_ENVS)
            else:
                states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
                with torch.no_grad():
                    actions = dqn(states_tensor).argmax(1).cpu().numpy()

            # Step environments in parallel
            next_states = []
            rewards = []
            new_dones = []
            for i, env in enumerate(envs):
                if not dones[i]:
                    ns, r, done, _ = env.step(actions[i])
                else:
                    ns, r, done = states[i], 0.0, True  # keep old state if done
                next_states.append(ns)
                rewards.append(r)
                new_dones.append(done)
                total_rewards[i] += r
            next_states = np.stack(next_states)
            dones = np.array(new_dones)

            # Push all transitions to replay buffer
            for i in range(NUM_ENVS):
                memory.push((states[i], actions[i], rewards[i], next_states[i], float(dones[i])))
            states = next_states

            # --- Training ---
            if len(memory) >= BATCH_SIZE and step % TRAIN_EVERY == 0:
                s, a, r, ns, d = memory.sample(BATCH_SIZE)
                q_values = dqn(s).gather(1, a.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    q_next = dqn(ns).max(1)[0]
                target = r + GAMMA * q_next * (1 - d)
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # break if all done
            if dones.all():
                break

        # epsilon decay
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Update in-place progress bar with summary stats
        avg_reward = float(total_rewards.mean())
        try:
            pbar.set_postfix({"AvgReward": f"{avg_reward:.1f}", "Epsilon": f"{epsilon:.3f}", "Steps": step+1})
        except Exception:
            print(f"Episode {episode}, Avg Reward: {avg_reward:.1f}, Epsilon: {epsilon:.3f}")

if __name__ == "__main__":
    train()

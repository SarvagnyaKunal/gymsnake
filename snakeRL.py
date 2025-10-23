import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import pygame
import torch.nn.functional as F
import time
from snake_game import SnakeEnv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class DQNNet(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=512, output_dim=4):  # Changed from 8 to 12
        super().__init__()
        # Deeper network with residual connections for better learning
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        
        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.ln4 = nn.LayerNorm(hidden_dim // 4)
        
        # Dueling DQN architecture: separate value and advantage streams
        self.value_stream = nn.Linear(hidden_dim // 4, 1)
        self.advantage_stream = nn.Linear(hidden_dim // 4, output_dim)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        x = F.relu(self.ln4(self.fc4(x)))
        
        # Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

class ReplayBuffer:
    def __init__(self, max_size=50000):  # Increased buffer size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (torch.tensor(s, dtype=torch.float32, device=device),
                torch.tensor(a, dtype=torch.long, device=device),
                torch.tensor(r, dtype=torch.float32, device=device),
                torch.tensor(ns, dtype=torch.float32, device=device),
                torch.tensor(d, dtype=torch.float32, device=device))
    
    def __len__(self):
        return len(self.buffer)

env = SnakeEnv()
dqn = DQNNet().to(device)
target_dqn = DQNNet().to(device)
target_dqn.load_state_dict(dqn.state_dict())
target_dqn.eval()

# Use Adam with weight decay for regularization
optimizer = optim.Adam(dqn.parameters(), lr=3e-4, weight_decay=1e-5)
# Learning rate scheduler for better convergence
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

replay = ReplayBuffer()

gamma = 0.995  # Higher discount for very long-term planning
epsilon = 1.0
epsilon_min = 0.05  # Keep some exploration
epsilon_decay = 0.9998  # Much slower decay for more exploration
batch_size = 64
update_target_every = 200  # Less frequent updates for stability
num_episodes = 30000
best_score = 0
moving_avg_reward = []
moving_avg_score = []


for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    steps = 0
    max_steps = 10000  # Prevent infinite loops

    while True:
        steps += 1
        
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension
                q_values = dqn(state_tensor)
                action = torch.argmax(q_values).item()
        
        # Environment step
        next_state, reward, done = env.step(action)
        
        # Add timeout penalty to prevent wandering
        if steps >= max_steps:
            done = True
            reward -= 5.0
        
        replay.push(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state

        # Train network more frequently
        if len(replay) >= batch_size * 2:  # Start training after enough samples
            s, a, r, ns, d = replay.sample(batch_size, device)

            # Double DQN: use online network to select action, target network to evaluate
            q_vals = dqn(s).gather(1, a.unsqueeze(1)).squeeze()

            with torch.no_grad():
                # Select best action using online network
                next_actions = dqn(ns).max(1)[1]
                # Evaluate using target network
                q_next = target_dqn(ns).gather(1, next_actions.unsqueeze(1)).squeeze()
                q_target = r + gamma * q_next * (1 - d)

            # Huber loss for robustness
            loss = F.smooth_l1_loss(q_vals, q_target)
            
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=10.0)
            optimizer.step()

        # Handle pygame events and render during gameplay
        if episode % 10 == 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            env.render()
        
        if done:
            break

    # Handle pygame events even when not rendering
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # Update target network
    if episode % update_target_every == 0 and episode > 0:
        target_dqn.load_state_dict(dqn.state_dict())

    # Epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Step learning rate scheduler
    scheduler.step()
    
    # Track moving average reward and score
    moving_avg_reward.append(total_reward)
    moving_avg_score.append(env.score)
    if len(moving_avg_reward) > 100:
        moving_avg_reward.pop(0)
        moving_avg_score.pop(0)
    avg_reward = np.mean(moving_avg_reward)
    avg_score = np.mean(moving_avg_score)

    # Save best model based on score
    if env.score > best_score:
        best_score = env.score
        torch.save(dqn.state_dict(), "best_snake_model.pth")
        print(f"🎯 New best score: {best_score}!")


    # Print progress - less frequent to avoid spam
    if episode % 50 == 0 or env.score > 10:
        print(f"Ep:{episode:5d} | Reward:{total_reward:7.2f} | AvgR:{avg_reward:6.2f} | Score:{env.score:3d} | AvgS:{avg_score:5.2f} | Eps:{epsilon:.4f} | Steps:{steps:4d}")

print("\n🎉 Training complete!")
print(f"Best score achieved: {best_score}")

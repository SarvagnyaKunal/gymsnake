import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from snake_game import SnakeEnv
import pygame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNNet(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=512, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)

        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.ln4 = nn.LayerNorm(hidden_dim // 4)

        self.value_stream = nn.Linear(hidden_dim // 4, 1)
        self.advantage_stream = nn.Linear(hidden_dim // 4, output_dim)

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

        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


def load_model(model_path, device):
    model = DQNNet().to(device)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def play(model, env, device, max_steps=10000, render=True, pause=0.0):
    state = env.reset()
    steps = 0
    total_reward = 0.0

    while True:
        steps += 1

        # handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q = model(s)
            action = int(torch.argmax(q, dim=1).item())

        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

        if render:
            env.render()
            if pause > 0.0:
                time.sleep(pause)

        if done or steps >= max_steps:
            break

    return total_reward, env.score, steps


def main():
    parser = argparse.ArgumentParser(description="Play saved snake model")
    parser.add_argument("--model", type=str, default="best_snake_model.pth", help="Path to model state dict")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering (headless) - still requires display backend for pygame) )")
    parser.add_argument("--pause", type=float, default=0.0, help="Pause seconds between frames to slow down play")
    args = parser.parse_args()

    env = SnakeEnv()
    try:
        model = load_model(args.model, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Device: {device} | Model: {args.model}")

    for ep in range(1, args.episodes + 1):
        result = play(model, env, device, render=(not args.no_render), pause=args.pause)
        if result is None:
            print("Exited by user.")
            break
        total_reward, score, steps = result
        print(f"Episode {ep}: Reward={total_reward:.2f} | Score={score} | Steps={steps}")

    print("Finished playing. Closing.")
    pygame.quit()


if __name__ == '__main__':
    main()

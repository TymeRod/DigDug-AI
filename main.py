import asyncio
import getpass
import json
import os
from queue import PriorityQueue
import websockets
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
import argparse
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_state = 48
y_state = 24
action_size = 5
discount_rate = 0.99
learning_rate = 5e-4
eps_start = 1
eps_end = 0.01
eps_decay = 500
time_step_reward = -1
dropout = 0.3
r_scaling = 2

# Epsilon decay function
def epsilon_by_episode(episode):
    return eps_end + (eps_start - eps_end) * math.exp(-1. * episode / eps_decay)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

        self.dropout = nn.Dropout(p=dropout)
        # Initialize weights using Kaiming He initialization for ReLU
        nn.init.kaiming_uniform_(self.net[0].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.net[2].weight, nonlinearity='relu')

    def forward(self, x):
        out = self.dropout(self.net(x))
        return out

class Game:
    def __init__(self, model):
        self.model = model.to(device)
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.digdug = None
        self.dir = 0

    def get_state(self, main_state):

        self.lives = main_state['lives'] if 'lives' in main_state.keys() else 0
        self.digdug = main_state['digdug'] if 'digdug' in main_state.keys() else self.digdug
        self.enemies = [enemy['pos'] for enemy in main_state['enemies']] if 'enemies' in main_state.keys() else []
        self.rocks = [rock['pos'] for rock in main_state['rocks']] if 'rocks' in main_state.keys() else []
        self.rope = [rop for rop in main_state['rope']['pos']] if 'rope' in main_state.keys() else []

        self.state[self.digdug[0]][self.digdug[1]] += 2
        
        for enemy in self.enemies:
            self.state[enemy[0]][enemy[1]] += 3

        for rock in self.rocks:
            self.state[rock[0]][rock[1]] += 4

        for rop in self.rope:
            self.state[rop[0]][rop[1]] += 5

        return self.state.flatten().unsqueeze(0).to(device)
    
    def get_next_state(self, main_state):
        self.next_state = self.state.clone()

        self.next_state[self.digdug[0]][self.digdug[1]] = 0
        
        for enemy in self.enemies:
            self.next_state[enemy[0]][enemy[1]] = 0
        
        for rock in self.rocks:
            self.next_state[rock[0]][rock[1]] = 0

        for rop in self.rope:
            self.next_state[rop[0]][rop[1]] = 0

        self.next_lives = main_state['lives'] if 'lives' in main_state.keys() else 0
        self.next_digdug = main_state['digdug'] if 'digdug' in main_state.keys() else self.digdug
        self.next_enemies = [enemy['pos'] for enemy in main_state['enemies']] if 'enemies' in main_state.keys() else []
        self.next_rocks = [rock['pos'] for rock in main_state['rocks']] if 'rocks' in main_state.keys() else []
        self.next_rope = [rop for rop in main_state['rope']['pos']] if 'rope' in main_state.keys() else []

        self.next_state[self.next_digdug[0]][self.next_digdug[1]] += 2
        
        for enemy in self.next_enemies:
            self.next_state[enemy[0]][enemy[1]] += 3

        for rock in self.next_rocks:
            self.next_state[rock[0]][rock[1]] += 4

        return self.next_state.flatten().unsqueeze(0).to(device)
    
    def closest_enemy(self, digdug, enemies):
        digdug = digdug
        min_distance = float('inf')
        min_enemy = None

        for enemy in enemies:
            distance = abs(digdug[0] - enemy[0]) + abs(digdug[1] - enemy[1])
            if distance < min_distance:
                min_distance = distance
                min_enemy = enemy

        return min_distance, min_enemy

    def check_out_of_bounds(self, digdug, move):
        return digdug[0] + move[0] < 0 or digdug[0] + move[0] > x_state or digdug[1] + move[1] < 0 or digdug[1] + move[1] > y_state

    def calculate_reward(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        reward = 0

        if action != 4:
            move = moves[action]

            # punish for trying to walk out of bounds
            if self.check_out_of_bounds(self.digdug, move):
                reward -= 1

            #reward for using tunnels
            if not self.check_out_of_bounds(self.digdug, move) and self.state[self.digdug[0] + move[0]][self.digdug[1] + move[1]] == 0:
                reward += 0.2

        # find the closest enemy
        dist_enemy, enemy = self.closest_enemy(self.digdug, self.enemies)
        dist_next_enemy, next_enemy = self.closest_enemy(self.next_digdug, self.next_enemies)

        delta_dist = dist_enemy - dist_next_enemy

        # reward for getting closer to the enemy
        if delta_dist > 0:
            reward += 1
        else:
            reward -= 1

        # reward for looking at the enemy while in range to attack
        if self.dir == 0 and self.digdug[0] == enemy[0] and self.digdug[1] < enemy[1] and (self.digdug[1] + 1 == enemy[1] or self.digdug[1] + 2 == enemy[1]):
            reward += 2
        elif self.dir == 1 and self.digdug[1] == enemy[1] and self.digdug[0] < enemy[0] and (self.digdug[0] + 1 == enemy[0] or self.digdug[0] + 2 == enemy[0]):
            reward += 2
        elif self.dir == 2 and self.digdug[0] == enemy[0] and self.digdug[1] > enemy[1] and (self.digdug[1] - 1 == enemy[1] or self.digdug[1] - 2 == enemy[0]):
            reward += 2
        elif self.dir == 3 and self.digdug[1] == enemy[1] and self.digdug[0] > enemy[0] and (self.digdug[0] - 1 == enemy[0] or self.digdug[0] - 2 == enemy[0]):
            reward += 2

        #punish for staying in the same place
        if self.digdug == self.next_digdug:
            reward -= 1

        #reward for hitting a enemy
        for rop in self.rope:
            if rop in self.next_enemies:
                reward += 100
                break
        
        #punish for dying
        if self.lives > self.next_lives:
            reward -= 100


        return reward
    
    def train_step(self, action, reward):
        next_state = self.next_state.view(-1, 48 * 24).to(device)
        state = self.state.view(-1, 48 * 24).to(device)
        action = action.view(1, -1).to(device)
        reward = torch.tensor([reward], device=device, dtype=torch.float32)

        # Compute state-action values
        state_action_values = self.model(state).gather(1, action)

        # Initialize next state values
        next_state_values = torch.zeros(1, device=device)

        # Update next state values if enemies exist
        if len(self.next_enemies) > 0:
            next_state_values = self.model(next_state).max(1)[0].detach()

        # Ensure discount_rate is defined
        expected_state_action_values = (next_state_values * discount_rate) + torch.tanh(reward.clone().detach().requires_grad_(False)) * r_scaling

        # Compute loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def get_key(self, action):
        actions = ["w", "s", "a", "d", "A"]

        if actions[action] == "w":
            self.dir = 0
        elif actions[action] == "d":
            self.dir = 1
        elif actions[action] == "s":
            self.dir = 2
        elif actions[action] == "a":
            self.dir = 3

        return actions[action]
    
def select_action(policy_net, state, episode):
    eps_threshold = epsilon_by_episode(episode)
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_size)]], dtype=torch.long).to(device)

async def agent_loop(server_address="localhost:8000", agent_name="student"):
    """Example client loop."""
    async with websockets.connect(f"ws://{server_address}/player") as websocket:
        # Receive information about static game properties
        await websocket.send(json.dumps({"cmd": "join", "name": agent_name}))

        episode = 1

        policy_net = DQN(x_state * y_state, action_size).to(device)
        game = Game(policy_net)
        if os.path.exists('weights/model-v0.pth'):
            policy_net.load_state_dict(torch.load('weights/model-v0.pth', map_location=device))
        else:
            print("Model weights file does not exist.")
        policy_net.train()

        total_reward = 0
        action = None
        while True:
            try:
                total_reward = 0
                state = json.loads(
                    await websocket.recv()
                )  # receive game update, this must be called timely or your game will get out of sync with the server

                if 'map' in state.keys():
                    game.state = torch.tensor(state['map'], dtype=torch.float32, device=device)
                    continue
                    
                if action != None:
                    next_state = game.get_next_state(state)
                    reward = game.calculate_reward(action.item())
                    total_reward += reward
                    game.train_step(action, reward)

                state = game.get_state(state)
                action = select_action(policy_net, state, episode)
                key = game.get_key(action.item())

                episode += 1
                await websocket.send(
                    json.dumps({"cmd": "key", "key": key})
                )

                next_state = json.loads(
                    await websocket.recv()
                )  # receive the next game state after your action

                next_state = game.get_next_state(next_state)
                reward = game.calculate_reward(action.item())
                total_reward += reward
                game.train_step(action, reward)

                action = select_action(policy_net, next_state, episode)
                key = game.get_key(action.item())

                episode += 1
                
                print(f"Episode {episode}: Total Reward: {total_reward:.2f}, Epsilon: {epsilon_by_episode(episode):.2f}")

                await websocket.send(
                    json.dumps({"cmd": "key", "key": key})
                )

            except websockets.exceptions.ConnectionClosedOK:
                torch.save(policy_net.state_dict(), 'weights/model-v0.pth')
                print('Model saved')
                print("Server has cleanly disconnected us")
                return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", help="IP address to bind to", default="")
    parser.add_argument("--port", help="TCP port", type=int, default=8000)
    parser.add_argument("--seed", help="Seed number", type=int, default=0)
    parser.add_argument("--debug", help="Open Bitmap with map on gameover", action='store_true')

    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    SERVER = os.environ.get("SERVER", "localhost")
    PORT = os.environ.get("PORT", args.port)
    NAME = os.environ.get("NAME", getpass.getuser())
    loop.run_until_complete(agent_loop(f"{SERVER}:{PORT}", NAME))
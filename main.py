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


device = 'cpu'

x_state = 48
y_state = 24
action_size = 5
discount_rate = 0.8
learning_rate = 5e-4
eps_start = 1
eps_end = 0.01
eps_decay = 2000
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
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.digdug = None


    def get_state(self, main_state):
        

        self.digdug = main_state['digdug']
        self.enemies = [enemy['pos'] for enemy in main_state['enemies']] if 'enemies' not in main_state else []
        self.rocks = [rock['pos'] for rock in main_state['rocks']] if 'rocks' not in main_state else []

        self.state[self.digdug[0]][self.digdug[1]] = 2
        
        for enemy in self.enemies:
            self.state[enemy[0]][enemy[1]] = 3

        for rock in self.rocks:
            self.state[rock[0]][rock[1]] = 4

        return self.state.flatten().unsqueeze(0)
    
    def get_next_state(self, main_state):
        self.next_state = self.state.clone()

        self.next_state[self.digdug[0]][self.digdug[1]] = 0
        
        for enemy in self.enemies:
            self.next_state[enemy[0]][enemy[1]] = 0
        
        for rock in self.rocks:
            self.next_state[rock[0]][rock[1]] = 0

        self.next_digdug = main_state['digdug']if 'digdug' in main_state else self.digdug
        self.next_enemies = [enemy['pos'] for enemy in main_state['enemies']] if 'enemies' not in main_state else []
        self.next_rocks = [rock['pos'] for rock in main_state['rocks']] if 'rocks' not in main_state else []

        self.next_state[self.next_digdug[0]][self.next_digdug[1]] = 2
        
        for enemy in self.next_enemies:
            self.next_state[enemy[0]][enemy[1]] = 3

        for rock in self.next_rocks:
            self.next_state[rock[0]][rock[1]] = 4

        return self.next_state.flatten().unsqueeze(0)
    
    def closest_enemy(self, digdug, enemies):
        digdug = digdug
        min_distance = float('inf')

        for enemy in enemies:
            distance = abs(digdug[0] - enemy[0]) + abs(digdug[1] - enemy[1])
            if distance < min_distance:
                min_distance = distance


        return min_distance

    def calculate_reward(self, action):
        if action != 4:
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
            move = moves[action]

            reward = time_step_reward

            #find the closest enemy
            dist_enemy = self.closest_enemy(self.digdug, self.enemies)
            dist_next_enemy = self.closest_enemy(self.next_digdug, self.next_enemies)

            delta_dist = dist_enemy - dist_next_enemy

            #reward for getting closer to the enemy
            if delta_dist > 0:
                reward += 20 / delta_dist
            else:
                reward -= 10 / abs(delta_dist)

            #punish for trying to walk out of bounds
            if self.digdug[0] + move[0] < 0 or self.next_digdug[0] + move[0] >= x_state \
            or self.next_digdug[1] + move[1] < 0 or self.next_digdug[1] + move[1] >= y_state:
                reward -= 10
            
            #reward for using tunnels
            if self.state[self.digdug[0] + move[0]][self.digdug[1] + move[1]] == 0:
                reward += 0.2
            else:
                reward += 0.1
            
            #reward for killing enemies
            if len(self.enemies) > len(self.next_enemies):
                reward += 100
            
            return reward
        return 0
    
    def train_step(self, action, reward):
        next_state = self.next_state
        state = self.state.view(-1, 48 * 24)
        action = action.view(1, -1)
        reward = torch.tensor([reward], device=device, dtype=torch.float32)

        # Compute state-action values
        state_action_values = self.model(state).gather(1, action)

        # Initialize next state values
        next_state_values = torch.zeros(1, device=device)

        # Update next state values if enemies exist
        if len(self.next_enemies) > 0:
            next_state_values = self.model(next_state).max(1)[0].detach()

        # Ensure discount_rate is defined
        expected_state_action_values = (next_state_values * discount_rate) + reward
        # Compute loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
def select_action(policy_net, state, episode):
    eps_threshold = epsilon_by_episode(episode)
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_size)]], dtype=torch.long)

def get_key(action):
    actions = ["w", "s", "a", "d", "A"]
    return actions[action]

async def agent_loop(server_address="localhost:8000", agent_name="student"):
    """Example client loop."""
    async with websockets.connect(f"ws://{server_address}/player") as websocket:
        # Receive information about static game properties
        await websocket.send(json.dumps({"cmd": "join", "name": agent_name}))

        episode = 1

        policy_net = DQN(x_state * y_state, action_size)
        game = Game(policy_net)
        policy_net.load_state_dict(torch.load('weights/model-v0.pth'))
        policy_net.train()


        while True:
            try:

                state = json.loads(
                    await websocket.recv()
                )  # receive game update, this must be called timely or your game will get out of sync with the server

                if 'map' in state.keys():
                    game.state = torch.tensor(state['map'], dtype=torch.float32, device=device)
                    continue


                state = game.get_state(state)
                action = select_action(policy_net, state, episode)
                key = get_key(action.item())

                await websocket.send(
                    json.dumps({"cmd": "key", "key": key})
                )

                next_state = json.loads(
                    await websocket.recv()
                )  # receive the next game state after your action

                next_state = game.get_next_state(next_state)
                reward = game.calculate_reward(action.item())
                game.train_step(action, reward)

                episode += 1

            except websockets.exceptions.ConnectionClosedOK:
                torch.save(policy_net.state_dict(), 'weights/model-v0.pth')
                print('Model saved')
                print("Server has cleanly disconnected us")
                return


# DO NOT CHANGE THE LINES BELLOW
# You can change the default values using the command line, example:
# $ NAME='arrumador' python3 client.py
loop = asyncio.get_event_loop()
SERVER = os.environ.get("SERVER", "localhost")
PORT = os.environ.get("PORT", "8000")
NAME = os.environ.get("NAME", getpass.getuser())
loop.run_until_complete(agent_loop(f"{SERVER}:{PORT}", NAME))


# {'level': 1, 'step': 11, 'timeout': 3000, 'player': 'tiago', 'score': 0, 'lives': 3, 'digdug': [4, 8], 'enemies': [{'name': 'Fygar', 'id': 'f43675b1-5c1c-49ca-a714-b9d64c529100', 'pos': [17, 7], 'dir': 2}, {'name': 'Pooka', 'id': '29215d7b-02dc-48de-823e-d4577cfec1df', 'pos': [4, 10], 'dir': 0}, {'name': 'Pooka', 'id': '07a370e1-f95f-40d9-8b07-a3fe95dda278', 'pos': [40, 13], 'dir': 3}], 'rocks': [{'id': '939afabe-1aac-429e-80b0-d9b35ccccd40', 'pos': [30, 18]}], 'ts': 1725576860.650459}
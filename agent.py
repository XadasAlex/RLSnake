from collections import deque
from random import randint, choice, sample
from rl_model import Linear_QNet, QTrainer
from snake import Directions, SnakeGame
import torch

class Agent:
    def __init__(self):
        self.n_games = 0
        self.learning_rate = 0.001
        self.batch_size = 1000
        self.epsilon = 0.05 # randomness
        self.gamma = 0.9 # narrow-minded-ness
        self.memory = deque(maxlen=100_000)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=self.learning_rate, gamma=self.gamma)

    def get_action(self, state):
        epsilon = 80 - self.n_games
        directions = [direction.value for direction in Directions]

        if (randint(0, 100) / 100) > epsilon:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            direction =  directions[move]
        else:
            direction = choice(directions)

        return direction

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_short_memory(self, state, action, reward, next_state, game_over):
       self.trainer.train_step(state, action, reward, next_state, game_over)

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

if __name__ == "__main__":
    game = SnakeGame()
    agent = Agent()

    while True:
        state = game.get_game_state()
        action = agent.get_action(state)
        reward, game_over, score = game.play_step(action)
        next_state = game.get_game_state()
        agent.train_short_memory(state, action, reward, next_state, game_over)
        agent.remember(state, action, reward, next_state, game_over)

        if game_over:
            game.restart()
            agent.train_long_memory()
            agent.n_games += 1
            print(agent.n_games)

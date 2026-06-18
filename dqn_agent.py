from collections import deque
from random import randint, choice, sample
from dqn_model import LinearQnet, QTrainer
from snake import Directions, SnakeGame
import torch

class Agent:
    label = "DQN"

    def __init__(self, pretrained=True):
        self.n_games = 0
        self.learning_rate = 0.001
        self.batch_size = 1000
        self.epsilon = 0.05 # explore
        self.gamma = 0.9
        self.memory = deque(maxlen=100_000)
        self.model = LinearQnet(17, 256, 3)
        if pretrained:
            self.model.load("dqn_model.pth")
        self.trainer = QTrainer(self.model, lr=self.learning_rate, gamma=self.gamma)
        self.explore = True

    def get_action(self, state, game):
        epsilon = max(0.0, self.epsilon - self.n_games * 0.001) if self.explore else -1
        directions = [direction.value for direction in Directions]

        if (randint(0, 100) / 100) >= epsilon:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            direction =  directions[move]
        else:
            non_danger_directions = [
                d for d in directions
                if not game.snake.collision(game.snake.get_next_head(game.snake.get_new_facing(d)))
            ]

            if not non_danger_directions:
                direction = choice(directions)
            else:
                direction = choice(non_danger_directions)

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
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    high_score = 0
    game = SnakeGame()
    agent = Agent()

    while True:
        state = game.get_game_state()
        action = agent.get_action(state, game)
        reward, game_over, score = game.play_step(action, agent)
        next_state = game.get_game_state()

        agent.train_short_memory(state, action, reward, next_state, game_over)

        agent.remember(state, action, reward, next_state, game_over)

        if game_over:
            agent.n_games += 1
            agent.train_long_memory()

            total_score += score
            mean_score = total_score / agent.n_games

            if score > high_score:
                high_score = score
                agent.model.save("dqn_model.pth")

            print("Game:", agent.n_games, "Score:", score, "Record:", high_score)

            plot_scores.append(score)
            plot_mean_scores.append(mean_score)
            game.restart()

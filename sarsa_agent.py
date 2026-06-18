from random import randint, choice
from sarsa_model import SarsaTrainer
from dqn_model import LinearQnet
from snake import Directions, SnakeGame
import torch

class SarsaAgent:
    label = "SARSA"

    def __init__(self, pretrained=True):
        self.n_games = 0
        self.learning_rate = 0.001
        self.gamma = 0.9
        self.epsilon = 0.05
        self.explore = True
        self.model = LinearQnet(17, 256, 3)
        if pretrained:
            self.model.load(file_name='sarsa_model.pth')

        self.trainer = SarsaTrainer(self.model, lr=self.learning_rate, gamma=self.gamma)

    def get_action(self, state, game):
        epsilon = max(0.0, self.epsilon - self.n_games * 0.001) if self.explore else -1
        directions = [direction.value for direction in Directions]

        if (randint(0, 100) / 100) >= epsilon:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            direction = directions[move]
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

    def train_step(self, state, action, reward, next_state, next_action, game_over):
        return self.trainer.train_step(state, action, reward, next_state, next_action, game_over)


if __name__ == "__main__":
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    high_score = 0

    game = SnakeGame()
    agent = SarsaAgent()

    state = game.get_game_state()
    action = agent.get_action(state, game)

    while True:
        reward, game_over, score = game.play_step(action, agent)
        next_state = game.get_game_state()

        next_action = agent.get_action(next_state, game) if not game_over else [0, 0, 0]

        agent.train_step(state, action, reward, next_state, next_action, game_over)

        state = next_state
        action = next_action

        if game_over:
            agent.n_games += 1
            total_score += score
            mean_score = total_score / agent.n_games

            if score > high_score:
                high_score = score
                agent.model.save(file_name="sarsa_model.pth")

            print("SARSA Game:", agent.n_games, "Score:", score, "Record:", high_score)

            plot_scores.append(score)
            plot_mean_scores.append(mean_score)
            game.restart()

            state = game.get_game_state()
            action = agent.get_action(state, game)
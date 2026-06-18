import argparse
import os
from datetime import datetime

from dqn_agent import Agent
from sarsa_agent import SarsaAgent
from astar_agent import AStarAgent
from snake import SnakeGame
from comparison_metrics import MetricsLogger, plot_comparison_3way, generate_report_3way

def _dqn_episode(game, agent):
    steps = 0
    total_loss = 0
    loss_count = 0
    while True:
        state = game.get_game_state()
        action = agent.get_action(state, game)
        reward, game_over, score = game.play_step(action, agent)
        next_state = game.get_game_state()
        agent.train_short_memory(state, action, reward, next_state, game_over)
        agent.remember(state, action, reward, next_state, game_over)
        steps += 1
        if game_over:
            agent.train_long_memory()
            game.restart()
            return score, steps, total_loss / max(1, loss_count)


def _sarsa_episode(game, agent):
    steps = 0
    total_loss = 0
    state = game.get_game_state()
    action = agent.get_action(state, game)
    while True:
        reward, game_over, score = game.play_step(action, agent)
        next_state = game.get_game_state()
        next_action = agent.get_action(next_state, game) if not game_over else [0, 0, 0]
        loss = agent.train_step(state, action, reward, next_state, next_action, game_over)
        total_loss += loss
        steps += 1
        state = next_state
        action = next_action
        if game_over:
            game.restart()
            return score, steps, total_loss / steps


def _planner_episode(game, agent):
    steps = 0
    while True:
        state = game.get_game_state()
        action = agent.get_action(state, game)
        _, game_over, score = game.play_step(action, agent)
        steps += 1
        if game_over:
            game.restart()
            return score, steps, 0.0


def _run_episode(game, agent):
    if hasattr(agent, 'train_long_memory'):
        return _dqn_episode(game, agent)
    elif hasattr(agent, 'train_step'):
        return _sarsa_episode(game, agent)
    else:
        return _planner_episode(game, agent)


class ComparisonTrainer:

    def __init__(self, agent_classes, pretrained=True, num_episodes=500, save_interval=50):
        self.pretrained = pretrained
        self.num_episodes = num_episodes
        self.save_interval = save_interval

        self.agents = []
        self.games = []
        self.loggers = []
        self.totals = []

        for AgentClass in agent_classes:
            agent = AgentClass(pretrained=pretrained)
            label = getattr(agent, 'label', type(agent).__name__)
            self.agents.append(agent)
            self.games.append(SnakeGame())
            self.loggers.append(MetricsLogger(label))
            self.totals.append(0)

    def train(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_dir = f'comparison/{timestamp}'
        reports_dir = f'reports/{timestamp}'
        os.makedirs(comparison_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        metrics_dirs = {}
        for agent in self.agents:
            label = getattr(agent, 'label', type(agent).__name__)
            slug = label.lower().replace(' ', '_').replace('*', 'star')
            d = f'metrics/{slug}/{timestamp}'
            metrics_dirs[label] = d
            os.makedirs(d, exist_ok=True)

        labels = [getattr(a, 'label', type(a).__name__) for a in self.agents]

        print('=' * 80)
        print(f"Comparison: {' vs '.join(labels)}")
        print('=' * 80)
        print(f"Episodes     : {self.num_episodes}")
        print(f"Save interval: {self.save_interval}")
        print(f"Pretrained   : {self.pretrained}")
        print(f"Timestamp    : {timestamp}")
        print()

        for episode in range(1, self.num_episodes + 1):
            for i, (agent, game, logger) in enumerate(zip(self.agents, self.games, self.loggers)):
                score, steps, loss = _run_episode(game, agent)
                agent.n_games += 1
                self.totals[i] += score
                mean_score = self.totals[i] / agent.n_games
                logger.log_episode(
                    game_num=agent.n_games,
                    score=score,
                    mean_score=mean_score,
                    loss=loss,
                    steps=steps,
                )

            if episode % 10 == 0:
                print(f"Episode {episode}/{self.num_episodes}")
                for agent, logger in zip(self.agents, self.loggers):
                    label = getattr(agent, 'label', type(agent).__name__)
                    m = logger.metrics
                    print(f"  {label:12s} - Score: {m['scores'][-1]:3.0f}, "
                          f"Mean: {m['mean_scores'][-1]:6.2f}, "
                          f"Steps: {m['steps_per_episode'][-1]:4d}")
                print()

            if episode % self.save_interval == 0:
                print(f"Saving at episode {episode}…")
                loggers_dict = self._loggers_dict()
                for label, logger in loggers_dict.items():
                    slug = label.lower().replace(' ', '_').replace('*', 'star')
                    logger.save_metrics(f"{metrics_dirs[label]}/{slug}_ep{episode}.json")
                plot_comparison_3way(loggers_dict, f"{comparison_dir}/comparison_ep{episode}.png")
                print()

        print('\n' + '=' * 80)
        print("Training complete!")
        print('=' * 80)

        loggers_dict = self._loggers_dict()
        for label, logger in loggers_dict.items():
            slug = label.lower().replace(' ', '_').replace('*', 'star')
            logger.save_metrics(f"{metrics_dirs[label]}/{slug}_final.json")
        plot_comparison_3way(loggers_dict, f"{comparison_dir}/comparison_final.png")
        generate_report_3way(loggers_dict, f"{reports_dir}/comparison_report.txt")

        print("\nFinal Statistics:")
        for agent, total in zip(self.agents, self.totals):
            label = getattr(agent, 'label', type(agent).__name__)
            print(f"  {label:12s}: Mean={total / agent.n_games:.2f}, Episodes={agent.n_games}")
        print(f"\nOutputs in: comparison/{timestamp}/")

    def _loggers_dict(self):
        return {
            getattr(a, 'label', type(a).__name__): l
            for a, l in zip(self.agents, self.loggers)
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Snake RL agents side-by-side.")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of episodes per agent (default: 500)")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Save metrics / plot every N episodes (default: 50)")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Load existing *_model.pth weights (default: start from scratch)")
    args = parser.parse_args()

    trainer = ComparisonTrainer(
        agent_classes=[Agent, SarsaAgent, AStarAgent],
        pretrained=args.pretrained,
        num_episodes=args.episodes,
        save_interval=args.save_interval,
    )
    trainer.train()

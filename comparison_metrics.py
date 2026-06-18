import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class MetricsLogger:
    def __init__(self, algorithm_name):
        self.algorithm_name = algorithm_name
        self.metrics = {
            'scores': [],
            'mean_scores': [],
            'losses': [],
            'steps_per_episode': [],
            'game_numbers': []
        }

    def log_episode(self, game_num, score, mean_score, loss=None, steps=None):
        """Log metrics for one episode."""
        self.metrics['game_numbers'].append(game_num)
        self.metrics['scores'].append(score)
        self.metrics['mean_scores'].append(mean_score)
        self.metrics['losses'].append(loss if loss is not None else 0)
        self.metrics['steps_per_episode'].append(steps if steps is not None else 0)

    def save_metrics(self, filename=None):
        """Save metrics to JSON file."""
        if filename is None:
            filename = f'{self.algorithm_name}_metrics.json'

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print(f"Metrics saved to {filename}")

    def load_metrics(self, filename):
        """Load metrics from JSON file."""
        with open(filename, 'r') as f:
            self.metrics = json.load(f)
        print(f"Metrics loaded from {filename}")


def moving_average(data, window=100):
    """Calculate moving average for smoothing curves."""
    if len(data) < window:
        window = len(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_comparison(dqn_logger, sarsa_logger, save_path='comparison_results.png'):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    """
    Generate comprehensive comparison plots for DQN vs SARSA.

    Args:
        dqn_logger: MetricsLogger instance for DQN
        sarsa_logger: MetricsLogger instance for SARSA
        save_path: Path to save the comparison plot
    """

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN vs SARSA Performance Comparison', fontsize=16, fontweight='bold')

    dqn_metrics = dqn_logger.metrics
    sarsa_metrics = sarsa_logger.metrics

    # Plot 1: Raw scores over time
    ax1 = axes[0, 0]
    ax1.plot(dqn_metrics['game_numbers'], dqn_metrics['scores'],
             alpha=0.3, color='blue', label='DQN Raw')
    ax1.plot(sarsa_metrics['game_numbers'], sarsa_metrics['scores'],
             alpha=0.3, color='red', label='SARSA Raw')

    # Add smoothed curves
    window = 50
    if len(dqn_metrics['scores']) >= window:
        dqn_smooth = moving_average(dqn_metrics['scores'], window)
        ax1.plot(range(window-1, len(dqn_metrics['scores'])), dqn_smooth,
                color='blue', linewidth=2, label='DQN Smoothed')

    if len(sarsa_metrics['scores']) >= window:
        sarsa_smooth = moving_average(sarsa_metrics['scores'], window)
        ax1.plot(range(window-1, len(sarsa_metrics['scores'])), sarsa_smooth,
                color='red', linewidth=2, label='SARSA Smoothed')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Scores Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean scores (cumulative performance)
    ax2 = axes[0, 1]
    ax2.plot(dqn_metrics['game_numbers'], dqn_metrics['mean_scores'],
             color='blue', linewidth=2, label='DQN Mean Score')
    ax2.plot(sarsa_metrics['game_numbers'], sarsa_metrics['mean_scores'],
             color='red', linewidth=2, label='SARSA Mean Score')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Mean Score')
    ax2.set_title('Cumulative Mean Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Learning stability (score variance over windows)
    ax3 = axes[1, 0]
    variance_window = 100

    if len(dqn_metrics['scores']) >= variance_window:
        dqn_variance = [np.var(dqn_metrics['scores'][max(0, i-variance_window):i+1])
                       for i in range(len(dqn_metrics['scores']))]
        ax3.plot(dqn_metrics['game_numbers'], dqn_variance,
                color='blue', alpha=0.7, label='DQN Variance')

    if len(sarsa_metrics['scores']) >= variance_window:
        sarsa_variance = [np.var(sarsa_metrics['scores'][max(0, i-variance_window):i+1])
                         for i in range(len(sarsa_metrics['scores']))]
        ax3.plot(sarsa_metrics['game_numbers'], sarsa_variance,
                color='red', alpha=0.7, label='SARSA Variance')

    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Score Variance')
    ax3.set_title(f'Learning Stability (Window={variance_window})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Calculate statistics
    def get_stats(metrics, last_n=100):
        scores = metrics['scores']
        if len(scores) == 0:
            return {'mean': 0, 'max': 0, 'std': 0, 'last_mean': 0}

        last_scores = scores[-last_n:] if len(scores) >= last_n else scores
        return {
            'total_episodes': len(scores),
            'mean': np.mean(scores),
            'max': np.max(scores),
            'std': np.std(scores),
            'last_mean': np.mean(last_scores)
        }

    dqn_stats = get_stats(dqn_metrics)
    sarsa_stats = get_stats(sarsa_metrics)

    stats_text = f"""
    Performance Statistics

    DQN:
    Total Episodes: {dqn_stats['total_episodes']}
    Overall Mean: {dqn_stats['mean']:.2f}
    Max Score: {dqn_stats['max']:.0f}
    Std Dev: {dqn_stats['std']:.2f}
    Last 100 Mean: {dqn_stats['last_mean']:.2f}

    SARSA:
    Total Episodes: {sarsa_stats['total_episodes']}
    Overall Mean: {sarsa_stats['mean']:.2f}
    Max Score: {sarsa_stats['max']:.0f}
    Std Dev: {sarsa_stats['std']:.2f}
    Last 100 Mean: {sarsa_stats['last_mean']:.2f}

    Analysis:
    {"DQN" if dqn_stats['last_mean'] > sarsa_stats['last_mean'] else "SARSA"} has higher recent performance
    {"DQN" if dqn_stats['std'] > sarsa_stats['std'] else "SARSA"} has higher variance (less stable)
    """

    ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")

    return fig


def generate_report(dqn_logger, sarsa_logger, output_file='comparison_report.txt'):
    """Generate text report comparing both algorithms."""

    dqn_metrics = dqn_logger.metrics
    sarsa_metrics = sarsa_logger.metrics

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DQN vs SARSA: Performance Comparison Report\n")
        f.write("=" * 80 + "\n\n")

        for name, metrics in [("DQN", dqn_metrics), ("SARSA", sarsa_metrics)]:
            scores = metrics['scores']
            if len(scores) == 0:
                continue

            f.write(f"{name} Statistics:\n")
            f.write(f"  Total Episodes: {len(scores)}\n")
            f.write(f"  Mean Score: {np.mean(scores):.2f}\n")
            f.write(f"  Max Score: {np.max(scores)}\n")
            f.write(f"  Min Score: {np.min(scores)}\n")
            f.write(f"  Std Dev: {np.std(scores):.2f}\n")

            if len(scores) >= 100:
                last_100 = scores[-100:]
                f.write(f"  Last 100 Episodes Mean: {np.mean(last_100):.2f}\n")

            f.write("\n")

        f.write("=" * 80 + "\n")

    print(f"Report saved to {output_file}")


ALGO_COLORS = {
    'DQN': 'blue',
    'SARSA': 'red',
    'A*': 'green',
    'A* Dumb': 'orange',
}


def plot_comparison_3way(loggers: dict, save_path='comparison_3way.png'):
    """
    Generate comparison plots for an arbitrary dict of MetricsLogger objects.

    Args:
        loggers: {'DQN': logger, 'SARSA': logger, 'A*': logger, ...}
        save_path: Where to write the PNG.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('DQN vs SARSA vs A* — Performance Comparison', fontsize=16, fontweight='bold')

    window = 50

    # --- Plot 1: Raw + smoothed scores ---
    ax1 = axes[0, 0]
    for name, logger in loggers.items():
        color = ALGO_COLORS.get(name, 'gray')
        m = logger.metrics
        ax1.plot(m['game_numbers'], m['scores'], alpha=0.2, color=color)
        if len(m['scores']) >= window:
            smooth = moving_average(m['scores'], window)
            ax1.plot(range(window - 1, len(m['scores'])), smooth,
                     color=color, linewidth=2, label=f'{name} (smoothed)')
        else:
            ax1.plot(m['game_numbers'], m['scores'], color=color, linewidth=2, label=name)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title(f'Scores Over Time (smoothed w={window})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Cumulative mean scores ---
    ax2 = axes[0, 1]
    for name, logger in loggers.items():
        color = ALGO_COLORS.get(name, 'gray')
        m = logger.metrics
        ax2.plot(m['game_numbers'], m['mean_scores'], color=color, linewidth=2, label=name)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Mean Score')
    ax2.set_title('Cumulative Mean Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Rolling score variance ---
    ax3 = axes[1, 0]
    var_window = 100
    for name, logger in loggers.items():
        color = ALGO_COLORS.get(name, 'gray')
        scores = logger.metrics['scores']
        game_nums = logger.metrics['game_numbers']
        if len(scores) >= var_window:
            variance = [np.var(scores[max(0, i - var_window):i + 1])
                        for i in range(len(scores))]
            ax3.plot(game_nums, variance, color=color, alpha=0.8, label=name)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Score Variance')
    ax3.set_title(f'Learning Stability (rolling window={var_window})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Summary statistics text panel ---
    ax4 = axes[1, 1]
    ax4.axis('off')

    lines = ['Performance Statistics\n']
    for name, logger in loggers.items():
        scores = logger.metrics['scores']
        if not scores:
            continue
        last_n = scores[-100:] if len(scores) >= 100 else scores
        lines.append(f'{name}:')
        lines.append(f'  Episodes : {len(scores)}')
        lines.append(f'  Overall Mean: {np.mean(scores):.2f}')
        lines.append(f'  Max Score : {np.max(scores):.0f}')
        lines.append(f'  Std Dev   : {np.std(scores):.2f}')
        lines.append(f'  Last-100 Mean: {np.mean(last_n):.2f}')
        lines.append('')

    # Add winner line
    last_means = {n: np.mean(l.metrics['scores'][-100:]) if len(l.metrics['scores']) >= 100
                  else np.mean(l.metrics['scores']) if l.metrics['scores'] else 0
                  for n, l in loggers.items()}
    winner = max(last_means, key=last_means.get)
    lines.append(f'Best last-100 mean: {winner} ({last_means[winner]:.2f})')

    ax4.text(0.05, 0.95, '\n'.join(lines), fontsize=10, verticalalignment='top',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"3-way comparison plot saved to {save_path}")


def generate_report_3way(loggers: dict, output_file='comparison_3way_report.txt'):
    """Write a text report comparing an arbitrary number of algorithms."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('DQN vs SARSA vs A* — Performance Comparison Report\n')
        f.write('=' * 80 + '\n\n')

        for name, logger in loggers.items():
            scores = logger.metrics['scores']
            if not scores:
                continue
            f.write(f'{name} Statistics:\n')
            f.write(f'  Total Episodes : {len(scores)}\n')
            f.write(f'  Mean Score     : {np.mean(scores):.2f}\n')
            f.write(f'  Max Score      : {np.max(scores)}\n')
            f.write(f'  Min Score      : {np.min(scores)}\n')
            f.write(f'  Std Dev        : {np.std(scores):.2f}\n')
            if len(scores) >= 100:
                f.write(f'  Last-100 Mean  : {np.mean(scores[-100:]):.2f}\n')
            f.write('\n')

        f.write('=' * 80 + '\n')

    print(f"3-way report saved to {output_file}")

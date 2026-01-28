"""
Advanced Training Analyzer for ACC RL Agent

This script analyzes trained models and provides insights into agent behavior.
Useful for debugging poor training results.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from ACCEnv import ACCEnv
import matplotlib.pyplot as plt


class ModelAnalyzer:
    """Analyze trained PPO models for behavior patterns"""

    def __init__(self, model_path):
        """
        Initialize analyzer with a trained model

        Args:
            model_path: Path to the saved .zip model
        """
        self.model = PPO.load(model_path)
        self.env = None
        self.action_history = []
        self.observation_history = []
        self.reward_history = []

    def run_inference(self, num_episodes=5, max_steps=1000):
        """
        Run the agent on the environment without training

        Args:
            num_episodes: Number of episodes to run
            max_steps: Max steps per episode
        """
        print(f"Running inference for {num_episodes} episodes...")

        try:
            self.env = ACCEnv()
        except Exception as e:
            print(f"Failed to initialize environment: {e}")
            return False

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_actions = []
            episode_observations = []

            for step in range(max_steps):
                # Get action from model
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward
                episode_actions.append(action)
                episode_observations.append(obs.copy())
                self.reward_history.append(reward)

                if terminated or truncated:
                    break

            self.action_history.append(episode_actions)
            self.observation_history.append(episode_observations)

            print(
                f"Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={step + 1}"
            )

        self.env.close()
        return True

    def analyze_action_distribution(self):
        """Analyze the distribution of actions taken by the agent"""
        print("\n=== Action Distribution Analysis ===")

        all_actions = np.vstack([np.array(actions) for actions in self.action_history])

        steering = all_actions[:, 0]
        throttle = all_actions[:, 1]
        brake = all_actions[:, 2]

        print(f"\nSteering:")
        print(f"  Mean: {np.mean(steering):.4f}, Std: {np.std(steering):.4f}")
        print(f"  Min: {np.min(steering):.4f}, Max: {np.max(steering):.4f}")
        print(f"  Range: {np.max(steering) - np.min(steering):.4f}")

        print(f"\nThrottle:")
        print(f"  Mean: {np.mean(throttle):.4f}, Std: {np.std(throttle):.4f}")
        print(f"  Min: {np.min(throttle):.4f}, Max: {np.max(throttle):.4f}")
        print(
            f"  % Full Throttle (>0.9): {100 * np.sum(throttle > 0.9) / len(throttle):.1f}%"
        )
        print(f"  % Idle (0-0.1): {100 * np.sum(throttle < 0.1) / len(throttle):.1f}%")

        print(f"\nBrake:")
        print(f"  Mean: {np.mean(brake):.4f}, Std: {np.std(brake):.4f}")
        print(f"  Min: {np.min(brake):.4f}, Max: {np.max(brake):.4f}")
        print(f"  % Using Brake: {100 * np.sum(brake > 0.1) / len(brake):.1f}%")

    def analyze_reward_distribution(self):
        """Analyze reward statistics"""
        print("\n=== Reward Analysis ===")

        rewards = np.array(self.reward_history)

        print(f"Mean Reward per Step: {np.mean(rewards):.4f}")
        print(f"Std Dev: {np.std(rewards):.4f}")
        print(f"Min: {np.min(rewards):.4f}, Max: {np.max(rewards):.4f}")

        positive_rewards = rewards[rewards > 0]
        negative_rewards = rewards[rewards < 0]

        if len(positive_rewards) > 0:
            print(
                f"Positive Rewards: {len(positive_rewards)} ({100*len(positive_rewards)/len(rewards):.1f}%)"
            )
            print(f"  Mean Positive: {np.mean(positive_rewards):.4f}")

        if len(negative_rewards) > 0:
            print(
                f"Negative Rewards: {len(negative_rewards)} ({100*len(negative_rewards)/len(rewards):.1f}%)"
            )
            print(f"  Mean Negative: {np.mean(negative_rewards):.4f}")

    def plot_action_distributions(self, output_file=None):
        """Visualize action distributions"""
        all_actions = np.vstack([np.array(actions) for actions in self.action_history])

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Agent Action Distributions", fontsize=14)

        axes[0].hist(all_actions[:, 0], bins=50, edgecolor="black")
        axes[0].set_xlabel("Steering")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Steering Distribution")
        axes[0].axvline(
            np.mean(all_actions[:, 0]), color="red", linestyle="--", label="Mean"
        )
        axes[0].legend()

        axes[1].hist(all_actions[:, 1], bins=50, edgecolor="black")
        axes[1].set_xlabel("Throttle")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Throttle Distribution")
        axes[1].axvline(
            np.mean(all_actions[:, 1]), color="red", linestyle="--", label="Mean"
        )
        axes[1].legend()

        axes[2].hist(all_actions[:, 2], bins=50, edgecolor="black")
        axes[2].set_xlabel("Brake")
        axes[2].set_ylabel("Frequency")
        axes[2].set_title("Brake Distribution")
        axes[2].axvline(
            np.mean(all_actions[:, 2]), color="red", linestyle="--", label="Mean"
        )
        axes[2].legend()

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=100)
            print(f"Plot saved to {output_file}")
        else:
            plt.show()

        plt.close()

    def plot_reward_distribution(self, output_file=None):
        """Visualize reward distribution"""
        rewards = np.array(self.reward_history)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        ax1.hist(rewards, bins=100, edgecolor="black")
        ax1.set_xlabel("Reward Value")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Reward Distribution")
        ax1.axvline(
            np.mean(rewards),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(rewards):.4f}",
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Reward over time
        ax2.plot(rewards, alpha=0.6)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Reward")
        ax2.set_title("Reward Over Time")
        # Rolling average
        if len(rewards) > 100:
            rolling_avg = np.convolve(rewards, np.ones(100) / 100, mode="valid")
            ax2.plot(rolling_avg, color="red", linewidth=2, label="100-step MA")
            ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=100)
            print(f"Plot saved to {output_file}")
        else:
            plt.show()

        plt.close()


def diagnose_model_issues(model_path):
    """
    Run comprehensive diagnostics on a trained model

    Args:
        model_path: Path to the .zip model file
    """
    print("=" * 80)
    print("ACC RL Model Diagnostic Report")
    print("=" * 80)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        analyzer = ModelAnalyzer(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run inference
    if not analyzer.run_inference(num_episodes=3, max_steps=500):
        print("Failed to run inference")
        return

    # Analyze results
    analyzer.analyze_action_distribution()
    analyzer.analyze_reward_distribution()

    # Generate plots
    analyzer.plot_action_distributions("action_distributions.png")
    analyzer.plot_reward_distribution("reward_distribution.png")

    print("\n" + "=" * 80)
    print("Diagnostic Report Complete")
    print("Plots saved: action_distributions.png, reward_distribution.png")
    print("=" * 80)

    # Recommendations
    print("\n=== Recommendations ===")

    all_actions = np.vstack([np.array(actions) for actions in analyzer.action_history])
    throttle = all_actions[:, 1]
    brake = all_actions[:, 2]
    rewards = np.array(analyzer.reward_history)

    if np.mean(throttle) > 0.8:
        print("⚠️  Agent uses maximum throttle too often (>80%)")
        print("   → Reduce REWARD_PROGRESS_MULTIPLIER or increase speed penalties")

    if np.mean(brake) < 0.05:
        print("⚠️  Agent rarely uses brakes")
        print("   → Add explicit brake reward or penalties for off-track events")

    if np.std(rewards) > 100:
        print("⚠️  Reward variance is very high")
        print("   → Reward shaping might be unstable, check for outlier penalties")

    if np.mean(rewards) < 0:
        print("⚠️  Average reward is negative (agent keeps crashing)")
        print("   → Increase survival reward or reduce penalties")

    if np.sum(brake > 0.1) / len(brake) < 0.05:
        print("⚠️  Brake usage very low - might struggle at turns")
        print("   → Add penalty for speed at specific track positions")


if __name__ == "__main__":
    # Example usage
    MODEL_PATH = "models/ppo_acc_final_latest.zip"

    if os.path.exists(MODEL_PATH):
        diagnose_model_issues(MODEL_PATH)
    else:
        print(f"Model not found at {MODEL_PATH}")
        print("Please train a model first using RLtrain.py")

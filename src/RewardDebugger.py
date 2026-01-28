"""
Reward Debugging and Monitoring Tool for ACC RL Training

This module provides utilities to:
1. Log detailed reward components for each step
2. Visualize reward trends during training
3. Analyze agent behavior patterns
4. Detect reward engineering issues
"""

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime


class RewardDebugger:
    """Monitor and debug rewards during RL training"""
    
    def __init__(self, log_dir="reward_logs/"):
        self.log_dir = log_dir
        self.episode_stats = defaultdict(list)
        self.step_rewards = defaultdict(list)
        self.episode_count = 0
        
    def log_step_breakdown(self, step, reward_components):
        """
        Log individual reward components for a step
        
        Args:
            step: Step number
            reward_components: Dict with keys like 'speed', 'progress', 'damage', 'slip', etc.
        """
        for component_name, value in reward_components.items():
            self.step_rewards[f"step_{component_name}"].append(value)
    
    def log_episode_stats(self, episode_num, total_reward, episode_length, avg_speed, max_damage):
        """
        Log episode-level statistics
        
        Args:
            episode_num: Episode number
            total_reward: Total cumulative reward for episode
            episode_length: Number of steps
            avg_speed: Average speed during episode
            max_damage: Maximum damage sustained
        """
        self.episode_stats['episode'].append(episode_num)
        self.episode_stats['total_reward'].append(total_reward)
        self.episode_stats['length'].append(episode_length)
        self.episode_stats['avg_speed'].append(avg_speed)
        self.episode_stats['max_damage'].append(max_damage)
        self.episode_count += 1
    
    def print_episode_summary(self, episode_num, total_reward, episode_length, avg_speed, 
                              avg_steering_rate, max_slip, max_damage):
        """Print formatted episode summary"""
        print("\n" + "="*80)
        print(f"Episode {episode_num} Summary")
        print("="*80)
        print(f"Total Reward:        {total_reward:>12.2f}")
        print(f"Episode Length:      {episode_length:>12} steps")
        print(f"Avg Speed:           {avg_speed:>12.2f} km/h")
        print(f"Avg Steering Rate:   {avg_steering_rate:>12.4f}")
        print(f"Max Wheel Slip:      {max_slip:>12.4f}")
        print(f"Max Damage:          {max_damage:>12.4f}")
        print("="*80 + "\n")
    
    def plot_reward_trends(self, output_file=None):
        """
        Plot reward trends for visualization
        
        Args:
            output_file: Optional file path to save plot
        """
        if len(self.episode_stats['episode']) == 0:
            print("No episode data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Total Reward trend
        axes[0, 0].plot(self.episode_stats['episode'], self.episode_stats['total_reward'], marker='.')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Cumulative Reward per Episode')
        axes[0, 0].grid(True)
        
        # Episode Length trend
        axes[0, 1].plot(self.episode_stats['episode'], self.episode_stats['length'], marker='.')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Episode Length (steps)')
        axes[0, 1].set_title('Episode Duration')
        axes[0, 1].grid(True)
        
        # Average Speed trend
        axes[1, 0].plot(self.episode_stats['episode'], self.episode_stats['avg_speed'], marker='.')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Speed (km/h)')
        axes[1, 0].set_title('Average Speed per Episode')
        axes[1, 0].grid(True)
        
        # Damage trend
        axes[1, 1].plot(self.episode_stats['episode'], self.episode_stats['max_damage'], marker='.')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Max Damage')
        axes[1, 1].set_title('Vehicle Damage per Episode')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=100)
            print(f"Plot saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def get_episode_summary(self):
        """Get summary statistics of all logged episodes"""
        if len(self.episode_stats['total_reward']) == 0:
            return None
        
        rewards = np.array(self.episode_stats['total_reward'])
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'recent_avg': np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards),
        }


class EpisodeMetricsTracker:
    """Track metrics during an episode for debugging"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset tracking for new episode"""
        self.speeds = []
        self.steering_rates = []
        self.slips = []
        self.damages = []
        self.rewards = []
        self.throttles = []
        self.brakes = []
    
    def record_step(self, speed, steering_rate, slip, damage, reward, throttle, brake):
        """Record metrics for a single step"""
        self.speeds.append(speed)
        self.steering_rates.append(steering_rate)
        self.slips.append(slip)
        self.damages.append(damage)
        self.rewards.append(reward)
        self.throttles.append(throttle)
        self.brakes.append(brake)
    
    def get_summary(self):
        """Get episode summary statistics"""
        return {
            'total_reward': sum(self.rewards),
            'avg_speed': np.mean(self.speeds) if self.speeds else 0,
            'avg_steering_rate': np.mean(self.steering_rates) if self.steering_rates else 0,
            'max_slip': np.max(self.slips) if self.slips else 0,
            'max_damage': np.max(self.damages) if self.damages else 0,
            'episode_length': len(self.rewards),
            'avg_reward_per_step': np.mean(self.rewards) if self.rewards else 0,
        }

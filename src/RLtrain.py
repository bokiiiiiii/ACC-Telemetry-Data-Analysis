import os
import time
from ACCEnv import ACCEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn as torch_nn

NUM_EQUALS = 50


def train_agent():
    """
    Main function to initialize, train, and save the PPO agent for ACC.

    Improvements in this version:
    - Better hyperparameter tuning for racing task
    - Deeper network architecture (256-256-128)
    - Higher entropy coefficient for better exploration
    - More training epochs for stability
    - Progress monitoring and logging
    """
    # --- Configuration ---
    LOG_DIR = "logs/"
    MODEL_DIR = "models/"

    # Training hyperparameters
    TOTAL_TIMESTEPS = 1000000
    SAVE_FREQ = 50000
    MODEL_NAME_PREFIX = "ppo_acc"
    LOG_INTERVAL = 10  # Log every 10 episodes

    # Path for continuing training from existing model.
    LOAD_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_acc_to_continue.zip")

    # PPO agent hyperparameters (Optimized for driving task)
    PPO_PARAMS = {
        "learning_rate": 3e-4,  # Increased from 0.0003 for faster learning
        "n_steps": 2048,
        "batch_size": 128,  # Increased from 64 for better batch statistics
        "n_epochs": 20,  # Increased from 10 for more thorough training
        "gamma": 0.99,  # Discount factor
        "gae_lambda": 0.95,  # GAE lambda
        "clip_range": 0.2,  # PPO clipping range
        "ent_coef": 0.03,  # Increased from 0.01 to encourage exploration
        "vf_coef": 0.5,  # Value function coefficient
        "max_grad_norm": 0.5,  # Gradient clipping
        "device": "auto",
        # Enhanced network architecture for complex driving task
        "policy_kwargs": dict(
            net_arch=dict(
                pi=[256, 256, 128],  # Policy network: deeper
                vf=[256, 256, 128],  # Value network: deeper
            ),
            activation_fn=None,  # Use default activation (ReLU)
        ),
    }

    # Create log and model save directories if not exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Initialize ACC Environment
    print("Initializing ACC Environment...")
    try:
        env = ACCEnv()
        print("ACC Environment initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize ACC Environment: {e}")
        print(
            "Please ensure Assetto Corsa Competizione is running and Shared Memory is enabled."
        )
        return

    # Initialize PPO Agent or load existing one
    print("Initializing PPO Agent...")
    try:
        if os.path.exists(LOAD_MODEL_PATH):
            print(f"Loading existing model from '{LOAD_MODEL_PATH}'...")
            model = PPO.load(
                LOAD_MODEL_PATH,
                env=env,
                tensorboard_log=LOG_DIR,
            )
            print("Model loaded successfully.")
        else:
            print(f"Model not found at '{LOAD_MODEL_PATH}'. Creating a new model...")
            model = PPO(
                policy="MlpPolicy",
                env=env,
                verbose=0,
                tensorboard_log=LOG_DIR,
                **PPO_PARAMS,
            )
            print("New PPO Agent initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize or load PPO Agent: {e}")
        env.close()
        return

    # Set up callback to save model periodically
    current_timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_save_path = os.path.join(
        MODEL_DIR, f"{MODEL_NAME_PREFIX}_{current_timestamp}"
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=checkpoint_save_path,
        name_prefix=MODEL_NAME_PREFIX,
    )

    # Start training
    print(f"Starting training, total timesteps: {TOTAL_TIMESTEPS}...")
    print(
        f"PPO Parameters: lr={PPO_PARAMS['learning_rate']}, batch_size={PPO_PARAMS['batch_size']}, "
        f"n_epochs={PPO_PARAMS['n_epochs']}, ent_coef={PPO_PARAMS['ent_coef']}"
    )
    print(f"Network Architecture: {PPO_PARAMS['policy_kwargs']}")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            log_interval=LOG_INTERVAL,  # Log every 10 episodes
            tb_log_name=f"{MODEL_NAME_PREFIX}_{current_timestamp}",
        )
        print("Training completed.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        # Save final model
        final_model_name = f"{MODEL_NAME_PREFIX}_final_{current_timestamp}.zip"
        final_model_path = os.path.join(MODEL_DIR, final_model_name)
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")

        # Close environment
        print("Closing ACC Environment...")
        env.close()
        print("ACC Environment closed.")


if __name__ == "__main__":
    print("=" * NUM_EQUALS)
    print("Assetto Corsa Competizione (ACC) Reinforcement Learning Training Script")
    print("=" * NUM_EQUALS)
    print("Important Notes:")
    print("1. Ensure ACC game is running.")
    print("2. Ensure Shared Memory is enabled in ACC.")
    print("3. Ensure vJoy driver is installed and configured.")
    print("4. Press Ctrl+C to terminate training and save the current model.")
    print("-" * NUM_EQUALS)

    # Wait a few seconds to allow user to switch to game window or prepare
    wait_time = 5
    print(
        f"Training will start in {wait_time} seconds, please prepare the ACC game environment..."
    )
    for i in range(wait_time, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("Starting training execution...")

    train_agent()

    print("=" * NUM_EQUALS)
    print("Training script execution finished.")
    print("=" * NUM_EQUALS)

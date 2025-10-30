"""
Training script for the layout optimization RL model.
Uses Stable-Baselines3 with PPO algorithm.
"""

import os
import sys
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.layout_env import LayoutOptimizationEnv
from utils.data_schema import create_sample_layout


class TrainingConfig:
    """Configuration for training"""
    def __init__(self):
        # Algorithm selection
        self.algorithm = "PPO"  # Options: PPO, SAC, A2C
        
        # Training parameters
        self.total_timesteps = 100000
        self.n_steps = 2048  # For PPO
        self.batch_size = 64
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # Network architecture
        self.policy_kwargs = dict(
            net_arch=[256, 256]
        )
        
        # Checkpoint and logging
        self.save_freq = 10000
        self.eval_freq = 5000
        self.n_eval_episodes = 10
        
        # Directories
        self.log_dir = "./logs"
        self.checkpoint_dir = "./models/checkpoints"
        self.best_model_dir = "./models/best_model"
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)


def make_env(config_layout):
    """Create and wrap the environment"""
    def _init():
        env = LayoutOptimizationEnv(config_layout)
        env = Monitor(env)
        return env
    return _init


def train_model(config: TrainingConfig, layout_config=None):
    """
    Train the RL model
    
    Args:
        config: Training configuration
        layout_config: Layout configuration (uses sample if None)
    """
    print("=" * 60)
    print("Layout Optimization RL Training")
    print("=" * 60)
    
    # Create layout configuration
    if layout_config is None:
        print("\nCreating sample layout configuration...")
        layout_config = create_sample_layout()
    
    print(f"Facility: {layout_config.facility.name}")
    print(f"Dimensions: {layout_config.facility.length}m x {layout_config.facility.width}m")
    print(f"Number of elements: {len(layout_config.elements)}")
    
    # Create environment
    print("\nCreating environment...")
    env = LayoutOptimizationEnv(layout_config)
    
    # Check environment
    print("Checking environment...")
    try:
        check_env(env, warn=True)
        print("✓ Environment check passed!")
    except Exception as e:
        print(f"✗ Environment check failed: {e}")
        return None
    
    # Create vectorized environment
    print("\nCreating vectorized environment...")
    vec_env = DummyVecEnv([make_env(layout_config)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(layout_config)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=config.checkpoint_dir,
        name_prefix="layout_model"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.best_model_dir,
        log_path=config.log_dir,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True
    )
    
    callback_list = CallbackList([checkpoint_callback, eval_callback])
    
    # Create model
    print(f"\nCreating {config.algorithm} model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log = os.path.join(config.log_dir, f"{config.algorithm}_{timestamp}")
    
    if config.algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            policy_kwargs=config.policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_log
        )
    elif config.algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            gamma=config.gamma,
            policy_kwargs=config.policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_log
        )
    elif config.algorithm == "A2C":
        model = A2C(
            "MlpPolicy",
            vec_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            policy_kwargs=config.policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_log
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")
    
    print(f"✓ Model created successfully!")
    print(f"  Policy: MlpPolicy")
    print(f"  Network architecture: {config.policy_kwargs['net_arch']}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Total timesteps: {config.total_timesteps}")
    
    # Check if progress bar is available
    try:
        import tqdm
        import rich
        use_progress_bar = True
    except ImportError:
        use_progress_bar = False
        print("\n⚠️  Note: Install tqdm and rich for progress bar: pip install tqdm rich")
    
    # Train model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print(f"Tensorboard log: {tensorboard_log}")
    print(f"Run 'tensorboard --logdir {config.log_dir}' to monitor training")
    print()
    
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback_list,
            progress_bar=use_progress_bar
        )
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        
        # Save final model
        final_model_path = os.path.join(config.checkpoint_dir, f"final_model_{timestamp}")
        model.save(final_model_path)
        vec_env.save(os.path.join(config.checkpoint_dir, f"vec_normalize_{timestamp}.pkl"))
        
        print(f"\nFinal model saved to: {final_model_path}")
        print(f"Best model saved to: {config.best_model_dir}")
        
        return model
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        # Save current model
        interrupted_model_path = os.path.join(config.checkpoint_dir, f"interrupted_model_{timestamp}")
        model.save(interrupted_model_path)
        print(f"Model saved to: {interrupted_model_path}")
        return model
    
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_model(model_path: str, layout_config=None, n_episodes: int = 10):
    """
    Evaluate a trained model
    
    Args:
        model_path: Path to saved model
        layout_config: Layout configuration
        n_episodes: Number of evaluation episodes
    """
    print(f"\nEvaluating model: {model_path}")
    
    # Create layout configuration
    if layout_config is None:
        layout_config = create_sample_layout()
    
    # Create environment
    env = LayoutOptimizationEnv(layout_config)
    
    # Load model
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "sac" in model_path.lower():
        model = SAC.load(model_path)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path)
    else:
        model = PPO.load(model_path)  # default
    
    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.4f}, Length = {step_count}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Average reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"Average length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.4f}")
    print(f"Max reward: {np.max(episode_rewards):.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train layout optimization RL model")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "SAC", "A2C"],
                       help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--eval", type=str, default=None,
                       help="Path to model for evaluation (skips training)")
    
    args = parser.parse_args()
    
    # Create training configuration
    config = TrainingConfig()
    config.algorithm = args.algorithm
    config.total_timesteps = args.timesteps
    
    if args.eval:
        # Evaluation mode
        evaluate_model(args.eval)
    else:
        # Training mode
        model = train_model(config)
        
        if model is not None:
            print("\n✓ Training pipeline completed successfully!")
            print(f"\nTo view training progress:")
            print(f"  tensorboard --logdir {config.log_dir}")

import torch
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from geometry_dash_env import GeometryDashEnv
import os

def main():
    parser = argparse.ArgumentParser(description='Train Geometry Dash AI')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--model_name', type=str, default='geometry_dash_ai', help='Name for saved model')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('./trained_models', exist_ok=True)
    os.makedirs('./training_logs', exist_ok=True)
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")
    if device == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    print("🎯 Creating training environments...")
    
    # Create training environment
    def make_env():
        def _init():
            env = GeometryDashEnv(headless=True)  # Headless for faster training
            return Monitor(env)
        return _init
    
    # Use DummyVecEnv for stability
    if args.n_envs > 1:
        env = DummyVecEnv([make_env() for _ in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env()])
    
    print("🤖 Creating PPO model...")
    
    # Create PPO model with optimized hyperparameters for platformers
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=2048,           # Number of steps per update
        batch_size=64,          # Batch size for training
        n_epochs=10,            # Number of epochs per update
        gamma=args.gamma,       # Discount factor
        gae_lambda=0.95,        # GAE parameter
        clip_range=0.2,         # Clipping parameter
        ent_coef=0.01,          # Entropy coefficient (encourages exploration)
        vf_coef=0.5,            # Value function coefficient
        max_grad_norm=0.5,      # Maximum gradient norm
        tensorboard_log="./training_logs/",
        policy_kwargs=dict(
            net_arch=[256, 256]  # Neural network architecture
        ),
        device=device,
        verbose=1
    )
    
    # Create callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path='./trained_models/',
        log_path='./training_logs/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./trained_models/',
        name_prefix=args.model_name
    )
    
    print("🎮 Starting Training!")
    print(f"📊 Total timesteps: {args.timesteps:,}")
    print(f"🔄 Parallel environments: {args.n_envs}")
    print(f"📈 Learning rate: {args.learning_rate}")
    print(f"🎯 Gamma: {args.gamma}")
    print("💡 Training in progress...")
    
    # Train the model
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name=args.model_name
    )
    
    # Save the final model
    model.save(f"./trained_models/{args.model_name}_final")
    print(f"✅ Training completed! Model saved as: {args.model_name}_final")
    
    env.close()

if __name__ == '__main__':
    main()
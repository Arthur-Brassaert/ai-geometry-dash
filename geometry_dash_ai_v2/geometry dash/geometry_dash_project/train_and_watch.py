import torch
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from geometry_dash_env import GeometryDashEnv
import os

def train_and_watch():
    # Create directories
    os.makedirs('./trained_models', exist_ok=True)
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")
    
    # Create environment
    env = GeometryDashEnv(headless=True)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        device=device,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    
    # Train in chunks and test periodically
    total_timesteps = 50000
    test_interval = 5000
    
    for chunk in range(total_timesteps // test_interval):
        print(f"\n🎯 Training chunk {chunk + 1}/{(total_timesteps // test_interval)}")
        
        # Train for one chunk
        model.learn(total_timesteps=test_interval, reset_num_timesteps=False)
        
        # Save checkpoint
        model.save(f"./trained_models/checkpoint_{chunk}")
        
        # Test the current model
        print("👀 Testing current model...")
        test_current_model(model)
    
    env.close()

def test_current_model(model):
    """Test the current model visually"""
    test_env = GeometryDashEnv(headless=False)
    
    obs, info = test_env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 500:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        
        test_env.render()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                test_env.close()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    test_env.close()
                    return
    
    print(f"   Current performance: Score={info['score']}, Steps={steps}, Reward={total_reward:.2f}")
    test_env.close()

if __name__ == '__main__':
    train_and_watch()
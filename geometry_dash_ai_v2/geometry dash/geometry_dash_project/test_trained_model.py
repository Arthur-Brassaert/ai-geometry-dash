import pygame
import numpy as np
from stable_baselines3 import PPO
from geometry_dash_env import GeometryDashEnv

def test_model(model_path, num_episodes=5, render=True):
    print(f"🧪 Testing model: {model_path}")
    
    # Load the trained model
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create environment
    env = GeometryDashEnv(headless=not render)
    
    scores = []
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n🎮 Starting Episode {episode + 1}")
        
        while not done and steps < 2000:  # Limit steps per episode
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
                
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            env.close()
                            return
        
        scores.append(info['score'])
        total_rewards.append(total_reward)
        print(f"   Score: {info['score']} | Steps: {steps} | Total Reward: {total_reward:.2f}")
    
    env.close()
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"   Average Score: {np.mean(scores):.2f}")
    print(f"   Average Reward: {np.mean(total_rewards):.2f}")
    print(f"   Best Score: {max(scores)}")

if __name__ == '__main__':
    # Test the best model (change the path as needed)
    test_model("./trained_models/best_model.zip", num_episodes=3, render=True)
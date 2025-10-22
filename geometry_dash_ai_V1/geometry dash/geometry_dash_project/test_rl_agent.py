import pygame
import numpy as np
from stable_baselines3 import PPO
from geometry_dash_env import GeometryDashEnv

def test_agent(model_path, num_episodes=5):
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create environment with rendering
    env = GeometryDashEnv(headless=False)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"Starting episode {episode + 1}")
        
        while not done and steps < 5000:  # Limit steps per episode
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Render the game
            env.render()
            
            # Handle pygame events (so we can close the window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        env.close()
                        return
        
        print(f"Episode {episode + 1}: Score = {info['score']}, Steps = {steps}, Total Reward = {total_reward:.2f}")
    
    env.close()

if __name__ == '__main__':
    test_agent("./trained_models/best_model.zip", num_episodes=3)
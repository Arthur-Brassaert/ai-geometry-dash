import os
import sys
import pygame
import numpy as np
from stable_baselines3 import PPO

# Resolve paths relative to this script so running from a different CWD still works
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from geometry_dash_env import GeometryDashEnv

def test_model(model_path, num_episodes=5, render=True):
    # If a relative path is given, resolve it relative to the script directory
    if not os.path.isabs(model_path):
        model_path = os.path.join(SCRIPT_DIR, model_path)

    print(f"🧪 Testing model: {model_path}")

    # If the model file doesn't exist, try a couple of likely locations and show helpful info
    if not os.path.exists(model_path):
        alt_paths = [
            os.path.join(SCRIPT_DIR, 'trained_models', os.path.basename(model_path)),
            os.path.join(os.path.dirname(SCRIPT_DIR), 'trained_models', os.path.basename(model_path)),
        ]
        found = False
        for p in alt_paths:
            if os.path.exists(p):
                model_path = p
                found = True
                break

        if not found:
            print(f"❌ Model file not found: {model_path}")
            print("Looked in:")
            print(f" - {model_path}")
            for p in alt_paths:
                print(f" - {p}")
            # List available trained_models dirs for easier debugging
            candidates = [
                os.path.join(SCRIPT_DIR, 'trained_models'),
                os.path.join(os.path.dirname(SCRIPT_DIR), 'trained_models'),
            ]
            for c in candidates:
                print(f"Contents of {c}:")
                try:
                    for f in os.listdir(c):
                        print(f"  - {f}")
                except Exception:
                    print("  (not present)")
            return

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
    # SB3 models are typically saved as .zip — include the extension for clarity
    test_model("./trained_models/best_model.zip", num_episodes=3, render=True)
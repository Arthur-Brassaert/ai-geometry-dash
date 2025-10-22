import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from geometry_dash_env import GeometryDashEnv

def simple_train():
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Testing environment creation...")
    # Test creating the environment first
    try:
        env = GeometryDashEnv(headless=True)
        obs, info = env.reset()
        print(f"Environment created successfully! Observation shape: {obs.shape}")
        env.close()
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return
    
    print("Creating training environment...")
    # Create a simple environment (no parallel environments)
    env = GeometryDashEnv(headless=True)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    print("Creating PPO model...")
    # Create a simple PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=1024,  # Smaller for testing
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        device=device,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    
    print("Starting simple training (5,000 steps)...")
    try:
        # Remove progress_bar=True to avoid the error
        model.learn(total_timesteps=5000)
        model.save("./trained_models/simple_model")
        print("Simple training completed successfully!")
        
        # Test the trained model
        print("Testing trained model...")
        test_trained_model("./trained_models/simple_model.zip")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    env.close()

def test_trained_model(model_path):
    """Test the trained model for a few episodes"""
    try:
        model = PPO.load(model_path)
        env = GeometryDashEnv(headless=False)
        
        for episode in range(3):
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 1000:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                
                # Render
                env.render()
                
            print(f"Episode {episode + 1}: Score = {info['score']}, Steps = {steps}, Total Reward = {total_reward:.2f}")
        
        env.close()
    except Exception as e:
        print(f"Model testing failed: {e}")

if __name__ == '__main__':
    simple_train()
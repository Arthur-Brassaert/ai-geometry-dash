from geometry_dash_env import GeometryDashEnv

def test_env_basic():
    print("Testing Geometry Dash Environment...")
    
    # Create environment
    env = GeometryDashEnv(headless=True)
    
    # Test reset
    print("Testing reset...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    
    # Test a few steps
    print("Testing steps...")
    for i in range(10):
        action = 0  # Don't jump
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.2f}, terminated={terminated}, score={info['score']}")
        
        if terminated:
            print("Game over!")
            break
    
    # Test with jumping
    print("Testing with jumping...")
    obs, info = env.reset()
    for i in range(20):
        # Jump every 10 frames
        action = 1 if i % 10 == 0 else 0
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.2f}, terminated={terminated}")
        
        if terminated:
            print("Game over!")
            break
    
    env.close()
    print("Environment test completed successfully!")

if __name__ == '__main__':
    test_env_basic()
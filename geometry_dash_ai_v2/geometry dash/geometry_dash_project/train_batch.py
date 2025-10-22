import subprocess
import sys

# Different training configurations
training_configs = [
    # Quick test
    {
        'name': 'quick_test',
        'timesteps': 50000,
        'n_envs': 2,
        'learning_rate': 0.0003
    },
    # Medium training
    {
        'name': 'medium_train',
        'timesteps': 200000,
        'n_envs': 4,
        'learning_rate': 0.0003
    },
    # Full training
    {
        'name': 'full_train',
        'timesteps': 500000,
        'n_envs': 4,
        'learning_rate': 0.0003
    }
]

def run_training(config):
    print(f"\n🚀 Starting training: {config['name']}")
    print(f"   Timesteps: {config['timesteps']:,}")
    print(f"   Environments: {config['n_envs']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    
    cmd = [
        sys.executable, 'train_geometry_dash.py',
        '--timesteps', str(config['timesteps']),
        '--n_envs', str(config['n_envs']),
        '--learning_rate', str(config['learning_rate']),
        '--model_name', config['name']
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Completed: {config['name']}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {config['name']} - {e}")

if __name__ == '__main__':
    print("🎯 Geometry Dash AI Training Batch")
    print("===================================")
    
    for i, config in enumerate(training_configs):
        print(f"\n{i+1}. {config['name']}")
        print(f"   Timesteps: {config['timesteps']:,}")
        print(f"   Environments: {config['n_envs']}")
    
    choice = input("\nChoose training configuration (1-3) or 'a' for all: ").strip().lower()
    
    if choice == 'a':
        for config in training_configs:
            run_training(config)
    elif choice.isdigit() and 1 <= int(choice) <= len(training_configs):
        run_training(training_configs[int(choice)-1])
    else:
        print("❌ Invalid choice")
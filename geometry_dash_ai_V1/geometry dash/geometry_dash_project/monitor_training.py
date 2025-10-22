import os
from tensorboard import program
import webbrowser
import time

def monitor_training():
    # Start TensorBoard
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', './training_logs/'])
    url = tb.launch()
    
    print(f"📊 TensorBoard started at: {url}")
    print("🔍 Monitoring training progress...")
    print("💡 Press Ctrl+C to stop monitoring")
    
    # Open in browser
    webbrowser.open(url)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping TensorBoard...")

if __name__ == '__main__':
    monitor_training()
import os
from tensorboard import program
import webbrowser
import time
from logging_config import get_tb_log_root


def monitor_training():
    # Start TensorBoard pointing at the canonical log root
    tb = program.TensorBoard()
    logdir = get_tb_log_root()
    tb.configure(argv=[None, '--logdir', logdir])
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
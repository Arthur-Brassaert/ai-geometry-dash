import subprocess
import sys
import torch

def setup_pytorch_for_cuda_12_9():
    print("🎮 Setting up PyTorch for CUDA 12.9 for RTX")
    print("==============================================")
    
    # Check current PyTorch
    print("🔍 Checking current PyTorch installation...")
    try:
        print(f"Current PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except:
        print("PyTorch not properly installed")
    
    # Uninstall current PyTorch
    print("\n🔄 Uninstalling current PyTorch...")
    subprocess.run([
        sys.executable, "-m", "pip", "uninstall", 
        "torch", "torchvision", "torchaudio", "-y"
    ], capture_output=True)
    
    # Install CUDA 12.1 PyTorch (compatible with 12.9)
    print("\n🚀 Installing PyTorch for CUDA 12.1 (compatible with 12.9)...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ CUDA 12.1 installation failed, trying CUDA 12.4...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ], capture_output=True, text=True)
    
    # Verify installation
    print("\n✅ Verifying installation...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Test GPU computation
            print("🧪 Testing GPU computation...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"✅ GPU test passed! Result shape: {z.shape}")
            
            return True
        else:
            print("❌ CUDA not available after installation")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    if setup_pytorch_for_cuda_12_9():
        print("\n🎉 SUCCESS! Your RTX  with CUDA 12.9 is ready for AI training!")
        print("🚀 You can now run: python train_rtx_3050.py")
    else:
        print("\n❌ Setup failed. Please check your NVIDIA drivers and try manual installation.")
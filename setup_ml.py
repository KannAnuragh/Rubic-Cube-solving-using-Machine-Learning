import os
import sys
import subprocess

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_step(number, text):
    """Print step number"""
    print(f"\n[{number}/6] {text}")

def check_python_version():
    """Check if Python version is sufficient"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   Requires Python 3.7+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_pytorch():
    """Install PyTorch"""
    print("\n   Installing PyTorch...")
    print("   Choose installation type:")
    print("   1. CPU only (lighter, works everywhere)")
    print("   2. GPU (CUDA) - faster training, requires NVIDIA GPU")
    
    choice = input("\n   Enter choice (1 or 2): ").strip()
    
    try:
        if choice == "2":
            print("\n   Installing PyTorch with CUDA support...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
        else:
            print("\n   Installing PyTorch (CPU version)...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio"
            ])
        print("   ✅ PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("   ❌ Failed to install PyTorch")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("\n   Installing dependencies (numpy, tqdm)...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "numpy", "tqdm"
        ])
        print("   ✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("   ❌ Failed to install dependencies")
        return False

def verify_installation():
    """Verify all packages are installed correctly"""
    print("\n   Verifying installation...")
    
    try:
        import torch
        import numpy
        import tqdm
        
        print(f"   ✅ PyTorch version: {torch.__version__}")
        print(f"   ✅ NumPy version: {numpy.__version__}")
        print(f"   ✅ tqdm installed")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("   ℹ️  CUDA not available (using CPU)")
        
        return True
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

def create_ml_files():
    """Check if ML files exist"""
    print("\n   Checking for ML files...")
    
    files = {
        'ml_solver.py': 'Main ML training and solver',
        'integrate_ml.py': 'Integration with existing project',
        'ML_README.md': 'Documentation'
    }
    
    all_exist = True
    for filename, description in files.items():
        if os.path.exists(filename):
            print(f"   ✅ {filename} - {description}")
        else:
            print(f"   ❌ {filename} missing - {description}")
            all_exist = False
    
    if not all_exist:
        print("\n   ⚠️  Some files are missing!")
        print("   Please ensure you have copied all the ML files to your project directory.")
        return False
    
    return True

def offer_training():
    """Offer to train the model"""
    print("\n   Do you want to train the ML model now?")
    print("   1. Quick demo training (5 minutes, ~30% accuracy)")
    print("   2. Full training (30-60 minutes, ~70% accuracy)")
    print("   3. Skip for now (train later)")
    
    choice = input("\n   Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        print("\n   Starting quick training...")
        print("   This will take approximately 5 minutes...")
        try:
            subprocess.check_call([
                sys.executable, "ml_solver.py",
                "--mode", "train",
                "--episodes", "1000",
                "--scramble", "3",
                "--model", "rubiks_dqn.pth"
            ])
            print("\n   ✅ Quick training complete!")
            return True
        except subprocess.CalledProcessError:
            print("\n   ❌ Training failed")
            return False
    
    elif choice == "2":
        print("\n   Starting full training...")
        print("   This will take 30-60 minutes...")
        print("   You can stop anytime with Ctrl+C (progress will be saved)")
        try:
            subprocess.check_call([
                sys.executable, "ml_solver.py",
                "--mode", "train",
                "--episodes", "10000",
                "--scramble", "7",
                "--model", "rubiks_dqn.pth"
            ])
            print("\n   ✅ Full training complete!")
            return True
        except KeyboardInterrupt:
            print("\n   ⚠️  Training interrupted (progress saved)")
            return True
        except subprocess.CalledProcessError:
            print("\n   ❌ Training failed")
            return False
    
    else:
        print("\n   ℹ️  Skipping training for now")
        print("   You can train later with:")
        print("   python ml_solver.py --mode train --episodes 1000 --scramble 3")
        return True

def test_integration():
    """Test if integration works"""
    print("\n   Testing ML integration...")
    
    try:
        from integrate_ml import MLSolverIntegration
        ml_solver = MLSolverIntegration()
        
        if ml_solver.is_model_ready():
            print("   ✅ ML model loaded and ready!")
            return True
        else:
            print("   ℹ️  ML integration works, but model not trained yet")
            return True
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False

def print_next_steps():
    """Print what to do next"""
    print_header("SETUP COMPLETE! 🎉")
    
    print("\n📝 NEXT STEPS:")
    print("\n1. Test the ML Solver:")
    print("   python ml_solver.py --mode test --scramble 3")
    
    print("\n2. Train the model (if you haven't):")
    print("   python ml_solver.py --mode train --episodes 1000 --scramble 3")
    
    print("\n3. Integrate with your main application:")
    print("   Add to train3d.py:")
    print("   ")
    print("   from integrate_ml import MLSolverIntegration")
    print("   ml_solver = MLSolverIntegration()")
    print("   ")
    print("   # In main loop, add:")
    print("   elif pr.is_key_pressed(pr.KEY_M):")
    print("       ml_moves = ml_solver.solve_with_ml(cv_solver.cube_state)")
    
    print("\n4. Read the documentation:")
    print("   See ML_README.md for detailed usage and interview prep")
    
    print("\n📊 INTERVIEW PREPARATION:")
    print("   You can now say: 'I implemented Deep Q-Learning with")
    print("   experience replay to train a neural network that learns")
    print("   to solve Rubik's Cube through reinforcement learning.'")
    
    print("\n🎓 KEY ML CONCEPTS YOU IMPLEMENTED:")
    print("   ✅ Deep Neural Networks (3-layer fully connected)")
    print("   ✅ Q-Learning (value-based RL)")
    print("   ✅ Experience Replay Memory")
    print("   ✅ Target Networks")
    print("   ✅ Epsilon-Greedy Exploration")
    print("   ✅ Reward Shaping")
    
    print("\n💡 DEMO SCRIPT FOR INTERVIEWS:")
    print("   1. Show training: python ml_solver.py --mode train --episodes 100")
    print("   2. Show testing: python ml_solver.py --mode test")
    print("   3. Show integration: Run train3d.py and press 'M' key")
    
    print("\n" + "="*70)

def main():
    """Main setup function"""
    print_header("RUBIK'S CUBE ML SETUP - Automated Installation")
    
    print("\nThis script will:")
    print("  • Check Python version")
    print("  • Install PyTorch (Deep Learning framework)")
    print("  • Install dependencies (numpy, tqdm)")
    print("  • Verify installation")
    print("  • Check ML files")
    print("  • Optionally train the model")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    if not check_python_version():
        print("\n❌ Setup failed: Python version too old")
        return False
    
    # Step 2: Install PyTorch
    print_step(2, "Installing PyTorch")
    if not install_pytorch():
        print("\n❌ Setup failed: Could not install PyTorch")
        return False
    
    # Step 3: Install dependencies
    print_step(3, "Installing dependencies")
    if not install_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        return False
    
    # Step 4: Verify installation
    print_step(4, "Verifying installation")
    if not verify_installation():
        print("\n❌ Setup failed: Verification failed")
        return False
    
    # Step 5: Check ML files
    print_step(5, "Checking ML files")
    if not create_ml_files():
        print("\n⚠️  Warning: Some ML files are missing")
        print("   Make sure you have:")
        print("   - ml_solver.py")
        print("   - integrate_ml.py")
        print("   - ML_README.md")
        choice = input("\n   Continue anyway? (y/n): ").strip().lower()
        if choice != 'y':
            return False
    
    # Step 6: Offer training
    print_step(6, "Model Training")
    offer_training()
    
    # Test integration
    test_integration()
    
    # Print next steps
    print_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        
        if success:
            print("\n✅ Setup completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Setup failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("Please report this issue or try manual installation")
        sys.exit(1)
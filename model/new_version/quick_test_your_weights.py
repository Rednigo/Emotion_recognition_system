#!/usr/bin/env python
"""
Quick test script for your specific SRGAN weights
Tests loading and creates a working model for the emotion recognition pipeline
"""

import sys
from pathlib import Path

#!/usr/bin/env python
"""
Quick test script for your specific SRGAN weights
Tests loading and creates a working model for the emotion recognition pipeline
"""

import sys
from pathlib import Path

def quick_test(weights_path):
    """Quick test of your weights file"""
    
    print("🚀 QUICK TEST FOR YOUR SRGAN WEIGHTS")
    print("=" * 50)
    
    if not Path(weights_path).exists():
        print(f"❌ Weights file not found: {weights_path}")
        return False
    
    try:
        # Import our specific loader
        from srgan_specific_loader import SRGANSpecificLoader
        
        # Create loader
        loader = SRGANSpecificLoader()
        
        # Create model from your weights
        print("🔧 Creating model from your weights...")
        model, load_method = loader.create_srgan_from_your_weights(weights_path)
        
        print(f"✅ Model created using: {load_method}")
        
        # Test functionality
        print("🧪 Testing model...")
        working_range = loader.test_model_functionality(model)
        
        if working_range:
            print(f"✅ Model works with {working_range}")
        else:
            print("⚠️  Model created but inference may have issues")
        
        # Save complete model
        save_path = "srgan_ready_for_emotion_recognition.h5"
        print(f"💾 Saving complete model to: {save_path}")
        
        success = loader.save_complete_model(model, save_path)
        
        if success:
            print("\n🎉 SUCCESS! Your SRGAN model is ready!")
            print("\n📋 Next steps:")
            print("1. Test the complete pipeline:")
            print(f"   python updated_complete_pipeline.py {save_path} /path/to/raf_db_dataset --test-mode --visualization")
            print("\n2. Full training:")
            print(f"   python updated_complete_pipeline.py {save_path} /path/to/raf_db_dataset --visualization")
            
            return True
        else:
            print("❌ Failed to save complete model")
            return False
            
    except ImportError:
        print("❌ Could not import SRGAN specific loader")
        print("Make sure srgan_specific_loader.py is in the same directory")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quick_test_your_weights.py your_weights.h5")
        sys.exit(1)
    
    weights_path = sys.argv[1]
    success = quick_test(weights_path)
    
    if success:
        print("\n✅ Ready for emotion recognition training!")
        sys.exit(0)
    else:
        print("\n❌ Issues detected - check the output above")
        sys.exit(1)
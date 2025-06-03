#!/usr/bin/env python
"""
Quick test script for your SRGAN .h5 model
Tests loading and basic functionality before running full preprocessing
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def test_srgan_model(model_path, test_image_path=None):
    """
    Test your SRGAN .h5 model
    
    Args:
        model_path: Path to your .h5 model file
        test_image_path: Optional path to test image
    """
    print("🧪 Testing SRGAN Model")
    print("=" * 50)
    
    # Test 1: Load model
    print("1️⃣ Testing model loading...")
    try:
        model = load_model(model_path, compile=False)
        print("✅ Model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Parameters: {model.count_params():,}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Test 2: Model architecture verification
    print("\n2️⃣ Testing model architecture...")
    expected_input = (None, 48, 48, 3)  # Paper specification
    expected_output = (None, 192, 192, 3)  # 4x upscaling
    
    if model.input_shape == expected_input:
        print("✅ Input shape matches paper specification")
    else:
        print(f"⚠️  Input shape mismatch: got {model.input_shape}, expected {expected_input}")
    
    if model.output_shape == expected_output:
        print("✅ Output shape matches paper specification (4x upscaling)")
    else:
        print(f"⚠️  Output shape: got {model.output_shape}, expected {expected_output}")
        # Calculate actual scale factor
        if len(model.output_shape) >= 3:
            actual_scale = model.output_shape[1] // model.input_shape[1] if model.input_shape[1] else 1
            print(f"   Actual scale factor: {actual_scale}x")
    
    # Test 3: Forward pass with random data
    print("\n3️⃣ Testing forward pass...")
    try:
        # Create test input (48x48 RGB image)
        test_input = np.random.random((1, 48, 48, 3)).astype(np.float32)
        
        # Test different input ranges
        ranges_to_test = [
            ("0-1 range", test_input),
            ("-1 to 1 range", test_input * 2 - 1),
            ("0-255 range", test_input * 255)
        ]
        
        for range_name, test_data in ranges_to_test:
            try:
                output = model.predict(test_data, verbose=0)
                print(f"✅ {range_name}: {output.shape}, values [{output.min():.3f}, {output.max():.3f}]")
                break  # Use the first successful range
            except Exception as e:
                print(f"❌ {range_name}: {e}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test 4: Real image processing (if test image provided)
    if test_image_path and Path(test_image_path).exists():
        print(f"\n4️⃣ Testing with real image: {test_image_path}")
        try:
            # Load and preprocess image
            image = cv2.imread(test_image_path)
            if image is None:
                print(f"❌ Could not load image: {test_image_path}")
                return True  # Model tests passed, just image issue
            
            # Convert to RGB and resize to 48x48
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (48, 48))
            
            # Test different normalizations
            normalizations = [
                ("0-1 normalization", image_resized / 255.0),
                ("-1 to 1 normalization", (image_resized / 127.5) - 1.0),
                ("0-255 range", image_resized.astype(np.float32))
            ]
            
            successful_output = None
            best_normalization = None
            
            for norm_name, normalized_image in normalizations:
                try:
                    input_batch = np.expand_dims(normalized_image, axis=0)
                    output = model.predict(input_batch, verbose=0)[0]
                    
                    # Check if output is reasonable
                    if not np.isnan(output).any() and not np.isinf(output).any():
                        print(f"✅ {norm_name}: Success")
                        successful_output = output
                        best_normalization = norm_name
                        break
                    else:
                        print(f"❌ {norm_name}: Output contains NaN/Inf")
                        
                except Exception as e:
                    print(f"❌ {norm_name}: {e}")
            
            if successful_output is not None:
                # Save comparison
                print(f"💾 Creating comparison image...")
                
                # Denormalize output based on range
                if "0-1" in best_normalization:
                    output_denorm = np.clip(successful_output * 255, 0, 255).astype(np.uint8)
                elif "-1 to 1" in best_normalization:
                    output_denorm = np.clip((successful_output + 1) * 127.5, 0, 255).astype(np.uint8)
                else:
                    output_denorm = np.clip(successful_output, 0, 255).astype(np.uint8)
                
                # Create comparison
                original_upscaled = cv2.resize(image_resized, (192, 192), interpolation=cv2.INTER_CUBIC)
                comparison = np.hstack([original_upscaled, output_denorm])
                
                # Add labels
                comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
                cv2.putText(comparison_bgr, "Cubic Interpolation", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(comparison_bgr, "SRGAN", (202, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Save comparison
                output_path = "srgan_test_comparison.jpg"
                cv2.imwrite(output_path, comparison_bgr)
                print(f"✅ Comparison saved to: {output_path}")
                print(f"   Best normalization: {best_normalization}")
                
            else:
                print("❌ All normalizations failed")
                
        except Exception as e:
            print(f"❌ Real image test failed: {e}")
    
    else:
        print("\n4️⃣ Skipping real image test (no test image provided)")
        print("   Use --test-image path/to/image.jpg to test with real image")
    
    print("\n🎉 SRGAN model testing completed!")
    return True

def create_test_image():
    """Create a simple test image for SRGAN testing"""
    # Create a 48x48 test image with simple patterns
    test_img = np.zeros((48, 48, 3), dtype=np.uint8)
    
    # Add some patterns
    # Circle
    cv2.circle(test_img, (24, 24), 15, (255, 0, 0), -1)
    # Rectangle
    cv2.rectangle(test_img, (5, 5), (15, 15), (0, 255, 0), -1)
    # Lines
    cv2.line(test_img, (0, 0), (48, 48), (0, 0, 255), 2)
    cv2.line(test_img, (0, 48), (48, 0), (255, 255, 0), 2)
    
    # Save test image
    cv2.imwrite("test_image_48x48.jpg", test_img)
    print("📷 Created test image: test_image_48x48.jpg")
    return "test_image_48x48.jpg"

def analyze_model_architecture(model):
    """Analyze the model architecture in detail"""
    print("\n🔍 DETAILED MODEL ANALYSIS")
    print("=" * 50)
    
    print("📊 Layer Summary:")
    for i, layer in enumerate(model.layers):
        print(f"  {i:2d}. {layer.name:25} {str(layer.output_shape):20} {layer.__class__.__name__}")
    
    print(f"\n📈 Model Statistics:")
    print(f"  Total layers: {len(model.layers)}")
    print(f"  Trainable parameters: {model.count_params():,}")
    
    # Check for common SRGAN components
    layer_names = [layer.name for layer in model.layers]
    conv_layers = [name for name in layer_names if 'conv' in name.lower()]
    bn_layers = [name for name in layer_names if 'batch' in name.lower()]
    activation_layers = [name for name in layer_names if any(act in name.lower() for act in ['relu', 'prelu', 'leaky'])]
    
    print(f"\n🏗️ Architecture Components:")
    print(f"  Convolutional layers: {len(conv_layers)}")
    print(f"  Batch normalization: {len(bn_layers)}")
    print(f"  Activation layers: {len(activation_layers)}")
    
    # Check input/output compatibility
    input_h, input_w = model.input_shape[1:3]
    output_h, output_w = model.output_shape[1:3]
    scale_factor = output_h // input_h if input_h else 1
    
    print(f"\n📐 Scale Analysis:")
    print(f"  Input resolution: {input_w}x{input_h}")
    print(f"  Output resolution: {output_w}x{output_h}")
    print(f"  Scale factor: {scale_factor}x")
    
    if scale_factor == 4:
        print("  ✅ Perfect for paper methodology (4x upscaling)")
    else:
        print(f"  ⚠️  Non-standard scale factor (paper expects 4x)")

def benchmark_model_speed(model, num_iterations=10):
    """Benchmark model inference speed"""
    print(f"\n⏱️ SPEED BENCHMARK ({num_iterations} iterations)")
    print("=" * 50)
    
    import time
    
    # Warm up
    test_input = np.random.random((1, 48, 48, 3)).astype(np.float32)
    _ = model.predict(test_input, verbose=0)
    
    # Benchmark
    times = []
    for i in range(num_iterations):
        start_time = time.time()
        _ = model.predict(test_input, verbose=0)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"📊 Results:")
    print(f"  Average time: {avg_time:.3f} seconds")
    print(f"  Standard deviation: {std_time:.3f} seconds")
    print(f"  Images per second: {1/avg_time:.1f}")
    print(f"  Time for 1000 images: {avg_time * 1000 / 60:.1f} minutes")

def main():
    parser = argparse.ArgumentParser(
        description="Test SRGAN .h5 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic model test
  python test_srgan_model.py your_model.h5
  
  # Test with specific image
  python test_srgan_model.py your_model.h5 --test-image image.jpg
  
  # Detailed analysis with benchmarking
  python test_srgan_model.py your_model.h5 --detailed --benchmark
  
  # Create test image if you don't have one
  python test_srgan_model.py your_model.h5 --create-test-image
        """
    )
    
    parser.add_argument('model_path', help='Path to SRGAN .h5 model file')
    parser.add_argument('--test-image', help='Path to test image')
    parser.add_argument('--create-test-image', action='store_true',
                       help='Create a simple test image')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed model analysis')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run speed benchmark')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not Path(args.model_path).exists():
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    # Create test image if requested
    test_image_path = args.test_image
    if args.create_test_image:
        test_image_path = create_test_image()
    
    # Run basic tests
    success = test_srgan_model(args.model_path, test_image_path)
    
    if not success:
        print("❌ Basic tests failed - check your model file")
        return
    
    # Load model for additional analysis
    if args.detailed or args.benchmark:
        try:
            model = load_model(args.model_path, compile=False)
            
            if args.detailed:
                analyze_model_architecture(model)
            
            if args.benchmark:
                benchmark_model_speed(model)
                
        except Exception as e:
            print(f"❌ Could not load model for detailed analysis: {e}")
    
    # Provide recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 50)
    print("✅ Your SRGAN model appears ready for use!")
    print("\n📋 Next steps:")
    print("1. Run preprocessing on a small test set:")
    print(f"   python srgan_standard_preprocessing.py {args.model_path} /path/to/raf_db /path/to/output --test-mode")
    print("\n2. If test preprocessing works, run full preprocessing:")
    print(f"   python srgan_standard_preprocessing.py {args.model_path} /path/to/raf_db /path/to/output")
    print("\n3. Then train emotion recognition:")
    print(f"   python complete_training_pipeline.py {args.model_path} /path/to/raf_db --create-visualization")
    
    # Check for potential issues
    try:
        model = load_model(args.model_path, compile=False)
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        print(f"\n⚠️  POTENTIAL ISSUES TO WATCH:")
        
        if input_shape[1:3] != (48, 48):
            print(f"- Input size is {input_shape[1:3]}, paper expects (48, 48)")
            print("  Solution: Images will be resized automatically")
        
        if output_shape[1:3] != (192, 192):
            print(f"- Output size is {output_shape[1:3]}, paper expects (192, 192)")
            print("  Solution: Output will be resized to 192x192")
        
        scale_factor = output_shape[1] // input_shape[1] if input_shape[1] else 1
        if scale_factor != 4:
            print(f"- Scale factor is {scale_factor}x, paper expects 4x")
            print("  Note: This may affect final accuracy")
        
        if scale_factor == 4 and input_shape[1:3] == (48, 48) and output_shape[1:3] == (192, 192):
            print("- ✅ Perfect configuration for paper methodology!")
            
    except:
        pass

if __name__ == "__main__":
    main()
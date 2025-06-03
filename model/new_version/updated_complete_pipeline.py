#!/usr/bin/env python
"""
Updated Complete Training Pipeline
Works with SRGAN weights files (.h5) instead of complete models
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

# Import our modules
try:
    from srgan_specific_loader import SRGANSpecificLoader
    from emotion_system_paper_compliant import PaperCompliantEmotionSystem
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required modules are in the same directory")
    print("Required files:")
    print("  - srgan_specific_loader.py")
    print("  - emotion_system_paper_compliant.py") 
    sys.exit(1)

class EmotionRecognitionPipeline:
    """
    Complete pipeline for emotion recognition with SRGAN weights
    """
    
    def __init__(self, srgan_weights_path, output_dir):
        self.srgan_weights_path = Path(srgan_weights_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.srgan_loader = SRGANSpecificLoader()
        self.emotion_system = None
        self.srgan_model = None
        
        # Logging
        self.log_file = self.output_dir / f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
    def log_and_print(self, message):
        """Log message to both console and file"""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now()}: {message}\n")
    
    def step1_prepare_srgan(self):
        """Step 1: Load SRGAN weights and create complete model"""
        self.log_and_print("\n" + "="*60)
        self.log_and_print("STEP 1: PREPARING SRGAN MODEL")
        self.log_and_print("="*60)
        
        if not self.srgan_weights_path.exists():
            raise FileNotFoundError(f"SRGAN weights not found: {self.srgan_weights_path}")
        
        # Check if we already have a complete model saved
        complete_model_path = self.output_dir / "srgan_complete.h5"
        
        if complete_model_path.exists():
            self.log_and_print(f"✅ Found existing complete model: {complete_model_path}")
            response = input("Use existing complete model? (y/n): ")
            if response.lower() == 'y':
                try:
                    from tensorflow.keras.models import load_model
                    self.srgan_model = load_model(complete_model_path, compile=False)
                    self.log_and_print("✅ Loaded existing complete model")
                    return str(complete_model_path)
                except Exception as e:
                    self.log_and_print(f"⚠️  Failed to load existing model: {e}")
                    self.log_and_print("🔄 Creating new model...")
        
        # Create model from weights
        try:
            self.log_and_print(f"🔧 Loading SRGAN weights: {self.srgan_weights_path}")
            
            # Inspect weights first
            structure = self.srgan_loader.analyze_weights_structure(str(self.srgan_weights_path))
            
            if structure is None:
                raise ValueError("Could not analyze weights file structure")
            
            # Create model
            self.srgan_model, load_method = self.srgan_loader.create_srgan_from_your_weights(
                str(self.srgan_weights_path)
            )
            
            self.log_and_print(f"✅ SRGAN model created using: {load_method}")
            
            # Test inference
            working_range = self.srgan_loader.test_model_functionality(self.srgan_model)
            
            if working_range:
                self.log_and_print(f"✅ Model inference test passed (best range: {working_range})")
            else:
                self.log_and_print("⚠️  Model inference test failed - proceeding anyway")
            
            # Save complete model for future use
            if self.srgan_loader.save_complete_model(self.srgan_model, str(complete_model_path)):
                self.log_and_print(f"💾 Complete model saved to: {complete_model_path}")
                return str(complete_model_path)
            else:
                self.log_and_print("⚠️  Could not save complete model - using temporary model")
                return "temporary_model"
            
        except Exception as e:
            self.log_and_print(f"❌ Failed to create SRGAN model: {e}")
            raise
    
    def step2_preprocess_dataset(self, dataset_path, test_mode=False):
        """Step 2: Preprocess dataset with SRGAN"""
        self.log_and_print("\n" + "="*60)
        self.log_and_print("STEP 2: SRGAN PREPROCESSING")
        self.log_and_print("="*60)
        
        # Set up paths
        preprocessed_path = self.output_dir / "preprocessed_dataset"
        
        # Check if already preprocessed
        if preprocessed_path.exists() and any(preprocessed_path.iterdir()):
            self.log_and_print(f"📁 Preprocessed data exists: {preprocessed_path}")
            response = input("Use existing preprocessed data? (y/n): ")
            if response.lower() == 'y':
                self.log_and_print("✅ Using existing preprocessed data")
                return str(preprocessed_path)
            else:
                import shutil
                shutil.rmtree(preprocessed_path)
                self.log_and_print("🗑️  Removed existing preprocessed data")
        
        # Run preprocessing
        if self.srgan_model is None:
            raise ValueError("SRGAN model not initialized")
        
        self.log_and_print(f"🔄 Starting SRGAN preprocessing...")
        self.log_and_print(f"   Input dataset: {dataset_path}")
        self.log_and_print(f"   Output: {preprocessed_path}")
        self.log_and_print(f"   Mode: {'TEST' if test_mode else 'FULL'}")
        
        # Create custom preprocessor with our loaded model
        from srgan_standard_preprocessing import SRGANStandardPreprocessor
        
        # Create a temporary complete model file if needed
        temp_model_path = self.output_dir / "temp_srgan.h5"
        try:
            self.srgan_model.save(temp_model_path)
            
            # Initialize preprocessor with the complete model
            preprocessor = SRGANStandardPreprocessor.__new__(SRGANStandardPreprocessor)
            preprocessor.input_size = (48, 48)
            preprocessor.output_size = (192, 192)
            preprocessor.scale_factor = 4
            preprocessor.srgan_model = self.srgan_model
            
            # Run preprocessing
            start_time = time.time()
            stats = preprocessor.preprocess_dataset(
                str(dataset_path),
                str(preprocessed_path),
                test_mode=test_mode
            )
            
            preprocessing_time = time.time() - start_time
            self.log_and_print(f"✅ Preprocessing completed in {preprocessing_time:.1f} seconds")
            self.log_and_print(f"📊 Processed: {stats['total_processed']} images")
            self.log_and_print(f"❌ Failed: {stats['total_failed']} images")
            
            return str(preprocessed_path)
            
        except Exception as e:
            self.log_and_print(f"❌ Preprocessing failed: {e}")
            raise
        finally:
            # Clean up temp file
            if temp_model_path.exists():
                temp_model_path.unlink()
    
    def step3_train_emotion_recognition(self, preprocessed_path, max_samples=None):
        """Step 3: Train emotion recognition models"""
        self.log_and_print("\n" + "="*60)
        self.log_and_print("STEP 3: EMOTION RECOGNITION TRAINING")
        self.log_and_print("="*60)
        
        # Initialize emotion recognition system
        try:
            self.emotion_system = PaperCompliantEmotionSystem(verbose=True)
            self.log_and_print("✅ Emotion recognition system initialized")
        except Exception as e:
            self.log_and_print(f"❌ Failed to initialize emotion system: {e}")
            raise
        
        # Validate implementation
        if self.emotion_system.validate_implementation():
            self.log_and_print("✅ Implementation validated - matches paper specifications")
        else:
            self.log_and_print("⚠️  Implementation validation warnings")
        
        # Load dataset
        self.log_and_print(f"📂 Loading preprocessed dataset: {preprocessed_path}")
        start_time = time.time()
        
        try:
            X_train, X_test, y_train, y_test = self.emotion_system.load_preprocessed_dataset(
                preprocessed_path,
                max_samples_per_emotion=max_samples
            )
            
            loading_time = time.time() - start_time
            self.log_and_print(f"✅ Dataset loaded in {loading_time:.1f} seconds")
            self.log_and_print(f"📊 Training samples: {X_train.shape[0]}")
            self.log_and_print(f"📊 Test samples: {X_test.shape[0]}")
            self.log_and_print(f"📊 Features: {X_train.shape[1]} angular features")
            
        except Exception as e:
            self.log_and_print(f"❌ Failed to load dataset: {e}")
            raise
        
        # Train models
        self.log_and_print(f"🎯 Training models with paper hyperparameters...")
        start_time = time.time()
        
        try:
            results = self.emotion_system.train_models_with_paper_hyperparameters(
                X_train, X_test, y_train, y_test
            )
            
            training_time = time.time() - start_time
            self.log_and_print(f"✅ Training completed in {training_time:.1f} seconds")
            
            return results
            
        except Exception as e:
            self.log_and_print(f"❌ Training failed: {e}")
            raise
    
    def step4_analyze_results(self, results):
        """Step 4: Analyze and save results"""
        self.log_and_print("\n" + "="*60)
        self.log_and_print("STEP 4: RESULTS ANALYSIS")
        self.log_and_print("="*60)
        
        # Sort results by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        # Display results
        self.log_and_print("\n🏆 MODEL PERFORMANCE RESULTS:")
        self.log_and_print("-" * 40)
        
        for i, (model_name, result) in enumerate(sorted_results):
            accuracy = result['accuracy']
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            self.log_and_print(f"{medal} {model_name:8}: {accuracy:.4f}")
        
        # Best model analysis
        best_model_name, best_result = sorted_results[0]
        best_accuracy = best_result['accuracy']
        
        self.log_and_print(f"\n🎉 Best performing model: {best_model_name}")
        self.log_and_print(f"🎯 Best accuracy: {best_accuracy:.4f}")
        
        # Paper comparison
        self.log_and_print(f"\n📄 Paper comparison:")
        if best_accuracy >= 0.70:
            self.log_and_print("🌟 EXCELLENT: Results exceed paper expectations!")
        elif best_accuracy >= 0.60:
            self.log_and_print("✅ GOOD: Results align with paper expectations (60-70%)")
        elif best_accuracy >= 0.50:
            self.log_and_print("⚠️  FAIR: Results below paper expectations but reasonable")
        else:
            self.log_and_print("❌ POOR: Results significantly below expectations")
        
        # Per-emotion analysis
        self.log_and_print(f"\n📊 Per-emotion performance (F1-scores):")
        emotion_names = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
        best_report = best_result['report']
        
        for emotion in emotion_names:
            if emotion in best_report:
                f1_score = best_report[emotion]['f1-score']
                precision = best_report[emotion]['precision']
                recall = best_report[emotion]['recall']
                self.log_and_print(f"  {emotion:12}: F1={f1_score:.3f}, P={precision:.3f}, R={recall:.3f}")
        
        return best_model_name, best_accuracy
    
    def step5_save_system(self, results, create_visualization=False):
        """Step 5: Save complete trained system"""
        self.log_and_print("\n" + "="*60)
        self.log_and_print("STEP 5: SAVING TRAINED SYSTEM")
        self.log_and_print("="*60)
        
        # Save emotion recognition model
        model_dir = self.output_dir / "emotion_model"
        try:
            model_save_path = self.emotion_system.save_best_model(str(model_dir))
            self.log_and_print(f"✅ Emotion model saved to: {model_save_path}")
        except Exception as e:
            self.log_and_print(f"❌ Failed to save emotion model: {e}")
        
        # Create visualization if requested
        if create_visualization:
            self.log_and_print(f"📊 Creating result visualizations...")
            try:
                viz_path = self.output_dir / "results_visualization.png"
                self.emotion_system.create_results_visualization(results, save_path=str(viz_path))
                self.log_and_print(f"✅ Visualization saved to: {viz_path}")
            except Exception as e:
                self.log_and_print(f"❌ Visualization failed: {e}")
        
        # Create comprehensive report
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        
        training_report = {
            "pipeline_info": {
                "timestamp": datetime.now().isoformat(),
                "srgan_weights": str(self.srgan_weights_path),
                "output_directory": str(self.output_dir)
            },
            "methodology": {
                "preprocessing": "SRGAN weights -> complete model -> 4x upscaling (48x48 -> 192x192)",
                "landmark_detection": "MediaPipe Face Mesh (468 landmarks)",
                "feature_extraction": "27 key landmarks -> 10 angular features",
                "classification": "8 ML models with paper hyperparameters"
            },
            "results": {
                "best_model": best_model_name,
                "best_accuracy": float(best_accuracy),
                "all_results": {name: float(result['accuracy']) for name, result in results.items()}
            },
            "paper_compliance": {
                "weights_loading": "Custom SRGAN architecture with loaded weights",
                "preprocessing_method": "SRGAN 4x super-resolution",
                "feature_extraction": "Exact paper methodology",
                "classification": "Exact paper hyperparameters"
            }
        }
        
        report_path = self.output_dir / "complete_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(training_report, f, indent=2)
        
        self.log_and_print(f"📄 Training report saved to: {report_path}")
        
        return model_save_path
    
    def run_complete_pipeline(self, dataset_path, test_mode=False, max_samples=None, create_visualization=False):
        """Run the complete pipeline from weights to trained system"""
        total_start_time = time.time()
        
        self.log_and_print("🚀 STARTING COMPLETE EMOTION RECOGNITION PIPELINE")
        self.log_and_print("=" * 80)
        self.log_and_print(f"📁 SRGAN weights: {self.srgan_weights_path}")
        self.log_and_print(f"📁 Dataset: {dataset_path}")
        self.log_and_print(f"📁 Output: {self.output_dir}")
        self.log_and_print(f"🎯 Mode: {'TEST' if test_mode else 'FULL'}")
        
        try:
            # Step 1: Prepare SRGAN
            srgan_model_path = self.step1_prepare_srgan()
            
            # Step 2: Preprocess dataset
            preprocessed_path = self.step2_preprocess_dataset(dataset_path, test_mode)
            
            # Step 3: Train emotion recognition
            results = self.step3_train_emotion_recognition(preprocessed_path, max_samples)
            
            # Step 4: Analyze results
            best_model_name, best_accuracy = self.step4_analyze_results(results)
            
            # Step 5: Save system
            model_save_path = self.step5_save_system(results, create_visualization)
            
            # Final summary
            total_time = time.time() - total_start_time
            
            self.log_and_print("\n" + "="*80)
            self.log_and_print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            self.log_and_print("="*80)
            self.log_and_print(f"✅ Best model: {best_model_name} ({best_accuracy:.4f})")
            self.log_and_print(f"⏰ Total time: {total_time/60:.1f} minutes")
            self.log_and_print(f"📁 Results saved to: {self.output_dir}")
            
            # Usage instructions
            self.log_and_print(f"\n💡 USAGE INSTRUCTIONS:")
            self.log_and_print(f"📱 For Android integration:")
            self.log_and_print(f"   1. Use MediaPipe landmarks (same as your current code)")
            self.log_and_print(f"   2. Select 27 key landmarks using paper specification")
            self.log_and_print(f"   3. Calculate 10 angular features")
            self.log_and_print(f"   4. Send features to server with trained model")
            
            self.log_and_print(f"\n🔧 Model files:")
            self.log_and_print(f"   📄 Emotion classifier: {model_save_path}/model.pkl")
            self.log_and_print(f"   📄 Feature scaler: {model_save_path}/scaler.pkl")
            self.log_and_print(f"   📄 PCA transformer: {model_save_path}/pca.pkl")
            
            return {
                'success': True,
                'best_model': best_model_name,
                'best_accuracy': best_accuracy,
                'model_path': model_save_path,
                'total_time': total_time
            }
            
        except Exception as e:
            self.log_and_print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'total_time': time.time() - total_start_time
            }

def main():
    parser = argparse.ArgumentParser(
        description="Complete emotion recognition pipeline with SRGAN weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with SRGAN weights
  python updated_complete_pipeline.py weights.h5 /path/to/raf_db --test-mode --visualization
  
  # Full training pipeline
  python updated_complete_pipeline.py weights.h5 /path/to/raf_db --output-dir my_emotion_system
  
  # Limited training data
  python updated_complete_pipeline.py weights.h5 /path/to/raf_db --max-samples 500 --visualization
        """
    )
    
    parser.add_argument('srgan_weights', help='Path to SRGAN weights .h5 file')
    parser.add_argument('dataset_path', help='Path to RAF-DB dataset')
    parser.add_argument('--output-dir', default='emotion_recognition_system',
                       help='Output directory for trained system')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: process only 5 images per emotion')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per emotion for training')
    parser.add_argument('--visualization', action='store_true',
                       help='Create result visualizations')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.srgan_weights).exists():
        print(f"❌ SRGAN weights file not found: {args.srgan_weights}")
        return
    
    if not Path(args.dataset_path).exists():
        print(f"❌ Dataset path not found: {args.dataset_path}")
        return
    
    # Create and run pipeline
    pipeline = EmotionRecognitionPipeline(args.srgan_weights, args.output_dir)
    
    result = pipeline.run_complete_pipeline(
        dataset_path=args.dataset_path,
        test_mode=args.test_mode,
        max_samples=args.max_samples,
        create_visualization=args.visualization
    )
    
    # Exit with appropriate code
    if result['success']:
        print(f"\n🎉 Success! Best accuracy: {result['best_accuracy']:.4f}")
        if result['best_accuracy'] >= 0.60:
            print("✅ Results meet paper expectations!")
        sys.exit(0)
    else:
        print(f"\n❌ Pipeline failed: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()